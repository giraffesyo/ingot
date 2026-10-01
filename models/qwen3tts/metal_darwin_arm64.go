//go:build darwin && arm64

package qwen3tts

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/safetensors"
)

func metalAvailable() bool { return metal.Available() }

// metalWeight is one Linear's [n, k] weight on the GPU: the checkpoint's
// bf16 bytes wrapped in place, plus an int8 copy (scales per 64) when the
// decode loop runs quantised.
type metalWeight struct {
	n, k int
	bf16 metal.Region
	q, s *metal.Buffer
}

type metalLayer struct {
	q, k, v, o, gate, up, down     *metalWeight
	inNorm, postNorm, qNorm, kNorm *metal.Buffer
	qkv                            metal.DecodeWeight // [q; k; v] rows for the fused decode GEMV
}

// dec is w in the fused decode kernels' form: its int8 copy, else the
// bf16 bytes in place.
func (w *metalWeight) dec() metal.DecodeWeight {
	if w.q != nil {
		return metal.DecodeWeight{W: w.q.At(0), S: w.s.At(0)}
	}
	return metal.DecodeWeight{W: w.bf16, BF16: true}
}

// metalLM runs a Qwen3 decoder stack (talker or code predictor) on the GPU
// with its KV cache resident there: one command buffer per run. Prefill
// (more than four rows) runs the bf16 GEMM; decode runs the GEMV kernels
// on bf16 or int8 weights, the residual adds folded into them.
type metalLM struct {
	dev     *metal.Device
	c       LMConfig
	in      int // input width (before proj)
	shards  map[unsafe.Pointer]*metal.Buffer
	layers  []metalLayer
	norm    *metal.Buffer
	heads   []*metalWeight
	proj    *metalWeight
	projB   *metal.Buffer
	kc, vc  []*metal.Buffer
	cos     *metal.Buffer
	sin     *metal.Buffer
	ones    *metal.Buffer
	xin, x  *metal.Buffer
	y, q, o *metal.Buffer
	g, u    *metal.Buffer
	hid     *metal.Buffer
	logits  *metal.Buffer
	maxT    int
	maxRows int
	int8    bool
	p       int // cached positions
	owned   []*metal.Buffer

	codes, allLogits *metal.Buffer // runFrame: sampled codes, each step's logits
	tables           []metal.Region
}

// newMetalLM maps prefix's decoder layers onto the GPU. heads are the
// output projections run() can apply (index); proj/projBias name an input
// Linear (empty for none).
func newMetalLM(set *safetensors.Set, c LMConfig, prefix string, in int, heads []string, proj string, maxT, maxRows int, q Quant) (m *metalLM, err error) {
	dev, err := metal.Open()
	if err != nil {
		return nil, err
	}
	if err := dev.Prepare(); err != nil {
		return nil, err
	}
	if err := dev.PrepareDecode(); err != nil {
		return nil, err
	}
	if err := dev.PrepareConv(); err != nil { // AddBias
		return nil, err
	}
	lm := &metalLM{dev: dev, c: c, in: in, shards: map[unsafe.Pointer]*metal.Buffer{}, maxT: maxT, maxRows: maxRows, int8: q == Int8}
	m = lm
	defer func() {
		if err != nil {
			lm.close()
		}
	}()
	D, H, KV, dh, I := c.HiddenSize, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim, c.IntermediateSize
	if dh > 128 || dh%32 != 0 {
		return nil, fmt.Errorf("qwen3tts: GPU decode needs head_dim a multiple of 32 up to 128, have %d", dh)
	}
	for i := range c.NumHiddenLayers {
		p := fmt.Sprintf("%slayers.%d.", prefix, i)
		var l metalLayer
		for _, w := range []struct {
			name    string
			dst     **metalWeight
			out, in int
			quant   bool
		}{
			{"self_attn.q_proj.weight", &l.q, H * dh, D, false}, {"self_attn.k_proj.weight", &l.k, KV * dh, D, false},
			{"self_attn.v_proj.weight", &l.v, KV * dh, D, false}, {"self_attn.o_proj.weight", &l.o, D, H * dh, true},
			{"mlp.gate_proj.weight", &l.gate, I, D, true}, {"mlp.up_proj.weight", &l.up, I, D, true},
			{"mlp.down_proj.weight", &l.down, D, I, true},
		} {
			if *w.dst, err = m.weightQ(set, p+w.name, w.out, w.in, w.quant); err != nil {
				return nil, err
			}
		}
		if l.qkv, err = m.concatQKV(set, []string{p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"}, D); err != nil {
			return nil, err
		}
		for _, n := range []struct {
			name string
			dst  **metal.Buffer
		}{{"input_layernorm.weight", &l.inNorm}, {"post_attention_layernorm.weight", &l.postNorm},
			{"self_attn.q_norm.weight", &l.qNorm}, {"self_attn.k_norm.weight", &l.kNorm}} {
			if *n.dst, err = m.f32(set, p+n.name); err != nil {
				return nil, err
			}
		}
		m.layers = append(m.layers, l)
	}
	if m.norm, err = m.f32(set, prefix+"norm.weight"); err != nil {
		return nil, err
	}
	for _, h := range heads {
		info, ok := set.Info(h)
		if !ok || len(info.Shape) != 2 {
			return nil, fmt.Errorf("qwen3tts: head %s missing", h)
		}
		w, err := m.weight(set, h, info.Shape[0], info.Shape[1])
		if err != nil {
			return nil, err
		}
		m.heads = append(m.heads, w)
	}
	if proj != "" {
		if m.proj, err = m.weight(set, proj+".weight", D, in); err != nil {
			return nil, err
		}
		if m.projB, err = m.f32(set, proj+".bias"); err != nil {
			return nil, err
		}
	}
	nb := func(floats int) *metal.Buffer {
		if err != nil {
			return nil
		}
		var b *metal.Buffer
		if b, err = dev.NewBuffer(4 * max(floats, 1)); err == nil {
			m.owned = append(m.owned, b)
		}
		return b
	}
	R := maxRows
	m.xin, m.x, m.y = nb(R*in), nb(R*D), nb(R*max(D, I))
	m.q, m.o, m.g, m.u = nb(R*H*dh), nb(R*H*dh), nb(R*I), nb(R*I)
	m.hid, m.ones = nb(D), nb(max(D, I))
	maxHead := 1
	for _, h := range m.heads {
		maxHead = max(maxHead, h.n)
	}
	m.logits = nb(maxHead)
	for range c.NumHiddenLayers {
		m.kc, m.vc = append(m.kc, nb(maxT*KV*dh)), append(m.vc, nb(maxT*KV*dh))
	}
	m.cos, m.sin = nb(maxT*dh/2), nb(maxT*dh/2)
	if err != nil {
		return nil, err
	}
	for i := range f32view(m.ones) {
		f32view(m.ones)[i] = 1
	}
	r := newRope(c.RopeTheta, dh, maxT)
	copy(f32view(m.cos), r.cos)
	copy(f32view(m.sin), r.sin)
	return m, nil
}

// weight wraps name's bf16 bytes in place (its shard once) and, for int8
// decode, quantises a copy.
func (m *metalLM) weight(set *safetensors.Set, name string, out, in int) (*metalWeight, error) {
	return m.weightQ(set, name, out, in, true)
}

// weightQ is weight with the int8 copy optional (quant).
func (m *metalLM) weightQ(set *safetensors.Set, name string, out, in int, quant bool) (*metalWeight, error) {
	r, err := m.weightBF16(set, name, out, in)
	if err != nil {
		return nil, err
	}
	w := &metalWeight{n: out, k: in, bf16: r}
	if quant && m.int8 && in%64 == 0 {
		mem, off, _, _ := set.Locate(name)
		bits := unsafe.Slice((*uint16)(unsafe.Pointer(&mem[off])), out*in)
		qw := gemm.QuantizeI8GBF16(bits, in, out, in)
		if w.q, err = m.dev.NewBuffer(len(qw.Q)); err != nil {
			return nil, err
		}
		m.owned = append(m.owned, w.q)
		copy(unsafe.Slice((*int8)(unsafe.Pointer(&w.q.Bytes()[0])), len(qw.Q)), qw.Q)
		if w.s, err = m.dev.NewBuffer(4 * len(qw.S)); err != nil {
			return nil, err
		}
		m.owned = append(m.owned, w.s)
		copy(f32view(w.s), qw.S)
	}
	return w, nil
}

// concatQKV builds the decode weight with the rows of names stacked (q,
// k, v projections, each [*, in]): int8 when quantising, else a bf16 copy
// (the checkpoint stores them apart).
func (m *metalLM) concatQKV(set *safetensors.Set, names []string, in int) (metal.DecodeWeight, error) {
	var rows int
	var bits [][]uint16
	for _, name := range names {
		mem, off, info, err := set.Locate(name)
		if err != nil {
			return metal.DecodeWeight{}, err
		}
		if info.DType != "BF16" || len(info.Shape) != 2 || info.Shape[1] != in {
			return metal.DecodeWeight{}, fmt.Errorf("qwen3tts: %s is %s%v, want BF16[*, %d]", name, info.DType, info.Shape, in)
		}
		bits = append(bits, unsafe.Slice((*uint16)(unsafe.Pointer(&mem[off])), info.Shape[0]*in))
		rows += info.Shape[0]
	}
	if m.int8 && in%64 == 0 {
		qb, err := m.dev.NewBuffer(rows * in)
		if err != nil {
			return metal.DecodeWeight{}, err
		}
		sb, err := m.dev.NewBuffer(4 * rows * in / 64)
		if err != nil {
			qb.Release()
			return metal.DecodeWeight{}, err
		}
		m.owned = append(m.owned, qb, sb)
		qd := unsafe.Slice((*int8)(unsafe.Pointer(&qb.Bytes()[0])), rows*in)
		sd := f32view(sb)
		r0 := 0
		for _, b := range bits {
			n := len(b) / in
			qw := gemm.QuantizeI8GBF16(b, in, n, in)
			copy(qd[r0*in:], qw.Q)
			copy(sd[r0*in/64:], qw.S)
			r0 += n
		}
		return metal.DecodeWeight{W: qb.At(0), S: sb.At(0)}, nil
	}
	wb, err := m.dev.NewBuffer(2 * rows * in)
	if err != nil {
		return metal.DecodeWeight{}, err
	}
	m.owned = append(m.owned, wb)
	dst := unsafe.Slice((*uint16)(unsafe.Pointer(&wb.Bytes()[0])), rows*in)
	r0 := 0
	for _, b := range bits {
		copy(dst[r0*in:], b)
		r0 += len(b) / in
	}
	return metal.DecodeWeight{W: wb.At(0), BF16: true}, nil
}

// weightBF16 wraps name's bf16 [out, in] bytes in place (its shard once).
func (m *metalLM) weightBF16(set *safetensors.Set, name string, out, in int) (metal.Region, error) {
	mem, off, info, err := set.Locate(name)
	if err != nil {
		return metal.Region{}, err
	}
	if info.DType != "BF16" || len(info.Shape) != 2 || info.Shape[0] != out || info.Shape[1] != in {
		return metal.Region{}, fmt.Errorf("qwen3tts: %s is %s%v, want BF16[%d %d]", name, info.DType, info.Shape, out, in)
	}
	if off%8 != 0 {
		return metal.Region{}, fmt.Errorf("qwen3tts: %s at offset %d: the GPU's bfloat4 loads need 8-byte alignment", name, off)
	}
	key := unsafe.Pointer(&mem[0])
	b, ok := m.shards[key]
	if !ok {
		if b, err = m.dev.Wrap(mem); err != nil {
			return metal.Region{}, fmt.Errorf("qwen3tts: wrap shard of %s: %w", name, err)
		}
		m.shards[key] = b
	}
	return b.At(off), nil
}

// f32 copies a small weight (widened) into its own buffer.
func (m *metalLM) f32(set *safetensors.Set, name string) (*metal.Buffer, error) {
	t, err := set.F32(name)
	if err != nil {
		return nil, err
	}
	b, err := m.dev.NewBuffer(4 * t.Numel())
	if err != nil {
		return nil, err
	}
	m.owned = append(m.owned, b)
	copy(f32view(b), t.F32())
	return b, nil
}

func (m *metalLM) reset()   { m.p = 0 }
func (m *metalLM) pos() int { return m.p }

// lin encodes out[rows, w.n] (+)= in[rows, w.k] · wᵀ (+ bias): GEMVs on
// the decode shapes, the bf16 GEMM (through scratch when accumulating) for
// prefill.
func (m *metalLM) lin(e *metal.Encoder, in, out metal.Region, w *metalWeight, rows int, acc bool, bias *metal.Buffer) {
	if rows <= 4 {
		g := metal.Gemv{N: w.n, K: w.k, Rows: rows, X: in, Y: out, Accumulate: acc}
		if bias != nil {
			g.Bias, g.HasBias = bias.At(0), true
		}
		if w.q != nil {
			g.W, g.S = w.q.At(0), w.s.At(0)
		} else {
			g.W, g.BF16 = w.bf16, true
		}
		e.Gemv(g)
		return
	}
	dst := out
	if acc {
		dst = m.y.At(0)
	}
	e.Gemm(metal.Gemm{M: rows, N: w.n, K: w.k, A: in, B: w.bf16, C: dst, TransB: true, BF16: true})
	if bias != nil {
		e.AddBias(dst, bias.At(0), rows, w.n, w.n)
	}
	if acc {
		e.GatedAdd(out, m.ones.At(0), dst, rows, w.n, w.n, w.n)
	}
}

// run feeds n input rows (x [n, in]) at the next positions and returns the
// last row's final-normed hidden state and, for head >= 0, that head's
// logits. The slices alias GPU buffers until the next run.
func (m *metalLM) run(x []float32, n, head int) (hidden, logits []float32, err error) {
	if err := m.load(x, n); err != nil {
		return nil, nil, err
	}
	pos := m.p
	err = m.dev.Run(func(e *metal.Encoder) {
		m.encodeStack(e, n, pos)
		if head >= 0 {
			m.lin(e, m.hid.At(0), m.logits.At(0), m.heads[head], 1, false, nil)
		}
	})
	if err != nil {
		return nil, nil, err
	}
	m.p += n
	hidden = f32view(m.hid)[:m.c.HiddenSize]
	if head >= 0 {
		logits = f32view(m.logits)[:m.heads[head].n]
	}
	return hidden, logits, nil
}

// load copies n input rows into the stack's input buffer.
func (m *metalLM) load(x []float32, n int) error {
	if n > m.maxRows || m.p+n > m.maxT {
		return fmt.Errorf("qwen3tts: GPU run of %d rows at %d exceeds %d rows / %d positions", n, m.p, m.maxRows, m.maxT)
	}
	if m.proj != nil {
		copy(f32view(m.xin), x[:n*m.in])
	} else {
		copy(f32view(m.x), x[:n*m.c.HiddenSize])
	}
	return nil
}

// encodeStack encodes the decoder over n rows at positions pos.. (input
// already in xin/x) and the final norm of the last row into hid.
func (m *metalLM) encodeStack(e *metal.Encoder, n, pos int) {
	D, eps := m.c.HiddenSize, m.c.RMSNormEps
	if m.proj != nil {
		m.lin(e, m.xin.At(0), m.x.At(0), m.proj, n, false, m.projB)
	}
	if n <= 4 {
		m.encodeDecodeLayers(e, n, pos)
	} else {
		m.encodePrefillLayers(e, n, pos)
	}
	e.RMSNormRows(m.x.At(4*(n-1)*D), m.hid.At(0), m.norm.At(0), 1, D, D, D, eps)
}

// encodeDecodeLayers is the decoder over n ≤ 4 rows on the fused one-row
// kernels (looped per row): per layer and row, the QKV GEMV with the input
// norm folded in and its outputs split into q and the cache rows; q/k
// norm + RoPE; then attention over the cache for all rows; then the O GEMV
// accumulating into the residual, the gate/up GEMV (post-norm folded in,
// SiLU·mul in-kernel) and the down GEMV accumulating.
func (m *metalLM) encodeDecodeLayers(e *metal.Encoder, n, pos int) {
	c := m.c
	D, H, KV, dh, I := c.HiddenSize, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim, c.IntermediateSize
	eps := c.RMSNormEps
	scale := float32(1 / math.Sqrt(float64(dh)))
	ldkv, Hd := KV*dh, H*dh
	for li, l := range m.layers {
		for r := range n {
			p := pos + r
			e.Gemv1(metal.FusedGemv{N: Hd + 2*ldkv, K: D, X: m.x.At(4 * r * D), W: l.qkv,
				Y0: m.q.At(4 * r * Hd), Y1: m.kc[li].At(4 * p * ldkv), Y2: m.vc[li].At(4 * p * ldkv),
				N1: Hd, N2: Hd + ldkv, Norm: true, NormW: l.inNorm.At(0), Eps: eps})
			e.RMSNormRoPEQK(m.q.At(4*r*Hd), m.kc[li].At(4*p*ldkv), l.qNorm.At(0), l.kNorm.At(0),
				m.cos.At(4*p*dh/2), m.sin.At(4*p*dh/2), H, KV, dh, metal.RopeHalf, eps)
		}
		e.AttnDecode(m.q.At(0), m.kc[li].At(0), m.vc[li].At(0), m.o.At(0), n, H, KV, dh, pos, Hd, Hd, scale)
		for r := range n {
			e.Gemv1(metal.FusedGemv{N: D, K: Hd, X: m.o.At(4 * r * Hd), W: l.o.dec(), Y0: m.x.At(4 * r * D), Accumulate: true})
			e.GLU1(I, D, m.x.At(4*r*D), l.gate.dec(), l.up.dec(), m.g.At(4*r*I), true, l.postNorm.At(0), eps)
			e.Gemv1(metal.FusedGemv{N: D, K: I, X: m.g.At(4 * r * I), W: l.down.dec(), Y0: m.x.At(4 * r * D), Accumulate: true})
		}
	}
}

// encodePrefillLayers is the decoder over n > 4 rows: bf16 GEMMs on the
// in-place weights, separate norms.
func (m *metalLM) encodePrefillLayers(e *metal.Encoder, n, pos int) {
	c := m.c
	D, H, KV, dh, I := c.HiddenSize, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim, c.IntermediateSize
	eps := c.RMSNormEps
	scale := float32(1 / math.Sqrt(float64(dh)))
	ldkv := KV * dh
	cos, sin := m.cos.At(4*pos*dh/2), m.sin.At(4*pos*dh/2)
	for li, l := range m.layers {
		kAt, vAt := m.kc[li].At(4*pos*ldkv), m.vc[li].At(4*pos*ldkv)
		e.RMSNormRows(m.x.At(0), m.y.At(0), l.inNorm.At(0), n, D, D, D, eps)
		m.lin(e, m.y.At(0), m.q.At(0), l.q, n, false, nil)
		m.lin(e, m.y.At(0), kAt, l.k, n, false, nil)
		m.lin(e, m.y.At(0), vAt, l.v, n, false, nil)
		e.RMSNormRoPEMode(m.q.At(0), l.qNorm.At(0), cos, sin, n, H, dh, H*dh, eps, metal.RopeHalf)
		e.RMSNormRoPEMode(kAt, l.kNorm.At(0), cos, sin, n, KV, dh, ldkv, eps, metal.RopeHalf)
		e.AttnDecode(m.q.At(0), m.kc[li].At(0), m.vc[li].At(0), m.o.At(0), n, H, KV, dh, pos, H*dh, H*dh, scale)
		m.lin(e, m.o.At(0), m.x.At(0), l.o, n, true, nil)
		e.RMSNormRows(m.x.At(0), m.y.At(0), l.postNorm.At(0), n, D, D, D, eps)
		m.lin(e, m.y.At(0), m.g.At(0), l.gate, n, false, nil)
		m.lin(e, m.y.At(0), m.u.At(0), l.up, n, false, nil)
		e.SiLUMul(m.g.At(0), m.u.At(0), m.g.At(0), n, I, I, I, I)
		m.lin(e, m.g.At(0), m.x.At(0), l.down, n, true, nil)
	}
}

// frameSampler is the GPU sampler's setting for the code predictor.
type frameSampler struct {
	greedy      bool
	topK        int
	temperature float32
}

// runFrame runs the code predictor's whole frame in one command buffer:
// the two input rows x, then for each codebook g = 1..G-1 its head,
// sampling on the GPU (with host uniforms us[g-1]) and the chosen code's
// embedding as the next row. It returns the codes and, with keepLogits,
// each step's logits.
func (m *metalLM) runFrame(x []float32, fs frameSampler, us []float32, keepLogits bool) ([]int64, [][]float32, error) {
	tables := m.tables
	if len(tables) != len(m.heads)-1 {
		return nil, nil, fmt.Errorf("qwen3tts: GPU frame needs %d embedding tables, have %d", len(m.heads)-1, len(tables))
	}
	steps := len(m.heads)
	if err := m.load(x, 2); err != nil {
		return nil, nil, err
	}
	if m.codes == nil {
		var err error
		if m.codes, err = m.dev.NewBuffer(4 * steps); err != nil {
			return nil, nil, err
		}
		m.owned = append(m.owned, m.codes)
		if m.allLogits, err = m.dev.NewBuffer(4 * steps * m.heads[0].n); err != nil {
			return nil, nil, err
		}
		m.owned = append(m.owned, m.allLogits)
	}
	V := m.heads[0].n
	in := m.x.At(0)
	if m.proj != nil {
		in = m.xin.At(0)
	}
	err := m.dev.Run(func(e *metal.Encoder) {
		n, pos := 2, m.p
		for g := range steps {
			m.encodeStack(e, n, pos)
			lg := m.allLogits.At(4 * g * V)
			m.lin(e, m.hid.At(0), lg, m.heads[g], 1, false, nil)
			e.SampleLogits(lg, m.codes.At(0), V, fs.topK, fs.greedy, fs.temperature, us[g], g)
			if g+1 < steps {
				e.EmbedRow(tables[g], m.codes.At(0), in, m.in, g)
			}
			pos += n
			n = 1
		}
	})
	if err != nil {
		return nil, nil, err
	}
	m.p += steps + 1
	cv := unsafe.Slice((*uint32)(unsafe.Pointer(&m.codes.Bytes()[0])), steps)
	codes := make([]int64, steps)
	for i, c := range cv {
		codes[i] = int64(c)
	}
	var logits [][]float32
	if keepLogits {
		lf := f32view(m.allLogits)
		for g := range steps {
			logits = append(logits, append([]float32(nil), lf[g*V:(g+1)*V]...))
		}
	}
	return codes, logits, nil
}

// setTables wraps the bf16 embedding tables runFrame feeds back (codebook
// g's chosen code → the next step's input), in place.
func (m *metalLM) setTables(set *safetensors.Set, names []string) error {
	for _, name := range names {
		info, ok := set.Info(name)
		if !ok || len(info.Shape) != 2 || info.Shape[1] != m.in {
			return fmt.Errorf("qwen3tts: embedding table %s missing or not [*, %d]", name, m.in)
		}
		r, err := m.weightBF16(set, name, info.Shape[0], info.Shape[1])
		if err != nil {
			return err
		}
		m.tables = append(m.tables, r)
	}
	return nil
}

func (m *metalLM) close() {
	for _, b := range m.owned {
		b.Release()
	}
	for _, b := range m.shards {
		b.Release()
	}
	m.owned, m.shards = nil, nil
}

func f32view(b *metal.Buffer) []float32 {
	by := b.Bytes()
	return unsafe.Slice((*float32)(unsafe.Pointer(&by[0])), len(by)/4)
}
