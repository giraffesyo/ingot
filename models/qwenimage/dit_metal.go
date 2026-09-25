//go:build darwin && arm64

package qwenimage

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// MetalDiT runs the DiT's per-step target pass on the GPU (kernels/metal):
// one command buffer per step, bf16 weights read in place from the mapped
// checkpoint (no copies, no packing). The prefix pass and the per-step
// modulation vectors stay on the CPU — both are tiny. The prefix K/V live in
// small per-layer buffers; target K/V share one buffer across layers, and
// attention covers [prefix, target] keys as two column blocks of S.
type MetalDiT struct {
	// Fast runs every GEMM with bf16 activations (the matrix units' fast
	// path, ~2.6x): inputs are rounded to bf16, accumulation and the
	// residual stream, norms and softmax stay f32. Set before Prefix.
	Fast bool

	dev    *metal.Device
	cfg    DiTConfig
	l      *DiTLayout
	layers int
	pv     int // valid prefix tokens
	mod    *graph.Session

	shards  map[unsafe.Pointer]*metal.Buffer
	lw      []metalLayer
	imgIn   metal.Region
	projOut metal.Region

	tw                 *metalWork // target working set (allocated by the first Step)
	lat, out           *metal.Buffer
	s1, g1, s2, g2, fs *metal.Buffer
	pvalid             *metal.Buffer   // valid prefix rows
	kp, vp             []*metal.Buffer // per-layer prefix K/V [valid prefix, dim] (f32 mode)
	kp16, vp16         []*metal.Buffer // bf16 prefix K/V (Fast)
	ktmp, vtmp         *metal.Buffer   // f32 staging for an f32 prefix pass under Fast
	textIn             *graph.Session

	mode        int8 // buffers allocated for: 0 not yet, 1 f32, 2 Fast
	bytes, peak int  // scratch allocated through buf: now and at most
}

// metalWork is one token set's activations: x the residual stream, y/o
// scratch, q/k/v projections, g/p the MLP halves, s one head's scores,
// cos/sin its rotary tables.
type metalWork struct {
	T                         int
	x, y, q, k, v, o, g, p, s *metal.Buffer
	cos, sin                  *metal.Buffer
	// bf16 GEMM inputs for Fast mode.
	y16, q16, k16, v16 *metal.Buffer
}

func (w *metalWork) buffers() []*metal.Buffer {
	return []*metal.Buffer{w.x, w.y, w.q, w.k, w.v, w.o, w.g, w.p, w.s, w.cos, w.sin, w.y16, w.q16, w.k16, w.v16}
}

type metalLayer struct {
	q, k, v, o, gate, proj, out metal.Region
	normQ, normK                *metal.Buffer
}

// NewMetalDiT prepares the GPU pass for layout l (layers < NumLayers
// builds a truncated model, for tests).
func NewMetalDiT(cfg DiTConfig, set *safetensors.Set, l *DiTLayout, layers int) (*MetalDiT, error) {
	dev, err := metal.Open()
	if err != nil {
		return nil, err
	}
	if err := dev.PrepareFlash(); err != nil {
		return nil, err
	}
	mod, err := compile(BuildDiTModulation(cfg, set))
	if err != nil {
		return nil, err
	}
	m := &MetalDiT{dev: dev, cfg: cfg, l: l, layers: layers, pv: len(l.prefixValid), mod: mod,
		shards: map[unsafe.Pointer]*metal.Buffer{}}
	if m.pv == 0 {
		return nil, fmt.Errorf("qwenimage: metal DiT needs a non-empty prefix")
	}
	var r metal.Region
	D, T := cfg.dim(), l.Target
	hid := D * cfg.MLPRatio
	if m.imgIn, err = m.weight(set, "img_in.weight", D, cfg.InChannels); err != nil {
		return nil, err
	}
	if m.projOut, err = m.weight(set, "proj_out.weight", cfg.OutChannels, D); err != nil {
		return nil, err
	}
	for i := range layers {
		p := fmt.Sprintf("transformer_blocks.%d.", i)
		var lw metalLayer
		for _, w := range []struct {
			name    string
			dst     *metal.Region
			out, in int
		}{
			{"attn.to_q.weight", &lw.q, D, D}, {"attn.to_k.weight", &lw.k, D, D}, {"attn.to_v.weight", &lw.v, D, D},
			{"attn.to_out.0.weight", &lw.o, D, D}, {"img_mlp.gate_layer.weight", &lw.gate, hid, D},
			{"img_mlp.proj.weight", &lw.proj, hid, D}, {"img_mlp.out.weight", &lw.out, D, hid},
		} {
			r, err = m.weight(set, p+w.name, w.out, w.in)
			if err != nil {
				return nil, err
			}
			*w.dst = r
		}
		if lw.normQ, err = m.f32Buffer(set, p+"attn.norm_q.weight"); err != nil {
			return nil, err
		}
		if lw.normK, err = m.f32Buffer(set, p+"attn.norm_k.weight"); err != nil {
			return nil, err
		}
		m.lw = append(m.lw, lw)
	}
	nb := func(floats int) *metal.Buffer { return m.buf(4*floats, &err) }
	m.lat, m.out = nb(T*cfg.InChannels), nb(T*cfg.OutChannels)
	m.s1, m.g1, m.s2, m.g2, m.fs = nb(D), nb(D), nb(D), nb(D), nb(D)
	m.pvalid = nb(m.pv)
	if err == nil {
		m.textIn, err = compile(BuildDiTTextIn(cfg, set, l))
	}
	if err != nil {
		m.Close()
		return nil, err
	}
	idx := unsafe.Slice((*uint32)(unsafe.Pointer(&m.pvalid.Bytes()[0])), m.pv)
	for i, v := range l.prefixValid {
		idx[i] = uint32(v)
	}
	return m, nil
}

// metalDiTScratch is the peak GPU memory a MetalDiT allocates for layout l
// beyond the mapped weights, in bytes, running as the pipeline does
// (Prefix, then Steps): the persistent buffers and prefix K/V, plus the
// larger of the prefix pass's working set (freed when it ends) and the
// steps'. The preflight's estimate; TestMetalDiTParity checks it against
// what a run allocates.
func metalDiTScratch(cfg DiTConfig, l *DiTLayout, layers int, fast bool) int {
	D, T, P, pv := cfg.dim(), l.Target, l.Prefix, len(l.prefixValid)
	hid := D * cfg.MLPRatio
	fastPrefix := fast && prefixKeyEnds(l) != nil
	work := func(T, sw, rope int, fast, scores bool) int {
		n := 4 * (6*T*D + 2*T*hid + rope)
		if scores {
			n += 4 * T * sw
		}
		if fast {
			n += 2 * (T*max(D, hid) + 3*T*D)
		}
		return n
	}
	n := 4 * (T*(cfg.InChannels+cfg.OutChannels) + 5*D + pv)
	switch {
	case !fast:
		n += 2 * layers * 4 * pv * D
	case fastPrefix:
		n += 2 * layers * 2 * pv * D
	default:
		n += 2*layers*2*pv*D + 2*4*pv*D
	}
	prefix := work(P, P, len(l.prefixCos)+len(l.prefixSin), fastPrefix, !fastPrefix)
	if fastPrefix {
		prefix += 4 * P
	} else {
		prefix += 4 * P * P
	}
	return n + max(prefix, work(T, pv+T, len(l.targetCos)+len(l.targetSin), fast, !fast))
}

// buf allocates n bytes of scratch (counted in bytes/peak), keeping the
// first error in *err.
func (m *MetalDiT) buf(n int, err *error) *metal.Buffer {
	n = max(n, 4)
	b, e := m.dev.NewBuffer(n)
	if e != nil {
		if *err == nil {
			*err = e
		}
		return nil
	}
	m.bytes += n
	m.peak = max(m.peak, m.bytes)
	return b
}

// free releases buf-allocated buffers (nil ones are skipped).
func (m *MetalDiT) free(bs ...*metal.Buffer) {
	for _, b := range bs {
		if b != nil {
			m.bytes -= b.Len()
			b.Release()
		}
	}
}

// work allocates a working set over T tokens: scores (sw columns) for the
// f32 attention path, bf16 operands for Fast.
func (m *MetalDiT) work(T, sw int, cos, sin []float32, fast, scores bool) (*metalWork, error) {
	D, hid := m.cfg.dim(), m.cfg.dim()*m.cfg.MLPRatio
	var err error
	f32 := func(n int) *metal.Buffer { return m.buf(4*n, &err) }
	bf16 := func(n int) *metal.Buffer { return m.buf(2*n, &err) }
	w := &metalWork{T: T, x: f32(T * D), y: f32(T * D), q: f32(T * D), k: f32(T * D), v: f32(T * D), o: f32(T * D),
		g: f32(T * hid), p: f32(T * hid), cos: f32(len(cos)), sin: f32(len(sin))}
	if scores {
		w.s = f32(T * sw)
	}
	if fast {
		w.y16, w.q16, w.k16, w.v16 = bf16(T*max(D, hid)), bf16(T*D), bf16(T*D), bf16(T*D)
	}
	if err != nil {
		m.free(w.buffers()...)
		return nil, err
	}
	copy(f32view(w.cos), cos)
	copy(f32view(w.sin), sin)
	return w, nil
}

// prepare allocates the prefix K/V for the mode Fast selects, on the first
// pass; the mode cannot change afterwards.
func (m *MetalDiT) prepare() error {
	mode := int8(1)
	if m.Fast {
		mode = 2
	}
	if m.mode != 0 {
		if m.mode != mode {
			return fmt.Errorf("qwenimage: MetalDiT.Fast changed after the first pass")
		}
		return nil
	}
	D := m.cfg.dim()
	var err error
	for range m.layers {
		if m.Fast {
			m.kp16, m.vp16 = append(m.kp16, m.buf(2*m.pv*D, &err)), append(m.vp16, m.buf(2*m.pv*D, &err))
		} else {
			m.kp, m.vp = append(m.kp, m.buf(4*m.pv*D, &err)), append(m.vp, m.buf(4*m.pv*D, &err))
		}
	}
	if m.Fast && prefixKeyEnds(m.l) == nil {
		m.ktmp, m.vtmp = m.buf(4*m.pv*D, &err), m.buf(4*m.pv*D, &err)
	}
	if err != nil {
		return err
	}
	m.mode = mode
	return nil
}

// weight wraps name's shard (once) and returns its bf16 [out, in] region.
func (m *MetalDiT) weight(set *safetensors.Set, name string, out, in int) (metal.Region, error) {
	return wrapWeight(m.dev, m.shards, set, name, out, in)
}

// f32Buffer copies a small weight (widened to f32) into its own buffer.
func (m *MetalDiT) f32Buffer(set *safetensors.Set, name string) (*metal.Buffer, error) {
	return f32Buffer(m.dev, set, name)
}

func f32view(b *metal.Buffer) []float32 {
	raw := b.Bytes()
	return unsafe.Slice((*float32)(unsafe.Pointer(&raw[0])), len(raw)/4)
}

// SetPrefix loads the prefix pass's per-layer K/V ("k<i>", "v<i>" [valid
// prefix tokens, dim]) — once per generation.
func (m *MetalDiT) SetPrefix(kv map[string]*tensor.Tensor) error {
	if m.Fast {
		return fmt.Errorf("qwenimage: SetPrefix needs Fast off")
	}
	if err := m.prepare(); err != nil {
		return err
	}
	for i := range m.layers {
		for _, p := range []struct {
			name string
			dst  *metal.Buffer
		}{{fmt.Sprintf("k%d", i), m.kp[i]}, {fmt.Sprintf("v%d", i), m.vp[i]}} {
			t := kv[p.name]
			if t == nil || t.Numel() != m.pv*m.cfg.dim() {
				return fmt.Errorf("qwenimage: prefix %s missing or mis-sized", p.name)
			}
			copy(f32view(p.dst), t.F32())
		}
	}
	return nil
}

// setMod computes the modulation vectors for model time t on the CPU.
func (m *MetalDiT) setMod(t float32) error {
	mod, err := m.mod.Run(map[string]*tensor.Tensor{"t": tensor.FromF32([]float32{t}, 1)})
	if err != nil {
		return err
	}
	for name, b := range map[string]*metal.Buffer{"s1": m.s1, "g1": m.g1, "s2": m.s2, "g2": m.g2, "fs": m.fs} {
		copy(f32view(b), mod[name].F32())
	}
	m.mod.Release(mod)
	return nil
}

// Prefix runs the prefix pass on the GPU and keeps its per-layer K/V:
// txt_in (and img_in for condition latents) on the CPU, then every block
// over the prefix tokens with t=0 modulation and block-causal attention.
// Once per generation, before Step.
func (m *MetalDiT) Prefix(txt, cond *tensor.Tensor) error {
	if err := m.prepare(); err != nil {
		return err
	}
	D, dh, P := m.cfg.dim(), m.cfg.AttentionHeadDim, m.l.Prefix
	scale := float32(1 / math.Sqrt(float64(dh)))
	// The prefix runs fast too when its mask is prefix-shaped: every row
	// attends to keys [0, end) (block-causal, no padding) — Flash's KeyEnd.
	var kends []int
	if m.Fast {
		kends = prefixKeyEnds(m.l)
	}
	fastPrefix := kends != nil
	w, err := m.work(P, P, m.l.prefixCos, m.l.prefixSin, fastPrefix, !fastPrefix)
	if err != nil {
		return err
	}
	var aux *metal.Buffer // key limits (fast) or the additive mask (f32)
	defer func() { m.free(append(w.buffers(), aux)...) }()
	if fastPrefix {
		if aux = m.buf(4*P, &err); err != nil {
			return err
		}
		iv := unsafe.Slice((*uint32)(unsafe.Pointer(&aux.Bytes()[0])), P)
		for r, e := range kends {
			iv[r] = uint32(e)
		}
	} else {
		if aux = m.buf(4*P*P, &err); err != nil {
			return err
		}
		copy(f32view(aux), m.l.prefixMask)
	}
	in, err := m.textIn.Run(m.l.PrefixFeeds(txt, cond))
	if err != nil {
		return err
	}
	copy(f32view(w.x), in["x"].F32())
	m.textIn.Release(in)
	if err := m.setMod(0); err != nil {
		return err
	}
	if fastPrefix {
		kend := aux.At(0)
		return m.dev.Run(func(e *metal.Encoder) {
			for li := range m.lw {
				m.blockAttn(e, li, w, true, func() {
					e.GatherRows16(w.k16.At(0), m.kp16[li].At(0), m.pvalid.At(0), m.pv, D, D, D)
					e.GatherRows16(w.v16.At(0), m.vp16[li].At(0), m.pvalid.At(0), m.pv, D, D, D)
				}, func() {
					e.Flash(metal.Flash{Q: w.q16.At(0), K1: w.k16.At(0), V1: w.v16.At(0), K2: w.k16.At(0), V2: w.v16.At(0),
						O: w.o.At(0), KeyEnd: &kend, Tq: P, N1: P, N2: 0, Heads: m.cfg.NumAttentionHeads,
						LDQ: D, LD1: D, LD2: D, LDO: D, Scale: scale})
				})
			}
		})
	}
	return m.dev.Run(func(e *metal.Encoder) {
		for li := range m.lw {
			m.block(e, li, w, false, func() {
				if m.Fast { // stage in f32, keep bf16
					e.GatherRows(w.k.At(0), m.ktmp.At(0), m.pvalid.At(0), m.pv, D, D, D)
					e.GatherRows(w.v.At(0), m.vtmp.At(0), m.pvalid.At(0), m.pv, D, D, D)
					e.CastBF16(m.ktmp.At(0), m.kp16[li].At(0), m.pv, D, D, D)
					e.CastBF16(m.vtmp.At(0), m.vp16[li].At(0), m.pv, D, D, D)
					return
				}
				e.GatherRows(w.k.At(0), m.kp[li].At(0), m.pvalid.At(0), m.pv, D, D, D)
				e.GatherRows(w.v.At(0), m.vp[li].At(0), m.pvalid.At(0), m.pv, D, D, D)
			}, func(ho int) {
				e.Gemm(metal.Gemm{M: P, N: P, K: dh, A: w.q.At(ho), LDA: D, B: w.k.At(ho), LDB: D, C: w.s.At(0), TransB: true})
				e.SoftmaxRowsMasked(w.s.At(0), aux.At(0), P, P, P, P, scale)
				e.Gemm(metal.Gemm{M: P, N: dh, K: P, A: w.s.At(0), B: w.v.At(ho), LDB: D, C: w.o.At(ho), LDC: D})
			})
		}
	})
}

// Step runs one denoising step: packed latents x [Target, in] at model
// time t → velocity [Target, out]. Attention covers [prefix, target] keys
// as two column blocks of each head's scores.
func (m *MetalDiT) Step(x *tensor.Tensor, t float32) (*tensor.Tensor, error) {
	cfg := m.cfg
	D, dh, T, P := cfg.dim(), cfg.AttentionHeadDim, m.l.Target, m.pv
	if x.Numel() != T*cfg.InChannels {
		return nil, fmt.Errorf("qwenimage: latents %v, want [%d %d]", x.Shape(), T, cfg.InChannels)
	}
	copy(f32view(m.lat), x.F32())
	if err := m.setMod(t); err != nil {
		return nil, err
	}
	scale := float32(1 / math.Sqrt(float64(dh)))
	SW := P + T
	fast := m.Fast
	if err := m.prepare(); err != nil {
		return nil, err
	}
	if m.tw == nil {
		w, err := m.work(T, SW, m.l.targetCos, m.l.targetSin, fast, !fast)
		if err != nil {
			return nil, err
		}
		m.tw = w
	}
	w := m.tw
	err := m.dev.Run(func(e *metal.Encoder) {
		e.Gemm(metal.Gemm{M: T, N: D, K: cfg.InChannels, A: m.lat.At(0), B: m.imgIn, C: w.x.At(0), TransB: true, BF16: true})
		for li := range m.lw {
			if fast {
				// One fused attention dispatch per layer: keys are the
				// cached prefix then this step's target tokens.
				m.blockAttn(e, li, w, true, nil, func() {
					e.Flash(metal.Flash{Q: w.q16.At(0), K1: m.kp16[li].At(0), V1: m.vp16[li].At(0),
						K2: w.k16.At(0), V2: w.v16.At(0), O: w.o.At(0), Tq: T, N1: P, N2: T,
						Heads: cfg.NumAttentionHeads, LDQ: D, LD1: D, LD2: D, LDO: D, Scale: scale})
				})
				continue
			}
			m.block(e, li, w, false, nil, func(ho int) {
				e.Gemm(metal.Gemm{M: T, N: P, K: dh, A: w.q.At(ho), LDA: D, B: m.kp[li].At(ho), LDB: D,
					C: w.s.At(0), LDC: SW, TransB: true})
				e.Gemm(metal.Gemm{M: T, N: T, K: dh, A: w.q.At(ho), LDA: D, B: w.k.At(ho), LDB: D,
					C: w.s.At(4 * P), LDC: SW, TransB: true})
				e.SoftmaxRows(w.s.At(0), T, SW, SW, scale)
				e.Gemm(metal.Gemm{M: T, N: dh, K: P, A: w.s.At(0), LDA: SW, B: m.vp[li].At(ho), LDB: D,
					C: w.o.At(ho), LDC: D})
				e.Gemm(metal.Gemm{M: T, N: dh, K: T, A: w.s.At(4 * P), LDA: SW, B: w.v.At(ho), LDB: D,
					C: w.o.At(ho), LDC: D, Accumulate: true})
			})
		}
		e.LayerNormMod(w.x.At(0), w.y.At(0), m.fs.At(0), T, D, D, D, cfg.Eps)
		m.linear(e, w, fast, w.y, D, m.projOut, m.out, cfg.OutChannels)
	})
	if err != nil {
		return nil, err
	}
	out := tensor.New(tensor.F32, T, cfg.OutChannels)
	copy(out.F32(), f32view(m.out))
	return out, nil
}

// block encodes one transformer block, attention head by head.
func (m *MetalDiT) block(e *metal.Encoder, li int, w *metalWork, fast bool, afterKV func(), attend func(ho int)) {
	H, dh := m.cfg.NumAttentionHeads, m.cfg.AttentionHeadDim
	m.blockAttn(e, li, w, fast, afterKV, func() {
		for h := range H {
			attend(4 * h * dh)
		}
	})
}

// blockAttn encodes one transformer block over w's tokens. afterKV (optional)
// runs once q/k/v are final (post-norm, post-RoPE; in fast mode also as
// bf16 in q16/k16/v16); attend(ho) fills head h's slice of w.o (ho = its
// byte offset in an f32 [T, dim] row).
func (m *MetalDiT) blockAttn(e *metal.Encoder, li int, w *metalWork, fast bool, afterKV func(), attendAll func()) {
	cfg, lw := m.cfg, m.lw[li]
	D, H, dh, T, eps := cfg.dim(), cfg.NumAttentionHeads, cfg.AttentionHeadDim, w.T, cfg.Eps
	hid := D * cfg.MLPRatio
	if fast { // producers write the bf16 GEMM operands directly
		e.LayerNormModBF16(w.x.At(0), w.y16.At(0), m.s1.At(0), T, D, D, D, eps)
	} else {
		e.LayerNormMod(w.x.At(0), w.y.At(0), m.s1.At(0), T, D, D, D, eps)
	}
	m.linearFrom(e, w, fast, w.y, D, lw.q, w.q, D)
	m.linearFrom(e, w, fast, w.y, D, lw.k, w.k, D)
	m.linearFrom(e, w, fast, w.y, D, lw.v, w.v, D)
	if fast {
		e.RMSNormRoPEBF16(w.q.At(0), w.q16.At(0), lw.normQ.At(0), w.cos.At(0), w.sin.At(0), T, H, dh, D, eps, metal.RopePairs)
		e.RMSNormRoPEBF16(w.k.At(0), w.k16.At(0), lw.normK.At(0), w.cos.At(0), w.sin.At(0), T, H, dh, D, eps, metal.RopePairs)
		e.CastBF16(w.v.At(0), w.v16.At(0), T, D, D, D)
	} else {
		e.RMSNormRoPE(w.q.At(0), lw.normQ.At(0), w.cos.At(0), w.sin.At(0), T, H, dh, D, eps)
		e.RMSNormRoPE(w.k.At(0), lw.normK.At(0), w.cos.At(0), w.sin.At(0), T, H, dh, D, eps)
	}
	if afterKV != nil {
		afterKV()
	}
	attendAll()
	m.linear(e, w, fast, w.o, D, lw.o, w.y, D)
	e.GatedAdd(w.x.At(0), m.g1.At(0), w.y.At(0), T, D, D, D)

	if fast {
		e.LayerNormModBF16(w.x.At(0), w.y16.At(0), m.s2.At(0), T, D, D, D, eps)
	} else {
		e.LayerNormMod(w.x.At(0), w.y.At(0), m.s2.At(0), T, D, D, D, eps)
	}
	m.linearFrom(e, w, fast, w.y, D, lw.gate, w.g, hid)
	m.linearFrom(e, w, fast, w.y, D, lw.proj, w.p, hid)
	if fast {
		e.SiLUMulBF16(w.g.At(0), w.p.At(0), w.y16.At(0), T, hid, hid, hid, hid)
		e.Gemm(metal.Gemm{M: T, N: D, K: hid, A: w.y16.At(0), B: lw.out, C: w.y.At(0), TransB: true, BF16: true, ABF16: true})
	} else {
		e.SiLUMul(w.g.At(0), w.p.At(0), w.g.At(0), T, hid, hid, hid, hid)
		m.linear(e, w, fast, w.g, hid, lw.out, w.y, D)
	}
	e.GatedAdd(w.x.At(0), m.g2.At(0), w.y.At(0), T, D, D, D)
}

// linear encodes c = a·Wᵀ for a [T, in] (casting a to bf16 first in fast
// mode, through w.y16 — sized for the widest input).
func (m *MetalDiT) linear(e *metal.Encoder, w *metalWork, fast bool, a *metal.Buffer, in int, wt metal.Region, c *metal.Buffer, out int) {
	if fast {
		e.CastBF16(a.At(0), w.y16.At(0), w.T, in, in, in)
	}
	m.linearFrom(e, w, fast, a, in, wt, c, out)
}

// linearFrom is linear with the bf16 copy of a already in w.y16.
func (m *MetalDiT) linearFrom(e *metal.Encoder, w *metalWork, fast bool, a *metal.Buffer, in int, wt metal.Region, c *metal.Buffer, out int) {
	if fast {
		e.Gemm(metal.Gemm{M: w.T, N: out, K: in, A: w.y16.At(0), B: wt, C: c.At(0), TransB: true, BF16: true, ABF16: true})
		return
	}
	e.Gemm(metal.Gemm{M: w.T, N: out, K: in, A: a.At(0), B: wt, C: c.At(0), TransB: true, BF16: true})
}

// prefixKeyEnds returns, per prefix row, the end of its allowed keys when
// every row allows exactly [0, end) — true of the block-causal mask without
// padding — or nil.
func prefixKeyEnds(l *DiTLayout) []int {
	P := l.Prefix
	if len(l.prefixValid) != P {
		return nil
	}
	ends := make([]int, P)
	for q := range P {
		row := l.prefixMask[q*P : (q+1)*P]
		end := 0
		for end < P && !math.IsInf(float64(row[end]), -1) {
			end++
		}
		for k := end; k < P; k++ {
			if !math.IsInf(float64(row[k]), -1) {
				return nil // allowed keys are not a prefix
			}
		}
		if end == 0 {
			return nil
		}
		ends[q] = end
	}
	return ends
}

// Close releases the GPU buffers (the mapped weights stay the caller's).
func (m *MetalDiT) Close() {
	if m.tw != nil {
		m.free(m.tw.buffers()...)
	}
	m.free(m.lat, m.out, m.s1, m.g1, m.s2, m.g2, m.fs, m.pvalid, m.ktmp, m.vtmp)
	m.free(m.kp...)
	m.free(m.vp...)
	m.free(m.kp16...)
	m.free(m.vp16...)
	for _, lw := range m.lw {
		lw.normQ.Release()
		lw.normK.Release()
	}
	for _, b := range m.shards {
		b.Release()
	}
}
