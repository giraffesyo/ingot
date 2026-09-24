//go:build darwin && arm64

package qwenimage

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// MetalTextEncoder runs the Qwen3-VL language model on the GPU with its bf16
// weights read in place from the mapped checkpoint: a prompt is a few dozen
// tokens, so the CPU path's cost is packing ~30 GB of weights, which this
// skips entirely. Same outputs as TextEncoder (pre-final-norm hidden
// states, system rows dropped).
type MetalTextEncoder struct {
	dev    *metal.Device
	cfg    TextConfig
	te     *TextEncoder // host embedding lookup
	shards map[unsafe.Pointer]*metal.Buffer
	layers []metalTextLayer
	ones   *metal.Buffer
}

type metalTextLayer struct {
	q, k, v, o, gate, up, down     metal.Region
	inNorm, postNorm, qNorm, kNorm *metal.Buffer
}

// NewMetalTextEncoder maps the language model's weights onto the GPU.
func NewMetalTextEncoder(cfg TextConfig, set *safetensors.Set) (*MetalTextEncoder, error) {
	dev, err := metal.Open()
	if err != nil {
		return nil, err
	}
	if err := dev.Prepare(); err != nil {
		return nil, err
	}
	te, err := NewTextEncoder(cfg, set)
	if err != nil {
		return nil, err
	}
	m := &MetalTextEncoder{dev: dev, cfg: cfg, te: te, shards: map[unsafe.Pointer]*metal.Buffer{}}
	D, H, KV, dh, I := cfg.HiddenSize, cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim, cfg.IntermediateSize
	fail := func(err error) (*MetalTextEncoder, error) { m.Close(); return nil, err }
	for i := range cfg.NumHiddenLayers {
		p := fmt.Sprintf("%slayers.%d.", textPrefix, i)
		var l metalTextLayer
		for _, w := range []struct {
			name    string
			dst     *metal.Region
			out, in int
		}{
			{"self_attn.q_proj.weight", &l.q, H * dh, D}, {"self_attn.k_proj.weight", &l.k, KV * dh, D},
			{"self_attn.v_proj.weight", &l.v, KV * dh, D}, {"self_attn.o_proj.weight", &l.o, D, H * dh},
			{"mlp.gate_proj.weight", &l.gate, I, D}, {"mlp.up_proj.weight", &l.up, I, D},
			{"mlp.down_proj.weight", &l.down, D, I},
		} {
			r, err := wrapWeight(dev, m.shards, set, p+w.name, w.out, w.in)
			if err != nil {
				return fail(err)
			}
			*w.dst = r
		}
		for _, n := range []struct {
			name string
			dst  **metal.Buffer
		}{{"input_layernorm.weight", &l.inNorm}, {"post_attention_layernorm.weight", &l.postNorm},
			{"self_attn.q_norm.weight", &l.qNorm}, {"self_attn.k_norm.weight", &l.kNorm}} {
			b, err := f32Buffer(dev, set, p+n.name)
			if err != nil {
				return fail(err)
			}
			*n.dst = b
		}
		m.layers = append(m.layers, l)
	}
	if m.ones, err = dev.NewBuffer(4 * D); err != nil {
		return fail(err)
	}
	for i := range f32view(m.ones) {
		f32view(m.ones)[i] = 1
	}
	return m, nil
}

// Encode runs a text-only prompt and returns hidden states
// [len(ids)-drop, D]. layers < NumHiddenLayers runs a truncated model.
func (m *MetalTextEncoder) Encode(ids []int64, drop, layers int) (*tensor.Tensor, error) {
	return m.EncodeMM(TextInputs{IDs: ids}, drop, layers)
}

// EncodeMM runs a prompt with image placeholders: their embeddings are
// replaced by the vision tower's merged tokens, the deepstack features are
// added to their hidden states after the first decoder layers, and every
// token rotates by its M-RoPE (t, h, w) position.
func (m *MetalTextEncoder) EncodeMM(in TextInputs, drop, layers int) (*tensor.Tensor, error) {
	c := m.cfg
	ids := in.IDs
	T, D, H, KV, dh, I := len(ids), c.HiddenSize, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim, c.IntermediateSize
	emb, err := m.te.Embed(ids)
	if err != nil {
		return nil, err
	}
	nImg := len(in.ImagePositions)
	if nImg > 0 {
		if in.ImageEmbeds == nil || in.ImageEmbeds.Numel() != nImg*D {
			return nil, fmt.Errorf("qwenimage: %d image positions but image embeds %v", nImg, in.ImageEmbeds)
		}
		for r, p := range in.ImagePositions {
			copy(emb.F32()[p*D:(p+1)*D], in.ImageEmbeds.F32()[r*D:(r+1)*D])
		}
	}
	pos := in.Positions
	if pos == nil {
		pos = make([][3]int, T)
		for p := range pos {
			pos[p] = [3]int{p, p, p}
		}
	}
	if len(pos) != T {
		return nil, fmt.Errorf("qwenimage: %d positions for %d tokens", len(pos), T)
	}
	var bufs []*metal.Buffer
	defer func() {
		for _, b := range bufs {
			b.Release()
		}
	}()
	nb := func(floats int) *metal.Buffer {
		b, e := m.dev.NewBuffer(4 * max(floats, 1))
		if e != nil && err == nil {
			err = e
		}
		if b != nil {
			bufs = append(bufs, b)
		}
		return b
	}
	x, y, q, k, v, o := nb(T*D), nb(T*D), nb(T*H*dh), nb(T*KV*dh), nb(T*KV*dh), nb(T*H*dh)
	g, u, s, mask := nb(T*I), nb(T*I), nb(T*T), nb(T*T)
	cos, sin := nb(T*dh/2), nb(T*dh/2)
	var deep []*metal.Buffer
	var imgIdx *metal.Buffer
	if nImg > 0 {
		for range in.Deepstack {
			deep = append(deep, nb(nImg*D))
		}
		imgIdx = nb(nImg)
	}
	if err != nil {
		return nil, err
	}
	for i, d := range in.Deepstack {
		if d.Numel() != nImg*D {
			return nil, fmt.Errorf("qwenimage: deepstack %d is %v, want [%d %d]", i, d.Shape(), nImg, D)
		}
		copy(f32view(deep[i]), d.F32())
	}
	if nImg > 0 {
		iv := unsafe.Slice((*uint32)(unsafe.Pointer(&imgIdx.Bytes()[0])), nImg)
		for r, p := range in.ImagePositions {
			iv[r] = uint32(p)
		}
	}
	copy(f32view(x), emb.F32())
	mf := f32view(mask)
	for i := range T {
		for j := i + 1; j < T; j++ {
			mf[i*T+j] = float32(math.Inf(-1))
		}
	}
	cf, sf := f32view(cos), f32view(sin)
	for i := 0; i < dh/2; i++ { // as TextEncoder.Build: f32 inv_freq, f32 angle
		inv := 1 / float32(math.Pow(c.RopeTheta, float64(float32(2*i)/float32(dh))))
		ax := mropeAxis(i)
		for p := range T {
			a := float64(float32(pos[p][ax]) * inv)
			cf[p*dh/2+i], sf[p*dh/2+i] = float32(math.Cos(a)), float32(math.Sin(a))
		}
	}
	scale := float32(1 / math.Sqrt(float64(dh)))
	eps := c.RMSNormEps
	lin := func(e *metal.Encoder, a *metal.Buffer, w metal.Region, out *metal.Buffer, n, k int) {
		e.Gemm(metal.Gemm{M: T, N: n, K: k, A: a.At(0), B: w, C: out.At(0), TransB: true, BF16: true})
	}
	err = m.dev.Run(func(e *metal.Encoder) {
		for li, l := range m.layers[:layers] {
			e.RMSNormRows(x.At(0), y.At(0), l.inNorm.At(0), T, D, D, D, eps)
			lin(e, y, l.q, q, H*dh, D)
			lin(e, y, l.k, k, KV*dh, D)
			lin(e, y, l.v, v, KV*dh, D)
			e.RMSNormRoPEMode(q.At(0), l.qNorm.At(0), cos.At(0), sin.At(0), T, H, dh, H*dh, eps, metal.RopeHalf)
			e.RMSNormRoPEMode(k.At(0), l.kNorm.At(0), cos.At(0), sin.At(0), T, KV, dh, KV*dh, eps, metal.RopeHalf)
			for h := range H {
				kv := h / (H / KV) // grouped-query: query head h reads kv head h/(H/KV)
				e.Gemm(metal.Gemm{M: T, N: T, K: dh, A: q.At(4 * h * dh), LDA: H * dh, B: k.At(4 * kv * dh), LDB: KV * dh,
					C: s.At(0), TransB: true})
				e.SoftmaxRowsMasked(s.At(0), mask.At(0), T, T, T, T, scale)
				e.Gemm(metal.Gemm{M: T, N: dh, K: T, A: s.At(0), B: v.At(4 * kv * dh), LDB: KV * dh, C: o.At(4 * h * dh), LDC: H * dh})
			}
			lin(e, o, l.o, y, D, H*dh)
			e.GatedAdd(x.At(0), m.ones.At(0), y.At(0), T, D, D, D)
			e.RMSNormRows(x.At(0), y.At(0), l.postNorm.At(0), T, D, D, D, eps)
			lin(e, y, l.gate, g, I, D)
			lin(e, y, l.up, u, I, D)
			e.SiLUMul(g.At(0), u.At(0), g.At(0), T, I, I, I, I)
			lin(e, g, l.down, y, D, I)
			e.GatedAdd(x.At(0), m.ones.At(0), y.At(0), T, D, D, D)
			if li < len(deep) { // deepstack: visual features join the early layers
				e.ScatterAddRows(x.At(0), deep[li].At(0), imgIdx.At(0), nImg, D, D, D)
			}
		}
	})
	if err != nil {
		return nil, err
	}
	out := tensor.New(tensor.F32, T-drop, D)
	copy(out.F32(), f32view(x)[drop*D:])
	return out, nil
}

// Close releases the GPU buffers (the mapped weights stay the caller's).
func (m *MetalTextEncoder) Close() {
	for _, l := range m.layers {
		for _, b := range []*metal.Buffer{l.inNorm, l.postNorm, l.qNorm, l.kNorm} {
			if b != nil {
				b.Release()
			}
		}
	}
	if m.ones != nil {
		m.ones.Release()
	}
	for _, b := range m.shards {
		b.Release()
	}
}

// wrapWeight wraps name's shard (once per shard, cached in shards) and
// returns its bf16 [out, in] region.
func wrapWeight(dev *metal.Device, shards map[unsafe.Pointer]*metal.Buffer, set *safetensors.Set, name string, out, in int) (metal.Region, error) {
	mem, off, info, err := set.Locate(name)
	if err != nil {
		return metal.Region{}, err
	}
	if info.DType != "BF16" || len(info.Shape) != 2 || info.Shape[0] != out || info.Shape[1] != in {
		return metal.Region{}, fmt.Errorf("qwenimage: %s is %s%v, want BF16[%d %d]", name, info.DType, info.Shape, out, in)
	}
	key := unsafe.Pointer(&mem[0])
	b, ok := shards[key]
	if !ok {
		if b, err = dev.Wrap(mem); err != nil {
			return metal.Region{}, fmt.Errorf("qwenimage: wrap shard of %s: %w", name, err)
		}
		shards[key] = b
	}
	return b.At(off), nil
}

// f32Buffer copies a small weight (widened to f32) into its own buffer.
func f32Buffer(dev *metal.Device, set *safetensors.Set, name string) (*metal.Buffer, error) {
	t, err := set.F32(name)
	if err != nil {
		return nil, err
	}
	b, err := dev.NewBuffer(4 * t.Numel())
	if err != nil {
		return nil, err
	}
	copy(f32view(b), t.F32())
	return b, nil
}
