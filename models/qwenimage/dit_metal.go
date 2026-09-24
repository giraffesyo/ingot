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

	tw, pw             *metalWork // target and prefix working sets
	lat, out           *metal.Buffer
	s1, g1, s2, g2, fs *metal.Buffer
	pmask, pvalid      *metal.Buffer   // prefix attention mask, valid-row indices
	kp, vp             []*metal.Buffer // per-layer prefix K/V [valid prefix, dim]
	kp16, vp16         []*metal.Buffer // their bf16 copies (Fast)
	textIn             *graph.Session
}

// metalWork is one token set's activations: x the residual stream, y/o
// scratch, q/k/v projections, g/p the MLP halves, s one head's scores,
// cos/sin its rotary tables.
type metalWork struct {
	T                         int
	x, y, q, k, v, o, g, p, s *metal.Buffer
	cos, sin                  *metal.Buffer
	// bf16 GEMM inputs for Fast mode (allocated on first use).
	y16, q16, k16, v16, s16 *metal.Buffer
}

func (w *metalWork) release() {
	for _, b := range []*metal.Buffer{w.x, w.y, w.q, w.k, w.v, w.o, w.g, w.p, w.s, w.cos, w.sin,
		w.y16, w.q16, w.k16, w.v16, w.s16} {
		if b != nil {
			b.Release()
		}
	}
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
	if err := dev.Prepare(); err != nil {
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
	ok := func(e error) {
		if err == nil {
			err = e
		}
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
	nb := func(floats int) *metal.Buffer {
		b, e := dev.NewBuffer(4 * max(floats, 1))
		ok(e)
		return b
	}
	work := func(T, sw int, cos, sin []float32) *metalWork {
		w := &metalWork{T: T, x: nb(T * D), y: nb(T * D), q: nb(T * D), k: nb(T * D), v: nb(T * D), o: nb(T * D),
			g: nb(T * hid), p: nb(T * hid), s: nb(T * sw), cos: nb(len(cos)), sin: nb(len(sin))}
		if err == nil {
			copy(f32view(w.cos), cos)
			copy(f32view(w.sin), sin)
		}
		return w
	}
	m.tw = work(T, m.pv+T, l.targetCos, l.targetSin)
	m.pw = work(l.Prefix, l.Prefix, l.prefixCos, l.prefixSin)
	m.lat, m.out = nb(T*cfg.InChannels), nb(T*cfg.OutChannels)
	m.s1, m.g1, m.s2, m.g2, m.fs = nb(D), nb(D), nb(D), nb(D), nb(D)
	m.pmask, m.pvalid = nb(l.Prefix*l.Prefix), nb(m.pv)
	for range layers {
		m.kp, m.vp = append(m.kp, nb(m.pv*D)), append(m.vp, nb(m.pv*D))
	}
	if err == nil {
		m.textIn, err = compile(BuildDiTTextIn(cfg, set, l))
	}
	if err != nil {
		m.Close()
		return nil, err
	}
	copy(f32view(m.pmask), l.prefixMask)
	idx := unsafe.Slice((*uint32)(unsafe.Pointer(&m.pvalid.Bytes()[0])), m.pv)
	for i, v := range l.prefixValid {
		idx[i] = uint32(v)
	}
	return m, nil
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
	in, err := m.textIn.Run(m.l.PrefixFeeds(txt, cond))
	if err != nil {
		return err
	}
	copy(f32view(m.pw.x), in["x"].F32())
	m.textIn.Release(in)
	if err := m.setMod(0); err != nil {
		return err
	}
	D, dh, P := m.cfg.dim(), m.cfg.AttentionHeadDim, m.l.Prefix
	scale := float32(1 / math.Sqrt(float64(dh)))
	w := m.pw
	if m.Fast {
		if err := m.fastBuffers(); err != nil {
			return err
		}
	}
	return m.dev.Run(func(e *metal.Encoder) {
		for li := range m.lw {
			m.block(e, li, w, false, func() {
				e.GatherRows(w.k.At(0), m.kp[li].At(0), m.pvalid.At(0), m.pv, D, D, D)
				e.GatherRows(w.v.At(0), m.vp[li].At(0), m.pvalid.At(0), m.pv, D, D, D)
				if m.Fast {
					e.CastBF16(m.kp[li].At(0), m.kp16[li].At(0), m.pv, D, D, D)
					e.CastBF16(m.vp[li].At(0), m.vp16[li].At(0), m.pv, D, D, D)
				}
			}, func(ho int) {
				e.Gemm(metal.Gemm{M: P, N: P, K: dh, A: w.q.At(ho), LDA: D, B: w.k.At(ho), LDB: D, C: w.s.At(0), TransB: true})
				e.SoftmaxRowsMasked(w.s.At(0), m.pmask.At(0), P, P, P, P, scale)
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
	w := m.tw
	fast := m.Fast
	if fast && len(m.kp16) == 0 {
		return nil, fmt.Errorf("qwenimage: Fast set after Prefix")
	}
	err := m.dev.Run(func(e *metal.Encoder) {
		e.Gemm(metal.Gemm{M: T, N: D, K: cfg.InChannels, A: m.lat.At(0), B: m.imgIn, C: w.x.At(0), TransB: true, BF16: true})
		for li := range m.lw {
			if fast {
				m.block(e, li, w, true, nil, func(ho int) {
					h2 := ho / 2 // byte offset of the head in a bf16 row
					e.Gemm(metal.Gemm{M: T, N: P, K: dh, A: w.q16.At(h2), LDA: D, B: m.kp16[li].At(h2), LDB: D,
						C: w.s.At(0), LDC: SW, TransB: true, BF16: true, ABF16: true})
					e.Gemm(metal.Gemm{M: T, N: T, K: dh, A: w.q16.At(h2), LDA: D, B: w.k16.At(h2), LDB: D,
						C: w.s.At(4 * P), LDC: SW, TransB: true, BF16: true, ABF16: true})
					e.SoftmaxRows(w.s.At(0), T, SW, SW, scale)
					e.CastBF16(w.s.At(0), w.s16.At(0), T, SW, SW, SW)
					e.Gemm(metal.Gemm{M: T, N: dh, K: P, A: w.s16.At(0), LDA: SW, B: m.vp16[li].At(h2), LDB: D,
						C: w.o.At(ho), LDC: D, BF16: true, ABF16: true})
					e.Gemm(metal.Gemm{M: T, N: dh, K: T, A: w.s16.At(2 * P), LDA: SW, B: w.v16.At(h2), LDB: D,
						C: w.o.At(ho), LDC: D, BF16: true, ABF16: true, Accumulate: true})
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

// block encodes one transformer block over w's tokens. afterKV (optional)
// runs once q/k/v are final (post-norm, post-RoPE; in fast mode also as
// bf16 in q16/k16/v16); attend(ho) fills head h's slice of w.o (ho = its
// byte offset in an f32 [T, dim] row).
func (m *MetalDiT) block(e *metal.Encoder, li int, w *metalWork, fast bool, afterKV func(), attend func(ho int)) {
	cfg, lw := m.cfg, m.lw[li]
	D, H, dh, T, eps := cfg.dim(), cfg.NumAttentionHeads, cfg.AttentionHeadDim, w.T, cfg.Eps
	hid := D * cfg.MLPRatio
	e.LayerNormMod(w.x.At(0), w.y.At(0), m.s1.At(0), T, D, D, D, eps)
	if fast {
		e.CastBF16(w.y.At(0), w.y16.At(0), T, D, D, D)
	}
	m.linearFrom(e, w, fast, w.y, D, lw.q, w.q, D)
	m.linearFrom(e, w, fast, w.y, D, lw.k, w.k, D)
	m.linearFrom(e, w, fast, w.y, D, lw.v, w.v, D)
	e.RMSNormRoPE(w.q.At(0), lw.normQ.At(0), w.cos.At(0), w.sin.At(0), T, H, dh, D, eps)
	e.RMSNormRoPE(w.k.At(0), lw.normK.At(0), w.cos.At(0), w.sin.At(0), T, H, dh, D, eps)
	if fast {
		e.CastBF16(w.q.At(0), w.q16.At(0), T, D, D, D)
		e.CastBF16(w.k.At(0), w.k16.At(0), T, D, D, D)
		e.CastBF16(w.v.At(0), w.v16.At(0), T, D, D, D)
	}
	if afterKV != nil {
		afterKV()
	}
	for h := range H {
		attend(4 * h * dh)
	}
	m.linear(e, w, fast, w.o, D, lw.o, w.y, D)
	e.GatedAdd(w.x.At(0), m.g1.At(0), w.y.At(0), T, D, D, D)

	e.LayerNormMod(w.x.At(0), w.y.At(0), m.s2.At(0), T, D, D, D, eps)
	if fast {
		e.CastBF16(w.y.At(0), w.y16.At(0), T, D, D, D)
	}
	m.linearFrom(e, w, fast, w.y, D, lw.gate, w.g, hid)
	m.linearFrom(e, w, fast, w.y, D, lw.proj, w.p, hid)
	e.SiLUMul(w.g.At(0), w.p.At(0), w.g.At(0), T, hid, hid, hid, hid)
	m.linear(e, w, fast, w.g, hid, lw.out, w.y, D)
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

// fastBuffers allocates the bf16 scratch Fast mode needs.
func (m *MetalDiT) fastBuffers() error {
	if len(m.kp16) > 0 {
		return nil
	}
	D, hid, T := m.cfg.dim(), m.cfg.dim()*m.cfg.MLPRatio, m.tw.T
	var err error
	nb := func(elems int) *metal.Buffer {
		b, e := m.dev.NewBuffer(2 * max(elems, 1))
		if err == nil {
			err = e
		}
		return b
	}
	w := m.tw
	w.y16, w.q16, w.k16, w.v16, w.s16 = nb(T*max(D, hid)), nb(T*D), nb(T*D), nb(T*D), nb(T*(m.pv+T))
	for range m.layers {
		m.kp16, m.vp16 = append(m.kp16, nb(m.pv*D)), append(m.vp16, nb(m.pv*D))
	}
	return err
}

// Close releases the GPU buffers (the mapped weights stay the caller's).
func (m *MetalDiT) Close() {
	for _, w := range []*metalWork{m.tw, m.pw} {
		if w != nil {
			w.release()
		}
	}
	for _, b := range []*metal.Buffer{m.lat, m.out, m.s1, m.g1, m.s2, m.g2, m.fs, m.pmask, m.pvalid} {
		if b != nil {
			b.Release()
		}
	}
	for i := range m.kp {
		m.kp[i].Release()
		m.vp[i].Release()
	}
	for i := range m.kp16 {
		m.kp16[i].Release()
		m.vp16[i].Release()
	}
	for _, lw := range m.lw {
		lw.normQ.Release()
		lw.normK.Release()
	}
	for _, b := range m.shards {
		b.Release()
	}
}
