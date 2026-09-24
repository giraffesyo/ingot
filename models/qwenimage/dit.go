package qwenimage

import (
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/ops"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// DiTConfig is transformer/config.json (QwenImage21Transformer2DModel).
type DiTConfig struct {
	PatchSize         int   `json:"patch_size"`
	InChannels        int   `json:"in_channels"`
	OutChannels       int   `json:"out_channels"`
	NumLayers         int   `json:"num_layers"`
	AttentionHeadDim  int   `json:"attention_head_dim"`
	NumAttentionHeads int   `json:"num_attention_heads"`
	ContextInDim      int   `json:"context_in_dim"`
	MLPRatio          int   `json:"mlp_ratio"`
	AxesDimsRope      []int `json:"axes_dims_rope"`
	CausalCondition   bool  `json:"causal_condition"`
	Eps               float32
}

// LoadDiTConfig reads dir/config.json.
func LoadDiTConfig(dir string) (DiTConfig, error) {
	var c struct {
		DiTConfig
		Eps float32 `json:"eps"`
	}
	if err := readConfig(dir, &c); err != nil {
		return c.DiTConfig, err
	}
	c.DiTConfig.Eps = c.Eps
	if c.OutChannels == 0 {
		c.OutChannels = c.InChannels
	}
	if c.PatchSize != 1 || !c.CausalCondition {
		return c.DiTConfig, fmt.Errorf("qwenimage: DiT config variant not implemented (patch_size=%d causal_condition=%v)", c.PatchSize, c.CausalCondition)
	}
	if s := c.AxesDimsRope; len(s) != 3 || s[0]+s[1]+s[2] != c.AttentionHeadDim {
		return c.DiTConfig, fmt.Errorf("qwenimage: axes_dims_rope %v must sum to head dim %d", s, c.AttentionHeadDim)
	}
	return c.DiTConfig, nil
}

func (c DiTConfig) dim() int { return c.NumAttentionHeads * c.AttentionHeadDim }

// imgTokensPerSlot: each vision-language image slot stands for a 2×2 group
// of latent tokens (_IMG_TOKENS_PER_SLOT).
const imgTokensPerSlot = 4

// DiTLayout is the joint token sequence of one generation, derived exactly
// as QwenImage21Transformer2DModel.forward does: the encoder's rows, with
// every image slot expanded to 2×2 latent tokens, and the target image's
// slots appended. It splits the sequence the way prefix-KV caching does:
// the prefix (text + condition images, modulated from t=0, never attending
// to the target) runs once; the target runs every step over the cached
// prefix K/V.
type DiTLayout struct {
	TxtRows int // encoder rows (vision-language sequence length)
	CondTok int // condition-image latent tokens
	Prefix  int // prefix tokens (joint sequence minus the target)
	Target  int // target latent tokens (h·w)
	TargetH int
	TargetW int

	prefixSrc   []int64   // prefix token → row of concat(txt_in(txt), img_in(cond))
	prefixValid []int64   // prefix tokens whose keys are not padding
	prefixMask  []float32 // [Prefix, Prefix] additive block-causal + key-valid mask
	prefixCos   []float32 // [Prefix, headDim/2] rotary tables
	prefixSin   []float32
	targetCos   []float32 // [Target, headDim/2]
	targetSin   []float32
}

// NewDiTLayout builds the layout. imgMask marks the encoder's image slots
// over the encoder rows *plus* the target's appended slots (target/4 of
// them, all true); txtValid marks non-padding encoder rows (len TxtRows,
// nil = all valid); imgShapes lists (frames, h, w) in latent tokens per
// image block, condition images first and the target last.
func NewDiTLayout(cfg DiTConfig, imgMask, txtValid []bool, imgShapes [][3]int) (*DiTLayout, error) {
	if len(imgShapes) == 0 {
		return nil, fmt.Errorf("qwenimage: layout needs at least the target image shape")
	}
	tgt := imgShapes[len(imgShapes)-1]
	target := tgt[0] * tgt[1] * tgt[2]
	if tgt[0] != 1 || target%imgTokensPerSlot != 0 {
		return nil, fmt.Errorf("qwenimage: target shape %v: want one frame, h·w divisible by %d", tgt, imgTokensPerSlot)
	}
	txtRows := len(imgMask) - target/imgTokensPerSlot
	if txtRows < 0 {
		return nil, fmt.Errorf("qwenimage: img_mask length %d shorter than the target's %d slots", len(imgMask), target/imgTokensPerSlot)
	}
	for _, m := range imgMask[txtRows:] {
		if !m {
			return nil, fmt.Errorf("qwenimage: img_mask must end in the target's image slots")
		}
	}
	if txtValid != nil && len(txtValid) != txtRows {
		return nil, fmt.Errorf("qwenimage: txt mask length %d, want %d encoder rows", len(txtValid), txtRows)
	}

	// Expand slots: the joint sequence, each token's source and validity.
	var isImg []bool
	var src []int64 // text: encoder row; image: -1 (filled below)
	var valid []bool
	for p, m := range imgMask {
		if m {
			for range imgTokensPerSlot {
				isImg, src, valid = append(isImg, true), append(src, -1), append(valid, true)
			}
			continue
		}
		isImg, src = append(isImg, false), append(src, int64(p))
		valid = append(valid, txtValid == nil || txtValid[p])
	}
	L := len(isImg)
	nImg := 0
	for _, m := range isImg {
		if m {
			nImg++
		}
	}
	total := 0
	for _, s := range imgShapes {
		total += s[0] * s[1] * s[2]
	}
	if total != nImg {
		return nil, fmt.Errorf("qwenimage: img_shapes account for %d image tokens, img_mask for %d", total, nImg)
	}
	// Image tokens take latent rows in order (conditions, then target);
	// ids label each image block for the block-causal mask.
	ids := make([]int, L)
	k, blk, left := 0, 0, imgShapes[0][0]*imgShapes[0][1]*imgShapes[0][2]
	for i := range L {
		ids[i] = -1
		if !isImg[i] {
			continue
		}
		for left == 0 {
			blk++
			left = imgShapes[blk][0] * imgShapes[blk][1] * imgShapes[blk][2]
		}
		ids[i] = blk
		src[i] = int64(txtRows + k)
		k++
		left--
	}
	prefix := L - target
	for i := prefix; i < L; i++ {
		if ids[i] != len(imgShapes)-1 {
			return nil, fmt.Errorf("qwenimage: target tokens must end the sequence")
		}
	}

	pos, err := ropePositions(isImg, imgShapes)
	if err != nil {
		return nil, err
	}
	l := &DiTLayout{TxtRows: txtRows, CondTok: total - target, Prefix: prefix, Target: target, TargetH: tgt[1], TargetW: tgt[2]}
	l.prefixSrc = src[:prefix]
	for i := range prefix {
		if valid[i] {
			l.prefixValid = append(l.prefixValid, int64(i))
		}
	}
	negInf := float32(math.Inf(-1))
	l.prefixMask = make([]float32, prefix*prefix)
	for q := range prefix {
		for kk := range prefix {
			ok := (q >= kk || (ids[q] >= 0 && ids[q] == ids[kk])) && valid[kk]
			if !ok {
				l.prefixMask[q*prefix+kk] = negInf
			}
		}
	}
	l.prefixCos, l.prefixSin = ropeTables(cfg, pos[:prefix])
	l.targetCos, l.targetSin = ropeTables(cfg, pos[prefix:])
	return l, nil
}

// ropePositions is QwenImage21Rope.forward's index bookkeeping: text
// advances one shared position on all three axes; each image block freezes
// the frame axis at the position reached and lays its tokens on an h×w grid
// centred on zero, then advances the position by max(h, w).
func ropePositions(isImg []bool, imgShapes [][3]int) ([][3]int, error) {
	L := len(isImg)
	pos := make([][3]int, 0, L)
	cursor, position := 0, 0
	for _, s := range imgShapes {
		h, w := s[1], s[2]
		start := cursor
		for start < L && !isImg[start] {
			start++
		}
		if start == L {
			return nil, fmt.Errorf("qwenimage: fewer image blocks in img_mask than img_shapes")
		}
		for i := cursor; i < start; i++ {
			pos = append(pos, [3]int{position, position, position})
			position++
		}
		for hh := -(h - h/2); hh < h/2; hh++ {
			for ww := -(w - w/2); ww < w/2; ww++ {
				pos = append(pos, [3]int{position, hh, ww})
			}
		}
		cursor = start + h*w
		position += max(h, w)
	}
	for i := cursor; i < L; i++ {
		pos = append(pos, [3]int{position, position, position})
		position++
	}
	return pos, nil
}

// ropeTables returns cos/sin [len(pos), headDim/2]: the frame, height and
// width axes' pair angles concatenated. Computed in float32 like the
// reference (theta^(-2i/d) and pos·inv_freq rounded to f32), so the tables
// match its complex polar() values to the ulp.
func ropeTables(cfg DiTConfig, pos [][3]int) (cos, sin []float32) {
	half := cfg.AttentionHeadDim / 2
	inv := make([]float32, 0, half)
	for _, d := range cfg.AxesDimsRope {
		for i := 0; i < d; i += 2 {
			e := float32(i) / float32(d)
			inv = append(inv, 1/float32(math.Pow(10000, float64(e))))
		}
	}
	cos, sin = make([]float32, len(pos)*half), make([]float32, len(pos)*half)
	for t, p := range pos {
		j := 0
		for ax, d := range cfg.AxesDimsRope {
			for range d / 2 {
				a := float64(float32(p[ax]) * inv[j])
				cos[t*half+j], sin[t*half+j] = float32(math.Cos(a)), float32(math.Sin(a))
				j++
			}
		}
	}
	return cos, sin
}

// PrefixFeeds returns the prefix graph's inputs: encoder hidden states
// [TxtRows, context_in_dim] and packed condition latents [CondTok, in]
// (nil when there are none).
func (l *DiTLayout) PrefixFeeds(txt, cond *tensor.Tensor) map[string]*tensor.Tensor {
	f := map[string]*tensor.Tensor{"txt": txt}
	if cond != nil {
		f["cond"] = cond
	}
	return f
}

// TargetFeeds returns the target graph's inputs for one step: packed target
// latents [Target, in], the timestep in [0, 1], and the prefix K/V from
// the prefix graph's outputs (k0, v0, k1, …).
func (l *DiTLayout) TargetFeeds(x *tensor.Tensor, t float32, prefixKV map[string]*tensor.Tensor) map[string]*tensor.Tensor {
	f := map[string]*tensor.Tensor{"x": x, "t": tensor.FromF32([]float32{t}, 1)}
	for k, v := range prefixKV {
		f[k] = v
	}
	return f
}

// dit holds what both graphs share while building.
type dit struct {
	cfg    DiTConfig
	w      weights
	layers int
}

// BuildDiTPrefix builds the prefix pass: inputs "txt" (and "cond" when the
// layout has condition images); outputs per-layer "k<i>", "v<i>" [valid
// prefix tokens, dim] — post-norm, post-RoPE keys and values with padding
// rows dropped, exactly the reference's KV cache minus keys every query
// masks — and, if withOut, "out" [Prefix, out_channels], the model output
// for the prefix rows. layers < cfg.NumLayers builds a truncated model.
func BuildDiTPrefix(cfg DiTConfig, set *safetensors.Set, l *DiTLayout, layers int, withOut bool) (g *graph.Graph, err error) {
	defer catch(&err)
	d := &dit{cfg: cfg, w: weights{set: set}, layers: layers}
	b := graph.NewBuilder("qwenimage21_dit_prefix")
	D := cfg.dim()
	txt := b.Input("txt", tensor.F32, l.TxtRows, cfg.ContextInDim)
	h := d.txtIn(b.Scope("txt_in"), txt)
	if l.CondTok > 0 {
		cond := b.Input("cond", tensor.F32, l.CondTok, cfg.InChannels)
		h = b.Concat(0, h, b.Scope("img_in").Linear(cond, d.w.raw("img_in.weight"), nil))
	}
	x := b.Op("Gather", graph.Attr("axis", 0), h, b.Const("prefix_src", tensor.FromI64(l.prefixSrc, l.Prefix)))
	temb := d.timeEmbed(b.Scope("time"), b.Scalar(0))
	mod := d.modulation(b.Scope("modulation"), temb)
	cos, sin := ropeConsts(b, l.prefixCos, l.prefixSin, l.Prefix, cfg.AttentionHeadDim/2)
	mask := b.Const("mask", tensor.FromF32(l.prefixMask, l.Prefix, l.Prefix))
	valid := b.Const("valid", tensor.FromI64(l.prefixValid, len(l.prefixValid)))
	for i := range layers {
		bi := b.Scope(fmt.Sprintf("blocks.%d", i))
		wi := d.w.scope(fmt.Sprintf("transformer_blocks.%d", i))
		x = d.block(bi, wi, x, mod, l.Prefix, cos, sin, func(q, k, v *graph.Value) *graph.Value {
			b.Output(fmt.Sprintf("k%d", i), bi.Op("Gather", graph.Attr("axis", 0), bi.Reshape(k, int64(l.Prefix), int64(D)), valid))
			b.Output(fmt.Sprintf("v%d", i), bi.Op("Gather", graph.Attr("axis", 0), bi.Reshape(v, int64(l.Prefix), int64(D)), valid))
			return bi.Op("ingot.SDPA", d.sdpaAttrs(), q, k, v, mask)
		})
	}
	if withOut {
		b.Output("out", d.final(b.Scope("final"), x, temb))
	}
	return b.Build()
}

// BuildDiTTarget builds one denoising step for the target tokens: inputs
// "x" [Target, in], "t" [1] and the prefix K/V "k<i>", "v<i>"; output
// "out" [Target, out_channels]. Attention runs over [prefix, target] keys
// with no mask (padding was dropped from the prefix K/V).
func BuildDiTTarget(cfg DiTConfig, set *safetensors.Set, l *DiTLayout, layers int) (g *graph.Graph, err error) {
	defer catch(&err)
	d := &dit{cfg: cfg, w: weights{set: set}, layers: layers}
	b := graph.NewBuilder("qwenimage21_dit_target")
	D, H, dh := cfg.dim(), cfg.NumAttentionHeads, cfg.AttentionHeadDim
	pv := int64(len(l.prefixValid))
	x := b.Scope("img_in").Linear(b.Input("x", tensor.F32, l.Target, cfg.InChannels), d.w.raw("img_in.weight"), nil)
	temb := d.timeEmbed(b.Scope("time"), b.Input("t", tensor.F32, 1))
	mod := d.modulation(b.Scope("modulation"), temb)
	cos, sin := ropeConsts(b, l.targetCos, l.targetSin, l.Target, dh/2)
	for i := range layers {
		bi := b.Scope(fmt.Sprintf("blocks.%d", i))
		wi := d.w.scope(fmt.Sprintf("transformer_blocks.%d", i))
		pk := b.Input(fmt.Sprintf("k%d", i), tensor.F32, int(pv), D)
		pvv := b.Input(fmt.Sprintf("v%d", i), tensor.F32, int(pv), D)
		x = d.block(bi, wi, x, mod, l.Target, cos, sin, func(q, k, v *graph.Value) *graph.Value {
			k = bi.Concat(1, bi.Reshape(pk, 1, pv, int64(H), int64(dh)), k)
			v = bi.Concat(1, bi.Reshape(pvv, 1, pv, int64(H), int64(dh)), v)
			return bi.Op("ingot.SDPA", d.sdpaAttrs(), q, k, v)
		})
	}
	b.Output("out", d.final(b.Scope("final"), x, temb))
	return b.Build()
}

// sdpaAttrs: q, k, v and the output all in [B, T, H, dh] (the projections'
// natural layout — no transposes), softmax scale 1/√dh.
func (d *dit) sdpaAttrs() ops.Attrs {
	return graph.Attr("scale", float32(1/math.Sqrt(float64(d.cfg.AttentionHeadDim))),
		"a_layout", 1, "b_layout", 1, "v_layout", 1, "stride_out", 1)
}

// txtIn is QwenImage21TextProjection: zero-centred RMSNorm (scale w+1) →
// Linear → GELU(tanh) → Linear.
func (d *dit) txtIn(b *graph.Builder, x *graph.Value) *graph.Value {
	w := d.w.scope("txt_in")
	norm := w.f32("text_norm.weight")
	scale := tensor.New(tensor.F32, norm.Shape()...)
	for i, v := range norm.F32() {
		scale.F32()[i] = v + 1
	}
	x = b.RMSNorm(x, scale, d.cfg.Eps)
	x = b.GeluTanh(b.Linear(x, w.raw("in_layer.weight"), nil))
	return b.Linear(x, w.raw("out_layer.weight"), nil)
}

// timeEmbed is QwenImage21TimestepProjEmbeddings for one timestep t [1]:
// [cos, sin] of 1000·t·freqs (256 channels) → Linear → SiLU → Linear.
func (d *dit) timeEmbed(b *graph.Builder, t *graph.Value) *graph.Value {
	const half = 128
	freqs := tensor.New(tensor.F32, 1, half)
	for i := range half {
		freqs.F32()[i] = float32(math.Exp(float64(float32(-math.Log(10000)) * float32(i) / half)))
	}
	args := b.Mul(b.Reshape(b.Mul(t, b.Scalar(1000)), 1, 1), b.Const("freqs", freqs))
	emb := b.Concat(1, b.Op("Cos", nil, args), b.Op("Sin", nil, args))
	w := d.w.scope("time_text_embed.timestep_embedder")
	h := b.SiLU(b.Linear(emb, w.raw("linear_1.weight"), nil))
	return b.Linear(h, w.raw("linear_2.weight"), nil)
}

// mods are the shared per-block modulation terms, broadcast over tokens:
// (1+scale) and tanh(gate) for the attention and MLP halves.
type mods struct{ s1, g1, s2, g2 *graph.Value }

// modulation is the model's one shared projection SiLU → Linear(dim → 4·dim),
// chunked as [scale1 | gate1 | scale2 | gate2].
func (d *dit) modulation(b *graph.Builder, temb *graph.Value) mods {
	D := int64(d.cfg.dim())
	m := b.Linear(b.SiLU(temb), d.w.raw("modulation.1.weight"), nil) // [1, 4D]
	one := b.Scalar(1)
	return mods{
		s1: b.Add(b.Slice(m, 1, 0, D), one),
		g1: b.Tanh(b.Slice(m, 1, D, 2*D)),
		s2: b.Add(b.Slice(m, 1, 2*D, 3*D), one),
		g2: b.Tanh(b.Slice(m, 1, 3*D, 4*D)),
	}
}

// ropeConsts adds the rotary tables [T, half] as constants.
func ropeConsts(b *graph.Builder, cos, sin []float32, t, half int) (*graph.Value, *graph.Value) {
	return b.Const("rope_cos", tensor.FromF32(cos, t, half)), b.Const("rope_sin", tensor.FromF32(sin, t, half))
}

// rope applies apply_rotary_emb_qwen (use_real=False): consecutive channel
// pairs of each head of x [T, H·dh] rotate as complex numbers by the
// token's angles (fused ingot.RoPE). Returns [1, T, H, dh].
func (d *dit) rope(b *graph.Builder, x, cos, sin *graph.Value, t int) *graph.Value {
	H, dh := int64(d.cfg.NumAttentionHeads), int64(d.cfg.AttentionHeadDim)
	x = b.Op("ingot.RoPE", graph.Attr("layout", 0), b.Reshape(x, int64(t), H, dh), cos, sin)
	return b.Reshape(x, 1, int64(t), H, dh)
}

// block is QwenImage21TransformerBlock over x [T, dim]. attend receives
// q, k (post-norm, post-RoPE) and v, each [1, T, H, dh], and returns the
// attention output [1, T, H, dh] — the prefix and target graphs differ only
// in what keys it attends over.
func (d *dit) block(b *graph.Builder, w weights, x *graph.Value, m mods, t int, cos, sin *graph.Value,
	attend func(q, k, v *graph.Value) *graph.Value) *graph.Value {
	D, H, dh := d.cfg.dim(), int64(d.cfg.NumAttentionHeads), int64(d.cfg.AttentionHeadDim)
	eps := d.cfg.Eps

	ab := b.Scope("attn")
	y := b.Mul(b.LayerNorm(x, D, nil, nil, eps), m.s1)
	q := ab.Linear(y, w.raw("attn.to_q.weight"), nil)
	k := ab.Linear(y, w.raw("attn.to_k.weight"), nil)
	v := ab.Linear(y, w.raw("attn.to_v.weight"), nil)
	q = ab.RMSNorm(ab.Reshape(q, int64(t), H, dh), w.f32("attn.norm_q.weight"), eps)
	k = ab.RMSNorm(ab.Reshape(k, int64(t), H, dh), w.f32("attn.norm_k.weight"), eps)
	q = d.rope(ab, ab.Reshape(q, int64(t), -1), cos, sin, t)
	k = d.rope(ab, ab.Reshape(k, int64(t), -1), cos, sin, t)
	v = ab.Reshape(v, 1, int64(t), H, dh)
	o := ab.Reshape(attend(q, k, v), int64(t), int64(D))
	o = ab.Linear(o, w.raw("attn.to_out.0.weight"), nil)
	x = b.Add(x, b.Mul(m.g1, o))

	mb := b.Scope("mlp")
	y = b.Mul(b.LayerNorm(x, D, nil, nil, eps), m.s2)
	gate := mb.SiLU(mb.Linear(y, w.raw("img_mlp.gate_layer.weight"), nil))
	proj := mb.Linear(y, w.raw("img_mlp.proj.weight"), nil)
	y = mb.Linear(mb.Mul(gate, proj), w.raw("img_mlp.out.weight"), nil)
	return b.Add(x, b.Mul(m.g2, y))
}

// final is norm_out (AdaLayerNormContinuous, scale only) → proj_out.
func (d *dit) final(b *graph.Builder, x, temb *graph.Value) *graph.Value {
	scale := b.Linear(b.SiLU(temb), d.w.raw("norm_out.linear.weight"), nil)
	x = b.Mul(b.LayerNorm(x, d.cfg.dim(), nil, nil, d.cfg.Eps), b.Add(scale, b.Scalar(1)))
	return b.Linear(x, d.w.raw("proj_out.weight"), nil)
}

// BuildDiTModulation builds the per-step conditioning the target pass
// broadcasts over tokens, for backends that run the blocks themselves:
// input "t" [1]; outputs "s1", "g1", "s2", "g2" (1+scale and tanh(gate) for
// the attention and MLP halves, shared by every block) and "fs" (1+scale of
// norm_out), each [1, dim].
func BuildDiTModulation(cfg DiTConfig, set *safetensors.Set) (g *graph.Graph, err error) {
	defer catch(&err)
	d := &dit{cfg: cfg, w: weights{set: set}}
	b := graph.NewBuilder("qwenimage21_dit_modulation")
	temb := d.timeEmbed(b.Scope("time"), b.Input("t", tensor.F32, 1))
	m := d.modulation(b.Scope("modulation"), temb)
	b.Output("s1", m.s1)
	b.Output("g1", m.g1)
	b.Output("s2", m.s2)
	b.Output("g2", m.g2)
	fs := b.Linear(b.SiLU(temb), d.w.raw("norm_out.linear.weight"), nil)
	b.Output("fs", b.Add(fs, b.Scalar(1)))
	return b.Build()
}

// BuildDiTTextIn builds the prefix's input stage for backends that run the
// blocks themselves: inputs "txt" (and "cond" when the layout has condition
// images); output "x" [Prefix, dim] — txt_in(txt) and img_in(cond) gathered
// into joint prefix order.
func BuildDiTTextIn(cfg DiTConfig, set *safetensors.Set, l *DiTLayout) (g *graph.Graph, err error) {
	defer catch(&err)
	d := &dit{cfg: cfg, w: weights{set: set}}
	b := graph.NewBuilder("qwenimage21_dit_text_in")
	h := d.txtIn(b.Scope("txt_in"), b.Input("txt", tensor.F32, l.TxtRows, cfg.ContextInDim))
	if l.CondTok > 0 {
		cond := b.Input("cond", tensor.F32, l.CondTok, cfg.InChannels)
		h = b.Concat(0, h, b.Scope("img_in").Linear(cond, d.w.raw("img_in.weight"), nil))
	}
	b.Output("x", b.Op("Gather", graph.Attr("axis", 0), h, b.Const("prefix_src", tensor.FromI64(l.prefixSrc, l.Prefix))))
	return b.Build()
}
