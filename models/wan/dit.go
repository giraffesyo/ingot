package wan

import (
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/ops"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// DiTConfig is transformer/config.json (diffusers WanTransformer3DModel).
type DiTConfig struct {
	PatchSize     [3]int  `json:"patch_size"`
	NumHeads      int     `json:"num_attention_heads"`
	HeadDim       int     `json:"attention_head_dim"`
	InChannels    int     `json:"in_channels"`
	OutChannels   int     `json:"out_channels"`
	TextDim       int     `json:"text_dim"`
	FreqDim       int     `json:"freq_dim"`
	FFNDim        int     `json:"ffn_dim"`
	NumLayers     int     `json:"num_layers"`
	CrossAttnNorm bool    `json:"cross_attn_norm"`
	QKNorm        string  `json:"qk_norm"`
	Eps           float32 `json:"eps"`
	ImageDim      *int    `json:"image_dim"`
	AddedKVDim    *int    `json:"added_kv_proj_dim"`
	RopeMaxSeqLen int     `json:"rope_max_seq_len"`
}

// LoadDiTConfig reads dir/config.json and rejects variants not built here
// (Wan 2.1 I2V's CLIP image tokens, other patch or norm layouts).
func LoadDiTConfig(dir string) (DiTConfig, error) {
	var c DiTConfig
	if err := readJSON(dir, "config.json", &c); err != nil {
		return c, err
	}
	if c.PatchSize != [3]int{1, 2, 2} || c.QKNorm != "rms_norm_across_heads" || !c.CrossAttnNorm ||
		c.ImageDim != nil || c.AddedKVDim != nil || c.FreqDim%2 != 0 {
		return c, fmt.Errorf("wan: transformer variant not implemented (patch %v, qk_norm %q, cross_attn_norm %v, image_dim %v)",
			c.PatchSize, c.QKNorm, c.CrossAttnNorm, c.ImageDim)
	}
	if c.OutChannels == 0 {
		c.OutChannels = c.InChannels
	}
	if c.HeadDim%2 != 0 || 2*(c.HeadDim/6) < 2 {
		return c, fmt.Errorf("wan: head dim %d", c.HeadDim)
	}
	return c, nil
}

func (c DiTConfig) dim() int { return c.NumHeads * c.HeadDim }

// patchIn is the patch embedding's input width (channels × 2 × 2).
func (c DiTConfig) patchIn() int { return c.InChannels * c.PatchSize[1] * c.PatchSize[2] }

// Grid is a latent video's shape: frames × height × width latent cells;
// the transformer sees F·(H/2)·(W/2) tokens, frame-major.
type Grid struct{ F, H, W int }

// Tokens is the transformer sequence length.
func (g Grid) Tokens() int { return g.F * (g.H / 2) * (g.W / 2) }

// FrameTokens is the number of tokens per latent frame.
func (g Grid) FrameTokens() int { return (g.H / 2) * (g.W / 2) }

// Patchify turns latents [C, F, H, W] into the patch embedding's input
// [F·H/2·W/2, C·4] (feature order c, kh, kw — the Conv3d weight's).
func Patchify(x *tensor.Tensor, g Grid) *tensor.Tensor {
	C := x.Shape()[0]
	h2, w2 := g.H/2, g.W/2
	out := tensor.New(tensor.F32, g.Tokens(), C*4)
	src, dst := x.F32(), out.F32()
	for f := range g.F {
		for i := range h2 {
			for j := range w2 {
				row := dst[((f*h2+i)*w2+j)*C*4:]
				for c := range C {
					base := ((c*g.F+f)*g.H+2*i)*g.W + 2*j
					row[c*4+0], row[c*4+1] = src[base], src[base+1]
					row[c*4+2], row[c*4+3] = src[base+g.W], src[base+g.W+1]
				}
			}
		}
	}
	return out
}

// Unpatchify turns the transformer's output [tokens, 4·C] (feature order
// kh, kw, c — proj_out's) back into [C, F, H, W].
func Unpatchify(y *tensor.Tensor, g Grid) *tensor.Tensor {
	C := y.Shape()[1] / 4
	h2, w2 := g.H/2, g.W/2
	out := tensor.New(tensor.F32, C, g.F, g.H, g.W)
	src, dst := y.F32(), out.F32()
	for f := range g.F {
		for i := range h2 {
			for j := range w2 {
				row := src[((f*h2+i)*w2+j)*C*4:]
				for c := range C {
					base := ((c*g.F+f)*g.H+2*i)*g.W + 2*j
					dst[base], dst[base+1] = row[c], row[C+c]
					dst[base+g.W], dst[base+g.W+1] = row[2*C+c], row[3*C+c]
				}
			}
		}
	}
	return out
}

// ropeTables returns cos/sin [tokens, headDim/2] of WanRotaryPosEmbed: the
// head's channel pairs split between frame, row and column positions
// (h and w each take headDim/6 pairs, the frame axis the rest), each axis
// at frequencies 10000^(-2i/axisDim). Interleaved pairs (ingot.RoPE
// layout 0).
func ropeTables(c DiTConfig, g Grid) (cos, sin []float32) {
	hDim := 2 * (c.HeadDim / 6)
	tDim := c.HeadDim - 2*hDim
	axes := [3]int{tDim / 2, hDim / 2, hDim / 2}
	freqs := make([][]float64, 3)
	for a, n := range axes {
		dim := 2 * n
		for i := range n {
			freqs[a] = append(freqs[a], 1/math.Pow(10000, float64(2*i)/float64(dim)))
		}
	}
	half := c.HeadDim / 2
	T := g.Tokens()
	cos, sin = make([]float32, T*half), make([]float32, T*half)
	h2, w2 := g.H/2, g.W/2
	for f := range g.F {
		for i := range h2 {
			for j := range w2 {
				row := ((f*h2+i)*w2 + j) * half
				k := 0
				for a, pos := range [3]int{f, i, j} {
					for _, fr := range freqs[a] {
						ang := float64(pos) * fr
						cos[row+k], sin[row+k] = float32(math.Cos(ang)), float32(math.Sin(ang))
						k++
					}
				}
			}
		}
	}
	return cos, sin
}

// Timesteps describes the transformer's per-token timesteps: a few
// distinct values and each token's index into them. Text-to-video uses one
// value for all; image-to-video holds the first latent frame (the image's
// latent) at 0 and the rest at t.
type Timesteps struct {
	Values []float32
	Seg    []int64 // per token; nil when len(Values) == 1
}

// FirstFrameFixed is image-to-video's timesteps: first-frame tokens at 0,
// the others at t.
func FirstFrameFixed(g Grid, t float32) Timesteps {
	seg := make([]int64, g.Tokens())
	for i := g.FrameTokens(); i < len(seg); i++ {
		seg[i] = 1
	}
	return Timesteps{Values: []float32{0, t}, Seg: seg}
}

// crossKey and crossValue name block i's cross-attention keys and values:
// outputs of BuildTextCond, inputs of BuildDiT.
func crossKey(i int) string   { return fmt.Sprintf("blocks.%d.cross_k", i) }
func crossValue(i int) string { return fmt.Sprintf("blocks.%d.cross_v", i) }

// BuildTextCond builds the text half of the transformer: input "text"
// [MaxTextTokens, text_dim] (umT5 embeddings, zero-padded); outputs every
// block's cross-attention keys (to_k, RMS-normalised) and values, [1, L,
// heads, head_dim]. They depend on the prompt alone, so a generation
// computes them once per prompt instead of once per step.
func BuildTextCond(cfg DiTConfig, set *safetensors.Set, blocks int, widen bool) (g *graph.Graph, err error) {
	defer catch(&err)
	w := weights{set: set, widen: widen}
	b := graph.NewBuilder("wan_text_cond")
	L, H, dh := int64(MaxTextTokens), int64(cfg.NumHeads), int64(cfg.HeadDim)
	x := b.Input("text", tensor.F32, MaxTextTokens, cfg.TextDim)
	te := w.scope("condition_embedder.text_embedder")
	wt, bias := te.linear("linear_1")
	x = b.GeluTanh(b.Linear(x, wt, bias))
	wt, bias = te.linear("linear_2")
	x = b.Linear(x, wt, bias)
	for i := range blocks {
		bl, wa := b.Scope(fmt.Sprintf("blocks.%d.attn2", i)), w.scope(fmt.Sprintf("blocks.%d.attn2", i))
		wt, bias = wa.linear("to_k")
		k := bl.RMSNorm(bl.Linear(x, wt, bias), wa.f32("norm_k.weight"), cfg.Eps)
		wt, bias = wa.linear("to_v")
		v := bl.Linear(x, wt, bias)
		b.Output(crossKey(i), bl.Reshape(k, 1, L, H, dh))
		b.Output(crossValue(i), bl.Reshape(v, 1, L, H, dh))
	}
	return b.Build()
}

// dit holds what the transformer graph shares while building.
type dit struct {
	cfg DiTConfig
	w   weights
	seg *graph.Value // per-token segment index, nil for one timestep
}

// perToken expands a per-segment [S, D] value to the tokens: a Gather by
// segment index, or the [1, D] value itself (broadcast) for one segment.
func (m *dit) perToken(b *graph.Builder, v *graph.Value) *graph.Value {
	if m.seg == nil {
		return v
	}
	return b.Op("Gather", graph.Attr("axis", 0), v, m.seg)
}

// BuildDiT builds one transformer evaluation over grid g: inputs "x"
// [tokens, in_channels·4] (Patchify), "t" [S] the distinct timesteps,
// "seg" [tokens] int64 each token's index into t (only when S > 1), and
// every block's cross-attention keys and values (BuildTextCond's outputs);
// output "v" [tokens, out_channels·4] (Unpatchify), the flow velocity.
// blocks < NumLayers builds a truncated model (tests). widen hands the
// matrix products f32 weights where the checkpoint stores bf16.
func BuildDiT(cfg DiTConfig, set *safetensors.Set, g Grid, S, blocks int, widen bool) (gr *graph.Graph, err error) {
	defer catch(&err)
	if g.H%2 != 0 || g.W%2 != 0 {
		return nil, fmt.Errorf("wan: latent %dx%d not a multiple of the 2x2 patch", g.H, g.W)
	}
	m := &dit{cfg: cfg, w: weights{set: set, widen: widen}}
	b := graph.NewBuilder("wan_dit")
	T, D, dh := g.Tokens(), cfg.dim(), cfg.HeadDim
	x := b.Input("x", tensor.F32, T, cfg.patchIn())
	t := b.Input("t", tensor.F32, S)
	if S > 1 {
		m.seg = b.Input("seg", tensor.I64, T)
	}
	pw := m.w.matrix("patch_embedding.weight")
	h := b.Scope("patch_embedding").Linear(x, pw.Reshape(D, cfg.patchIn()), m.w.f32("patch_embedding.bias"))

	temb, tproj := m.timeEmbed(b.Scope("condition_embedder"), t, S)
	c, s := ropeTables(cfg, g)
	cos := b.Const("rope_cos", tensor.FromF32(c, T, dh/2))
	sin := b.Const("rope_sin", tensor.FromF32(s, T, dh/2))
	for i := range blocks {
		k := b.Input(crossKey(i), tensor.F32, 1, MaxTextTokens, cfg.NumHeads, dh)
		v := b.Input(crossValue(i), tensor.F32, 1, MaxTextTokens, cfg.NumHeads, dh)
		h = m.block(b.Scope(fmt.Sprintf("blocks.%d", i)), m.w.scope(fmt.Sprintf("blocks.%d", i)), h, tproj, k, v, T, cos, sin)
	}
	// Output modulation: shift, scale = scale_shift_table + temb.
	tab := m.w.f32("scale_shift_table").F32() // [1, 2, D]
	shiftV := b.Add(b.Const("out_shift_table", tensor.FromF32(tab[:D], 1, D)), temb)
	scaleV := b.Add(b.Const("out_scale_table", tensor.FromF32(addOne(tab[D:2*D]), 1, D)), temb)
	h = b.Add(b.Mul(b.LayerNorm(h, D, nil, nil, cfg.Eps), m.perToken(b, scaleV)), m.perToken(b, shiftV))
	wt, bias := m.w.linear("proj_out")
	b.Output("v", b.Scope("proj_out").Linear(h, wt, bias))
	return b.Build()
}

// addOne returns a copy of v with 1 added (the "1 + scale" folded into the
// scale table).
func addOne(v []float32) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x + 1
	}
	return out
}

// timeEmbed is WanTimeTextImageEmbedding's time half over the S distinct
// timesteps: [cos, sin] of t·10000^(-i/half) → linear_1 → SiLU → linear_2
// = temb [S, D]; time_proj(SiLU(temb)) = tproj [S, 6·D].
func (m *dit) timeEmbed(b *graph.Builder, t *graph.Value, S int) (temb, tproj *graph.Value) {
	half := m.cfg.FreqDim / 2
	freqs := tensor.New(tensor.F32, 1, half)
	for i := range half {
		freqs.F32()[i] = float32(math.Exp(float64(float32(-math.Log(10000)) * float32(i) / float32(half))))
	}
	args := b.Mul(b.Reshape(t, int64(S), 1), b.Const("freqs", freqs))
	emb := b.Concat(1, b.Op("Cos", nil, args), b.Op("Sin", nil, args))
	wt, bias := m.w.linear("condition_embedder.time_embedder.linear_1")
	h := b.SiLU(b.Linear(emb, wt, bias))
	wt, bias = m.w.linear("condition_embedder.time_embedder.linear_2")
	temb = b.Linear(h, wt, bias)
	wt, bias = m.w.linear("condition_embedder.time_proj")
	return temb, b.Linear(b.SiLU(temb), wt, bias)
}

// sdpaAttrs: q, k, v and the output all [1, T, H, dh], scale 1/√dh.
func (m *dit) sdpaAttrs() ops.Attrs {
	return graph.Attr("scale", float32(1/math.Sqrt(float64(m.cfg.HeadDim))),
		"a_layout", 1, "b_layout", 1, "v_layout", 1, "stride_out", 1)
}

// block is WanTransformerBlock over x [T, D]: modulated self-attention
// with 3-D rotary positions, cross-attention to the text (keys and values
// from BuildTextCond), modulated GELU feed-forward — the six modulation
// vectors per timestep segment from scale_shift_table + tproj.
func (m *dit) block(b *graph.Builder, w weights, x, tproj, ck, cv *graph.Value, T int, cos, sin *graph.Value) *graph.Value {
	D, H, dh := m.cfg.dim(), int64(m.cfg.NumHeads), int64(m.cfg.HeadDim)
	eps := m.cfg.Eps
	tab := w.f32("scale_shift_table").F32() // [1, 6, D]
	mod := func(i int, plusOne bool) *graph.Value {
		row := tab[i*D : (i+1)*D]
		if plusOne {
			row = addOne(row)
		}
		v := b.Add(b.Const(fmt.Sprintf("mod%d", i), tensor.FromF32(row, 1, D)), b.Slice(tproj, 1, int64(i*D), int64((i+1)*D)))
		return m.perToken(b, v)
	}
	heads := func(sb *graph.Builder, v *graph.Value) *graph.Value { return sb.Reshape(v, int64(T), H, dh) }
	batch := func(sb *graph.Builder, v *graph.Value) *graph.Value { return sb.Reshape(v, 1, int64(T), H, dh) }

	sa, ws := b.Scope("attn1"), w.scope("attn1")
	y := b.Add(b.Mul(b.LayerNorm(x, D, nil, nil, eps), mod(1, true)), mod(0, false))
	wt, bias := ws.linear("to_q")
	q := sa.RMSNorm(sa.Linear(y, wt, bias), ws.f32("norm_q.weight"), eps)
	wt, bias = ws.linear("to_k")
	k := sa.RMSNorm(sa.Linear(y, wt, bias), ws.f32("norm_k.weight"), eps)
	wt, bias = ws.linear("to_v")
	v := batch(sa, sa.Linear(y, wt, bias))
	q = batch(sa, sa.Op("ingot.RoPE", graph.Attr("layout", 0), heads(sa, q), cos, sin))
	k = batch(sa, sa.Op("ingot.RoPE", graph.Attr("layout", 0), heads(sa, k), cos, sin))
	o := sa.Reshape(sa.Op("ingot.SDPA", m.sdpaAttrs(), q, k, v), int64(T), int64(D))
	wt, bias = ws.linear("to_out.0")
	x = b.Add(x, b.Mul(sa.Linear(o, wt, bias), mod(2, false)))

	ca, wc := b.Scope("attn2"), w.scope("attn2")
	y = b.LayerNorm(x, D, w.f32("norm2.weight"), w.f32("norm2.bias"), eps)
	wt, bias = wc.linear("to_q")
	q = batch(ca, ca.RMSNorm(ca.Linear(y, wt, bias), wc.f32("norm_q.weight"), eps))
	o = ca.Reshape(ca.Op("ingot.SDPA", m.sdpaAttrs(), q, ck, cv), int64(T), int64(D))
	wt, bias = wc.linear("to_out.0")
	x = b.Add(x, ca.Linear(o, wt, bias))

	fb, wf := b.Scope("ffn"), w.scope("ffn.net")
	y = b.Add(b.Mul(b.LayerNorm(x, D, nil, nil, eps), mod(4, true)), mod(3, false))
	wt, bias = wf.linear("0.proj")
	y = fb.GeluTanh(fb.Linear(y, wt, bias))
	wt, bias = wf.linear("2")
	return b.Add(x, b.Mul(fb.Linear(y, wt, bias), mod(5, false)))
}
