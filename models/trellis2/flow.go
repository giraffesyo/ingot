package trellis2

import (
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/ops"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// FlowConfig is the "args" of a flow-model checkpoint's JSON
// (SparseStructureFlowModel and SLatFlowModel share it).
type FlowConfig struct {
	Resolution     int     `json:"resolution"`
	InChannels     int     `json:"in_channels"`
	OutChannels    int     `json:"out_channels"`
	ModelChannels  int     `json:"model_channels"`
	CondChannels   int     `json:"cond_channels"`
	NumBlocks      int     `json:"num_blocks"`
	NumHeads       int     `json:"num_heads"`
	MLPRatio       float64 `json:"mlp_ratio"`
	PEMode         string  `json:"pe_mode"`
	ShareMod       bool    `json:"share_mod"`
	QKRMSNorm      bool    `json:"qk_rms_norm"`
	QKRMSNormCross bool    `json:"qk_rms_norm_cross"`
	// ImageAttnMode is "cross" (or empty): cross-attention to the image
	// tokens; or "proj": cross-attention to the image's global tokens plus,
	// per token, a linear projection of image features sampled where the
	// token's cell projects into the image (Pixal3D).
	ImageAttnMode  string `json:"image_attn_mode"`
	ProjInChannels int    `json:"proj_in_channels"`
}

// ProjChannels is the width of the per-token projected image features, or
// 0 for a model without them.
func (c FlowConfig) ProjChannels() int {
	if c.ImageAttnMode != "proj" {
		return 0
	}
	if c.ProjInChannels != 0 {
		return c.ProjInChannels
	}
	return c.CondChannels
}

// LoadFlowConfig reads a flow checkpoint's JSON (path without extension +
// ".json") and rejects variants this package does not build.
func LoadFlowConfig(path string) (FlowConfig, error) {
	var c struct {
		Name string     `json:"name"`
		Args FlowConfig `json:"args"`
	}
	if err := readJSON(path, &c); err != nil {
		return c.Args, err
	}
	a := c.Args
	if c.Name != "SparseStructureFlowModel" && c.Name != "SLatFlowModel" && c.Name != "ElasticSLatFlowModel" {
		return a, fmt.Errorf("trellis2: %s: model %q is not a flow model", path, c.Name)
	}
	if a.PEMode != "rope" || !a.ShareMod || !a.QKRMSNorm || !a.QKRMSNormCross {
		return a, fmt.Errorf("trellis2: %s: flow variant not implemented (pe_mode=%q share_mod=%v qk_rms_norm=%v/%v)",
			path, a.PEMode, a.ShareMod, a.QKRMSNorm, a.QKRMSNormCross)
	}
	if a.ImageAttnMode != "" && a.ImageAttnMode != "cross" && a.ImageAttnMode != "proj" {
		return a, fmt.Errorf("trellis2: %s: image_attn_mode %q not implemented", path, a.ImageAttnMode)
	}
	if a.NumHeads == 0 || a.ModelChannels%a.NumHeads != 0 || (a.ModelChannels/a.NumHeads)%2 != 0 {
		return a, fmt.Errorf("trellis2: %s: %d channels over %d heads", path, a.ModelChannels, a.NumHeads)
	}
	return a, nil
}

func (c FlowConfig) headDim() int { return c.ModelChannels / c.NumHeads }

// GridCoords returns the (x, y, z) of every cell of an n³ grid in row-major
// order — the token order of the dense sparse-structure latent.
func GridCoords(n int) [][3]int32 {
	out := make([][3]int32, 0, n*n*n)
	for x := range n {
		for y := range n {
			for z := range n {
				out = append(out, [3]int32{int32(x), int32(y), int32(z)})
			}
		}
	}
	return out
}

// ropeTables returns cos/sin [len(coords), headDim/2] of the reference's
// RotaryPositionEmbedder: each axis takes headDim/2/3 channel pairs at
// frequencies 10000^(-i/n), and the pairs left over rotate by zero.
// Angles are formed in float32 like the reference.
func ropeTables(coords [][3]int32, headDim int) (cos, sin []float32) {
	half := headDim / 2
	n := half / 3
	freqs := make([]float32, n)
	for i := range freqs {
		freqs[i] = 1 / float32(math.Pow(10000, float64(float32(i)/float32(n))))
	}
	cos, sin = make([]float32, len(coords)*half), make([]float32, len(coords)*half)
	for t, c := range coords {
		row := t * half
		for ax := range 3 {
			for i, f := range freqs {
				a := float64(float32(c[ax]) * f)
				cos[row+ax*n+i], sin[row+ax*n+i] = float32(math.Cos(a)), float32(math.Sin(a))
			}
		}
		for j := 3 * n; j < half; j++ {
			cos[row+j] = 1
		}
	}
	return cos, sin
}

// flow holds what a flow graph shares while building.
type flow struct {
	cfg FlowConfig
	w   weights
}

// crossKey and crossValue name block i's cross-attention keys and values:
// outputs of BuildFlowCond, inputs of BuildFlow.
func crossKey(i int) string   { return fmt.Sprintf("blocks.%d.cross_k", i) }
func crossValue(i int) string { return fmt.Sprintf("blocks.%d.cross_v", i) }

// BuildFlow builds one velocity evaluation of a flow transformer over the
// tokens at coords: inputs "x" [T, in_channels], "t" [1] (flow time in
// [0, 1]) and every block's cross-attention keys and values
// ("blocks.<i>.cross_k" / "cross_v", [1, condTokens, heads, head_dim] —
// BuildFlowCond's outputs, which depend on the condition alone and so are
// computed once per condition, not once per evaluation) — plus "proj" [T,
// ProjChannels] for a model with projected image features; output "v" [T,
// out_channels]. The sparse-structure model runs it over GridCoords; the
// structured-latent models over the occupied voxels, with any concatenated
// condition already appended to x's channels. blocks < cfg.NumBlocks
// builds a truncated model. f32 widens the checkpoint's bf16 weights to
// float32 constants — what the GPU executor needs to place the matrix
// products; the CPU reads bf16 in place at half the memory.
func BuildFlow(cfg FlowConfig, f *safetensors.File, coords [][3]int32, condTokens, blocks int, f32 bool) (g *graph.Graph, err error) {
	defer catch(&err)
	m := &flow{cfg: cfg, w: weights{f: f, widen: f32}}
	b := graph.NewBuilder("trellis2_flow")
	T, D, dh := len(coords), cfg.ModelChannels, cfg.headDim()
	x := b.Input("x", tensor.F32, T, cfg.InChannels)
	t := b.Input("t", tensor.F32, 1)
	var proj *graph.Value
	if pc := cfg.ProjChannels(); pc > 0 {
		proj = b.Input("proj", tensor.F32, T, pc)
	}

	wt, bias := m.w.linear("input_layer")
	h := b.Scope("input_layer").Linear(x, wt, bias)
	mod := m.timeMod(b.Scope("t_embedder"), t)
	c, s := ropeTables(coords, dh)
	cos := b.Const("rope_cos", tensor.FromF32(c, T, dh/2))
	sin := b.Const("rope_sin", tensor.FromF32(s, T, dh/2))
	for i := range blocks {
		k := b.Input(crossKey(i), tensor.F32, 1, condTokens, cfg.NumHeads, dh)
		v := b.Input(crossValue(i), tensor.F32, 1, condTokens, cfg.NumHeads, dh)
		h = m.block(b.Scope(fmt.Sprintf("blocks.%d", i)), m.w.scope(fmt.Sprintf("blocks.%d", i)), h, mod, k, v, proj, T, cos, sin)
	}
	h = b.LayerNorm(h, D, nil, nil, 1e-5)
	wt, bias = m.w.linear("out_layer")
	b.Output("v", b.Scope("out_layer").Linear(h, wt, bias))
	return b.Build()
}

// BuildFlowCond builds the condition half of a flow transformer: input
// "cond" [condTokens, cond_channels]; outputs, per block, the
// cross-attention keys (to_kv, RMS-normalised) and values BuildFlow takes.
// A sampling run evaluates it once per condition (the image features, the
// negative) rather than once per velocity evaluation. The weights stay in
// the checkpoint's dtype: the graph is small and runs on the CPU.
func BuildFlowCond(cfg FlowConfig, f *safetensors.File, condTokens, blocks int) (g *graph.Graph, err error) {
	defer catch(&err)
	m := &flow{cfg: cfg, w: weights{f: f}}
	b := graph.NewBuilder("trellis2_flow_cond")
	D, H, dh, L := int64(cfg.ModelChannels), int64(cfg.NumHeads), int64(cfg.headDim()), int64(condTokens)
	cond := b.Input("cond", tensor.F32, condTokens, cfg.CondChannels)
	for i := range blocks {
		ca := b.Scope(fmt.Sprintf("blocks.%d.cross_attn", i))
		wc := m.w.scope(fmt.Sprintf("blocks.%d.cross_attn", i))
		if cfg.ProjChannels() > 0 {
			wc = wc.scope("cross_attn_block") // see block
		}
		wt, bias := wc.linear("to_kv")
		kv := ca.Linear(cond, wt, bias)
		k := m.rmsNorm(ca, ca.Reshape(ca.Slice(kv, 1, 0, D), L, H, dh), wc.f32("k_rms_norm.gamma"))
		b.Output(crossKey(i), ca.Reshape(k, 1, L, H, dh))
		b.Output(crossValue(i), ca.Reshape(ca.Slice(kv, 1, D, 2*D), 1, L, H, dh))
	}
	return b.Build()
}

// timeMod is TimestepEmbedder followed by the shared adaLN_modulation:
// [cos, sin] of 1000·t·freqs (256 channels) → Linear → SiLU → Linear →
// SiLU → Linear(dim → 6·dim), [1, 6·dim].
func (m *flow) timeMod(b *graph.Builder, t *graph.Value) *graph.Value {
	const half = 128
	freqs := tensor.New(tensor.F32, 1, half)
	for i := range half {
		freqs.F32()[i] = float32(math.Exp(float64(float32(-math.Log(10000)) * float32(i) / half)))
	}
	args := b.Mul(b.Reshape(b.Mul(t, b.Scalar(1000)), 1, 1), b.Const("freqs", freqs))
	emb := b.Concat(1, b.Op("Cos", nil, args), b.Op("Sin", nil, args))
	wt, bias := m.w.linear("t_embedder.mlp.0")
	h := b.SiLU(b.Linear(emb, wt, bias))
	wt, bias = m.w.linear("t_embedder.mlp.2")
	h = b.Linear(h, wt, bias)
	wt, bias = m.w.linear("adaLN_modulation.1")
	return b.Linear(b.SiLU(h), wt, bias)
}

// sdpaAttrs: q, k, v and the output all in [1, T, H, dh] (the projections'
// natural layout — no transposes), softmax scale 1/√dh.
func (m *flow) sdpaAttrs() ops.Attrs {
	return graph.Attr("scale", float32(1/math.Sqrt(float64(m.cfg.headDim()))),
		"a_layout", 1, "b_layout", 1, "v_layout", 1, "stride_out", 1)
}

// rmsNorm is MultiHeadRMSNorm over x [T, H, dh]: unit-normalise each head's
// vector, then scale by gamma[H, dh]·√dh.
func (m *flow) rmsNorm(b *graph.Builder, x *graph.Value, gamma *tensor.Tensor) *graph.Value {
	norm := b.Op("ReduceL2", graph.Attr("keepdims", 1), x, b.Ints(-1))
	x = b.Div(x, b.Op("Max", nil, norm, b.Scalar(1e-12)))
	g := tensor.New(tensor.F32, gamma.Shape()...)
	scale := float32(math.Sqrt(float64(m.cfg.headDim())))
	for i, v := range gamma.F32() {
		g.F32()[i] = v * scale
	}
	return b.Mul(x, b.Const("gamma", g))
}

// block is ModulatedTransformerCrossBlock (share_mod) over x [T, dim]:
// modulated self-attention with rotary positions, cross-attention to the
// condition's keys and values ck, cv (see BuildFlowCond) — plus the
// projected features' linear, when the model has them — modulated MLP.
func (m *flow) block(b *graph.Builder, w weights, x, tmod, ck, cv, proj *graph.Value, T int, cos, sin *graph.Value) *graph.Value {
	D, H, dh := int64(m.cfg.ModelChannels), int64(m.cfg.NumHeads), int64(m.cfg.headDim())
	mod := b.Add(b.Const("modulation", w.f32("modulation").Reshape(1, int(6*D))), tmod)
	chunk := func(i int64) *graph.Value { return b.Slice(mod, 1, i*D, (i+1)*D) }
	shiftA, scaleA, gateA, shiftM, scaleM, gateM := chunk(0), chunk(1), chunk(2), chunk(3), chunk(4), chunk(5)
	one := b.Scalar(1)
	heads := func(sb *graph.Builder, v *graph.Value, n int) *graph.Value { return sb.Reshape(v, int64(n), H, dh) }
	batch := func(sb *graph.Builder, v *graph.Value, n int) *graph.Value { return sb.Reshape(v, 1, int64(n), H, dh) }

	sa, ws := b.Scope("self_attn"), w.scope("self_attn")
	y := b.Add(b.Mul(b.LayerNorm(x, int(D), nil, nil, 1e-6), b.Add(scaleA, one)), shiftA)
	wt, bias := ws.linear("to_qkv")
	qkv := sa.Linear(y, wt, bias)
	q := m.rmsNorm(sa, heads(sa, sa.Slice(qkv, 1, 0, D), T), ws.f32("q_rms_norm.gamma"))
	k := m.rmsNorm(sa, heads(sa, sa.Slice(qkv, 1, D, 2*D), T), ws.f32("k_rms_norm.gamma"))
	q = batch(sa, sa.Op("ingot.RoPE", graph.Attr("layout", 0), q, cos, sin), T)
	k = batch(sa, sa.Op("ingot.RoPE", graph.Attr("layout", 0), k, cos, sin), T)
	v := batch(sa, sa.Slice(qkv, 1, 2*D, 3*D), T)
	o := sa.Reshape(sa.Op("ingot.SDPA", m.sdpaAttrs(), q, k, v), int64(T), D)
	wt, bias = ws.linear("to_out")
	x = b.Add(x, b.Mul(sa.Linear(o, wt, bias), gateA))

	// With projected features the cross-attention is nested one level down
	// (ProjectAttention.cross_attn_block), next to its proj_linear.
	ca, wc := b.Scope("cross_attn"), w.scope("cross_attn")
	wp := wc
	if proj != nil {
		wc = wc.scope("cross_attn_block")
	}
	y = b.LayerNorm(x, int(D), w.f32("norm2.weight"), w.f32("norm2.bias"), 1e-6)
	wt, bias = wc.linear("to_q")
	q = batch(ca, m.rmsNorm(ca, heads(ca, ca.Linear(y, wt, bias), T), wc.f32("q_rms_norm.gamma")), T)
	o = ca.Reshape(ca.Op("ingot.SDPA", m.sdpaAttrs(), q, ck, cv), int64(T), D)
	wt, bias = wc.linear("to_out")
	o = ca.Linear(o, wt, bias)
	if proj != nil {
		wt, bias = wp.linear("proj_linear")
		o = ca.Add(ca.Scope("proj_linear").Linear(proj, wt, bias), o)
	}
	x = b.Add(x, o)

	mb, wm := b.Scope("mlp"), w.scope("mlp.mlp")
	y = b.Add(b.Mul(b.LayerNorm(x, int(D), nil, nil, 1e-6), b.Add(scaleM, one)), shiftM)
	wt, bias = wm.linear("0")
	y = mb.GeluTanh(mb.Linear(y, wt, bias))
	wt, bias = wm.linear("2")
	return b.Add(x, b.Mul(mb.Linear(y, wt, bias), gateM))
}
