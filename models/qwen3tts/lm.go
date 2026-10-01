package qwen3tts

import (
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/tensor"
)

// buildLM returns a Qwen3 decoder stack for cached autoregressive runs
// (graph.CompileDecode + RunDecode): inputs "x" [T, in] (T dynamic: the
// prompt at prefill, 1 per step), "cos"/"sin" [T, head_dim/2] for this
// run's positions; outputs "hidden" — the final-normed hidden state of the
// last position [1, hidden] — and, with head, "logits" [1, vocab]. proj
// (may be nil) is a Linear applied to x first (small_to_mtp_projection).
// layers < c.NumHiddenLayers truncates (tests). q = Int8 marks every Linear
// for int8 decode.
func buildLM(name string, c LMConfig, w weights, in int, proj, projBias, head *tensor.Tensor, layers int, q Quant) (g *graph.Graph, err error) {
	defer catch(&err)
	H, KV, dh := c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim
	b := graph.NewBuilder(name)
	b.QuantizeLinear = string(q)
	x := b.Input("x", tensor.F32, -1, in)
	cosV := b.Input("cos", tensor.F32, -1, dh/2)
	sinV := b.Input("sin", tensor.F32, -1, dh/2)
	if proj != nil {
		x = b.Linear(x, proj, projBias)
	}
	scale := float32(1 / math.Sqrt(float64(dh)))
	for i := range layers {
		bi, wi := b.Scope(fmt.Sprintf("layers.%d", i)), w.scope(fmt.Sprintf("layers.%d", i))
		h := bi.RMSNorm(x, wi.f32("input_layernorm.weight"), c.RMSNormEps)
		q := bi.Linear(h, wi.raw("self_attn.q_proj.weight"), nil)
		k := bi.Linear(h, wi.raw("self_attn.k_proj.weight"), nil)
		v := bi.Linear(h, wi.raw("self_attn.v_proj.weight"), nil)
		// q/k norms are per head (over head_dim), then rotate_half RoPE.
		q = bi.RMSNorm(bi.Reshape(q, -1, int64(H), int64(dh)), wi.f32("self_attn.q_norm.weight"), c.RMSNormEps)
		k = bi.RMSNorm(bi.Reshape(k, -1, int64(KV), int64(dh)), wi.f32("self_attn.k_norm.weight"), c.RMSNormEps)
		q = bi.Op("ingot.RoPE", graph.Attr("layout", 1), q, cosV, sinV)
		k = bi.Op("ingot.RoPE", graph.Attr("layout", 1), k, cosV, sinV)
		// Cached SDPA: K/V (KV heads, grouped-query) append to the decode
		// slot; every operand and the output stay [1, T, heads, dh].
		o := bi.Op("ingot.SDPA", graph.Attr("scale", scale, "cache", 1,
			"a_layout", 1, "k_layout", 1, "v_layout", 1, "stride_out", 1),
			bi.Reshape(q, 1, -1, int64(H), int64(dh)),
			bi.Reshape(k, 1, -1, int64(KV), int64(dh)),
			bi.Reshape(v, 1, -1, int64(KV), int64(dh)))
		x = bi.Add(x, bi.Linear(bi.Reshape(o, -1, int64(H*dh)), wi.raw("self_attn.o_proj.weight"), nil))

		h = bi.RMSNorm(x, wi.f32("post_attention_layernorm.weight"), c.RMSNormEps)
		gate := bi.SiLU(bi.Linear(h, wi.raw("mlp.gate_proj.weight"), nil))
		up := bi.Linear(h, wi.raw("mlp.up_proj.weight"), nil)
		x = bi.Add(x, bi.Linear(bi.Mul(gate, up), wi.raw("mlp.down_proj.weight"), nil))
	}
	last := b.Slice(x, 0, -1, math.MaxInt64)
	hid := b.RMSNorm(last, w.f32("norm.weight"), c.RMSNormEps)
	b.Output("hidden", hid)
	if head != nil {
		b.Output("logits", b.Linear(hid, head, nil))
	}
	return b.Build()
}

// buildTextProjection is talker.text_projection (ResizeMLP): x [T, text
// hidden] → fc2(silu(fc1(x))) [T, hidden], both Linears with bias.
func buildTextProjection(w weights) (g *graph.Graph, err error) {
	defer catch(&err)
	b := graph.NewBuilder("text_projection")
	w1 := w.raw("linear_fc1.weight")
	x := b.Input("x", tensor.F32, -1, w1.Shape()[1])
	h := b.SiLU(b.Linear(x, w1, w.f32("linear_fc1.bias")))
	b.Output("y", b.Linear(h, w.raw("linear_fc2.weight"), w.f32("linear_fc2.bias")))
	return b.Build()
}

// rope holds the rotate_half RoPE tables for positions [0, maxT): cos/sin
// [maxT, dh/2], rows sliced per run.
type rope struct {
	cos, sin []float32
	half     int
}

// newRope computes HF's default RoPE: inv_freq_i = theta^-(2i/dh) (f32
// exponent as in the reference), angle = pos·inv_freq in f32.
func newRope(theta float64, dh, maxT int) *rope {
	r := &rope{cos: make([]float32, maxT*dh/2), sin: make([]float32, maxT*dh/2), half: dh / 2}
	for i := range dh / 2 {
		inv := 1 / float32(math.Pow(theta, float64(float32(2*i)/float32(dh))))
		for p := range maxT {
			a := float64(float32(p) * inv)
			r.cos[p*dh/2+i], r.sin[p*dh/2+i] = float32(math.Cos(a)), float32(math.Sin(a))
		}
	}
	return r
}

// at returns the tables for positions [p0, p0+n).
func (r *rope) at(p0, n int) (cos, sin *tensor.Tensor) {
	lo, hi := p0*r.half, (p0+n)*r.half
	return tensor.FromF32(r.cos[lo:hi], n, r.half), tensor.FromF32(r.sin[lo:hi], n, r.half)
}
