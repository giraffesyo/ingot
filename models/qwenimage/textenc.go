package qwenimage

import (
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// TextConfig is text_encoder/config.json's text_config (Qwen3-VL language
// model).
type TextConfig struct {
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float64 `json:"rope_theta"`
	VocabSize         int     `json:"vocab_size"`
	HiddenAct         string  `json:"hidden_act"`
	AttentionBias     bool    `json:"attention_bias"`
}

// LoadTextConfig reads dir/config.json.
func LoadTextConfig(dir string) (TextConfig, error) {
	var c struct {
		TextConfig TextConfig `json:"text_config"`
	}
	if err := readConfig(dir, &c); err != nil {
		return c.TextConfig, err
	}
	t := c.TextConfig
	if t.HiddenAct != "silu" || t.AttentionBias || t.NumAttentionHeads%t.NumKeyValueHeads != 0 {
		return t, fmt.Errorf("qwenimage: text encoder variant not implemented: %+v", t)
	}
	return t, nil
}

// TextEncoder runs Qwen3-VL's language model over a text-only prompt and
// returns the hidden states Qwen-Image conditions on: the last decoder
// layer's output *before* the final RMSNorm (the pipeline hooks the norm
// away), with the leading system-prompt rows dropped.
type TextEncoder struct {
	cfg   TextConfig
	set   *safetensors.Set
	embed *tensor.Tensor // [vocab, hidden], stored dtype
}

const textPrefix = "model.language_model."

// NewTextEncoder opens the checkpoint's language model.
func NewTextEncoder(cfg TextConfig, set *safetensors.Set) (*TextEncoder, error) {
	e, err := set.Tensor(textPrefix + "embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	if s := e.Shape(); len(s) != 2 || s[1] != cfg.HiddenSize {
		return nil, fmt.Errorf("qwenimage: embed_tokens shape %v", s)
	}
	return &TextEncoder{cfg: cfg, set: set, embed: e}, nil
}

// Embed looks ids up in the embedding table on the host (the table stays a
// view of the mapped checkpoint), widening bf16 rows to f32.
func (te *TextEncoder) Embed(ids []int64) (*tensor.Tensor, error) {
	D := te.cfg.HiddenSize
	vocab := te.embed.Shape()[0]
	out := tensor.New(tensor.F32, len(ids), D)
	dst := out.F32()
	for i, id := range ids {
		if id < 0 || int(id) >= vocab {
			return nil, fmt.Errorf("qwenimage: token id %d outside vocab %d", id, vocab)
		}
		switch te.embed.DType() {
		case tensor.BF16:
			row := te.embed.BF16()[int(id)*D : (int(id)+1)*D]
			for j, v := range row {
				dst[i*D+j] = math.Float32frombits(uint32(v) << 16)
			}
		case tensor.F32:
			copy(dst[i*D:(i+1)*D], te.embed.F32()[int(id)*D:(int(id)+1)*D])
		default:
			return nil, fmt.Errorf("qwenimage: embed_tokens dtype %s", te.embed.DType())
		}
	}
	return out, nil
}

// Build returns the encoder graph for a T-token prompt: input "x" [T,
// hidden] (from Embed), output "hidden" [T-drop, hidden]. layers <
// NumHiddenLayers builds a truncated model (tests).
func (te *TextEncoder) Build(T, drop, layers int) (g *graph.Graph, err error) {
	defer catch(&err)
	c := te.cfg
	D, H, KV, dh := c.HiddenSize, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim
	b := graph.NewBuilder("qwen3vl_text")
	w := weights{set: te.set, prefix: textPrefix}
	x := b.Input("x", tensor.F32, T, D)

	// Causal mask and RoPE tables (text-only: every M-RoPE axis carries the
	// same position, which reduces to standard RoPE over cat(freqs, freqs)).
	mask := tensor.New(tensor.F32, T, T)
	for q := range T {
		for k := q + 1; k < T; k++ {
			mask.F32()[q*T+k] = float32(math.Inf(-1))
		}
	}
	maskV := b.Const("causal", mask)
	cos, sin := tensor.New(tensor.F32, T, 1, dh), tensor.New(tensor.F32, T, 1, dh)
	for i := 0; i < dh/2; i++ {
		inv := 1 / float32(math.Pow(c.RopeTheta, float64(float32(2*i)/float32(dh))))
		for p := range T {
			a := float64(float32(p) * inv)
			cv, sv := float32(math.Cos(a)), float32(math.Sin(a))
			cos.F32()[p*dh+i], cos.F32()[p*dh+i+dh/2] = cv, cv
			sin.F32()[p*dh+i], sin.F32()[p*dh+i+dh/2] = sv, sv
		}
	}
	cosV, sinV := b.Const("rope_cos", cos), b.Const("rope_sin", sin)
	rope := func(b *graph.Builder, x *graph.Value) *graph.Value { // x [T, heads, dh]
		x1 := b.Slice(x, 2, 0, int64(dh/2))
		x2 := b.Slice(x, 2, int64(dh/2), int64(dh))
		rot := b.Concat(2, b.Op("Neg", nil, x2), x1)
		return b.Add(b.Mul(x, cosV), b.Mul(rot, sinV))
	}
	scale := float32(1 / math.Sqrt(float64(dh)))
	rep := b.Ints(1, int64(T), int64(KV), int64(H/KV), int64(dh))

	for i := range layers {
		bi, wi := b.Scope(fmt.Sprintf("layers.%d", i)), w.scope(fmt.Sprintf("layers.%d", i))
		h := bi.RMSNorm(x, wi.f32("input_layernorm.weight"), c.RMSNormEps)
		q := bi.Linear(h, wi.raw("self_attn.q_proj.weight"), nil)
		k := bi.Linear(h, wi.raw("self_attn.k_proj.weight"), nil)
		v := bi.Linear(h, wi.raw("self_attn.v_proj.weight"), nil)
		q = rope(bi, bi.RMSNorm(bi.Reshape(q, int64(T), int64(H), int64(dh)), wi.f32("self_attn.q_norm.weight"), c.RMSNormEps))
		k = rope(bi, bi.RMSNorm(bi.Reshape(k, int64(T), int64(KV), int64(dh)), wi.f32("self_attn.k_norm.weight"), c.RMSNormEps))
		// repeat_kv: kv head j serves query heads j·(H/KV) … (j+1)·(H/KV)−1.
		expand := func(t *graph.Value) *graph.Value {
			t = bi.Op("Expand", nil, bi.Reshape(t, 1, int64(T), int64(KV), 1, int64(dh)), rep)
			return bi.Reshape(t, 1, int64(T), int64(H), int64(dh))
		}
		o := bi.Op("ingot.SDPA", graph.Attr("scale", scale, "a_layout", 1, "b_layout", 1, "v_layout", 1, "stride_out", 1),
			bi.Reshape(q, 1, int64(T), int64(H), int64(dh)), expand(k), expand(bi.Reshape(v, int64(T), int64(KV), int64(dh))), maskV)
		x = bi.Add(x, bi.Linear(bi.Reshape(o, int64(T), int64(H*dh)), wi.raw("self_attn.o_proj.weight"), nil))

		h = bi.RMSNorm(x, wi.f32("post_attention_layernorm.weight"), c.RMSNormEps)
		gate := bi.SiLU(bi.Linear(h, wi.raw("mlp.gate_proj.weight"), nil))
		up := bi.Linear(h, wi.raw("mlp.up_proj.weight"), nil)
		x = bi.Add(x, bi.Linear(bi.Mul(gate, up), wi.raw("mlp.down_proj.weight"), nil))
	}
	b.Output("hidden", b.Slice(x, 0, int64(drop), int64(T)))
	return b.Build()
}
