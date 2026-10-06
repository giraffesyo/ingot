package wan

import (
	"fmt"
	"html"
	"math"
	"strings"
	"unicode"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
	"golang.org/x/text/unicode/norm"
)

// TextConfig is text_encoder/config.json (transformers UMT5EncoderModel).
type TextConfig struct {
	VocabSize       int     `json:"vocab_size"`
	DModel          int     `json:"d_model"`
	DKV             int     `json:"d_kv"`
	DFF             int     `json:"d_ff"`
	NumLayers       int     `json:"num_layers"`
	NumHeads        int     `json:"num_heads"`
	Buckets         int     `json:"relative_attention_num_buckets"`
	MaxDistance     int     `json:"relative_attention_max_distance"`
	FeedForwardProj string  `json:"feed_forward_proj"`
	DenseActFn      string  `json:"dense_act_fn"`
	Eps             float32 `json:"layer_norm_epsilon"`
	ModelType       string  `json:"model_type"`
}

// LoadTextConfig reads dir/config.json and rejects variants not built here.
func LoadTextConfig(dir string) (TextConfig, error) {
	var c TextConfig
	if err := readJSON(dir, "config.json", &c); err != nil {
		return c, err
	}
	if c.ModelType != "umt5" || c.FeedForwardProj != "gated-gelu" || c.DenseActFn != "gelu_new" {
		return c, fmt.Errorf("wan: text encoder variant not implemented (model_type=%q feed_forward_proj=%q act=%q)",
			c.ModelType, c.FeedForwardProj, c.DenseActFn)
	}
	return c, nil
}

// MaxTextTokens is the pipeline's max_sequence_length: prompts are cut to
// this many tokens and their embeddings zero-padded to it, and the
// transformer cross-attends to all of them.
const MaxTextTokens = 512

// TextEncoder is umT5's encoder stack: per-layer relative position bias,
// unscaled attention, gated-GELU feed-forward, RMS norms.
type TextEncoder struct {
	cfg   TextConfig
	set   *safetensors.Set
	embed *tensor.Tensor // [vocab, d_model], stored dtype
	widen bool
}

// NewTextEncoder opens the encoder. widen hands the matrix products f32
// weights (the GPU executor's requirement) instead of reading bf16 in place.
func NewTextEncoder(cfg TextConfig, set *safetensors.Set, widen bool) (*TextEncoder, error) {
	e, err := set.Tensor("shared.weight")
	if err != nil {
		return nil, err
	}
	if s := e.Shape(); len(s) != 2 || s[1] != cfg.DModel {
		return nil, fmt.Errorf("wan: shared.weight shape %v", s)
	}
	return &TextEncoder{cfg: cfg, set: set, embed: e, widen: widen}, nil
}

// Embed looks ids up on the host, widening rows to f32.
func (te *TextEncoder) Embed(ids []int64) (*tensor.Tensor, error) {
	D, vocab := te.cfg.DModel, te.embed.Shape()[0]
	out := tensor.New(tensor.F32, len(ids), D)
	dst := out.F32()
	for i, id := range ids {
		if id < 0 || int(id) >= vocab {
			return nil, fmt.Errorf("wan: token id %d outside vocab %d", id, vocab)
		}
		switch te.embed.DType() {
		case tensor.BF16:
			for j, v := range te.embed.BF16()[int(id)*D : (int(id)+1)*D] {
				dst[i*D+j] = math.Float32frombits(uint32(v) << 16)
			}
		case tensor.F32:
			copy(dst[i*D:(i+1)*D], te.embed.F32()[int(id)*D:(int(id)+1)*D])
		default:
			return nil, fmt.Errorf("wan: shared.weight dtype %s", te.embed.DType())
		}
	}
	return out, nil
}

// relativeBucket is T5's bidirectional bucket of key − query distance rel.
func relativeBucket(rel, buckets, maxDist int) int64 {
	buckets /= 2
	b := 0
	if rel > 0 {
		b = buckets
	}
	n := rel
	if n < 0 {
		n = -n
	}
	exact := buckets / 2
	if n < exact {
		return int64(b + n)
	}
	// float32 log ratio truncated toward zero, as the reference's .to(long).
	r := float32(math.Log(float64(float32(n)/float32(exact)))) / float32(math.Log(float64(maxDist)/float64(exact)))
	large := exact + int(r*float32(buckets-exact))
	return int64(b + min(large, buckets-1))
}

// Build returns the encoder graph for a T-token prompt: input "x" [T,
// d_model] (from Embed), output "hidden" [T, d_model] (last_hidden_state).
// Only real tokens are encoded: with the pipeline's padding mask the
// padded positions never reach them. layers < NumLayers truncates (tests).
func (te *TextEncoder) Build(T, layers int) (g *graph.Graph, err error) {
	defer catch(&err)
	c := te.cfg
	D, H, dk := c.DModel, c.NumHeads, c.DKV
	b := graph.NewBuilder("umt5_encoder")
	w := weights{set: te.set, prefix: "encoder.", widen: te.widen}
	x := b.Input("x", tensor.F32, T, D)

	buckets := make([]int64, T*T)
	for q := range T {
		for k := range T {
			buckets[q*T+k] = relativeBucket(k-q, c.Buckets, c.MaxDistance)
		}
	}
	bidx := b.Const("buckets", tensor.FromI64(buckets, T, T))
	ti, hi, dki := int64(T), int64(H), int64(dk)
	for i := range layers {
		bl := b.Scope(fmt.Sprintf("block.%d", i))
		wl := w.scope(fmt.Sprintf("block.%d.layer", i))
		sa := wl.scope("0.SelfAttention")
		y := bl.RMSNorm(x, wl.f32("0.layer_norm.weight"), c.Eps)
		q := bl.Linear(y, sa.matrix("q.weight"), nil)
		k := bl.Linear(y, sa.matrix("k.weight"), nil)
		v := bl.Linear(y, sa.matrix("v.weight"), nil)
		q = bl.Transpose(bl.Reshape(q, ti, hi, dki), 1, 0, 2) // [H, T, dk]
		k = bl.Transpose(bl.Reshape(k, ti, hi, dki), 1, 2, 0) // [H, dk, T]
		v = bl.Transpose(bl.Reshape(v, ti, hi, dki), 1, 0, 2)
		// Position bias [H, T, T]: the layer's bucket table gathered.
		table := bl.Const("rel_bias", sa.f32("relative_attention_bias.weight")) // [buckets, H]
		bias := bl.Transpose(bl.Op("Gather", graph.Attr("axis", 0), table, bidx), 2, 0, 1)
		s := bl.Add(bl.Op("MatMul", nil, q, k), bias)
		p := bl.Op("Softmax", graph.Attr("axis", -1), s)
		o := bl.Reshape(bl.Transpose(bl.Op("MatMul", nil, p, v), 1, 0, 2), ti, int64(H*dk))
		x = bl.Add(x, bl.Linear(o, sa.matrix("o.weight"), nil))

		ff := wl.scope("1.DenseReluDense")
		y = bl.RMSNorm(x, wl.f32("1.layer_norm.weight"), c.Eps)
		gate := bl.GeluTanh(bl.Linear(y, ff.matrix("wi_0.weight"), nil))
		y = bl.Mul(gate, bl.Linear(y, ff.matrix("wi_1.weight"), nil))
		x = bl.Add(x, bl.Linear(y, ff.matrix("wo.weight"), nil))
	}
	b.Output("hidden", b.RMSNorm(x, w.f32("final_layer_norm.weight"), c.Eps))
	return b.Build()
}

// CleanPrompt is the pipeline's prompt_clean: ftfy.fix_text (the parts that
// touch prompts: full-width forms to ASCII, curly quotes straightened,
// Latin ligatures split, control characters dropped, NFC), HTML entities
// unescaped twice, whitespace collapsed and trimmed. The default Chinese
// negative prompt depends on it: its full-width commas tokenise as ",".
func CleanPrompt(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 0xFF01 && r <= 0xFF5E: // full-width ASCII
			b.WriteRune(r - 0xFF01 + 0x21)
		case r == 0x3000: // ideographic space
			b.WriteByte(' ')
		case r == '‘' || r == '’' || r == '‚' || r == '‛' || r == '′':
			b.WriteByte('\'')
		case r == '“' || r == '”' || r == '„' || r == '‟' || r == '″':
			b.WriteByte('"')
		case ligatures[r] != "":
			b.WriteString(ligatures[r])
		case unicode.IsControl(r) && !unicode.IsSpace(r):
		default:
			b.WriteRune(r)
		}
	}
	s = norm.NFC.String(b.String())
	s = html.UnescapeString(html.UnescapeString(s))
	return strings.Join(strings.Fields(s), " ")
}

// ligatures is ftfy's fix_latin_ligatures table.
var ligatures = map[rune]string{
	'Ĳ': "IJ", 'ĳ': "ij", 'ŉ': "ʼn", 'Ǳ': "DZ", 'ǲ': "Dz", 'ǳ': "dz", 'Ǆ': "DŽ", 'ǅ': "Dž", 'ǆ': "dž",
	'Ǉ': "LJ", 'ǈ': "Lj", 'ǉ': "lj", 'Ǌ': "NJ", 'ǋ': "Nj", 'ǌ': "nj",
	'ﬀ': "ff", 'ﬁ': "fi", 'ﬂ': "fl", 'ﬃ': "ffi", 'ﬄ': "ffl", 'ﬅ': "ſt", 'ﬆ': "st",
}
