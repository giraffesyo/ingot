package qwenimage

import (
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// VisionConfig is text_encoder/config.json's vision_config (the Qwen3-VL
// vision tower).
type VisionConfig struct {
	Depth                  int    `json:"depth"`
	HiddenSize             int    `json:"hidden_size"`
	NumHeads               int    `json:"num_heads"`
	IntermediateSize       int    `json:"intermediate_size"`
	HiddenAct              string `json:"hidden_act"`
	PatchSize              int    `json:"patch_size"`
	TemporalPatchSize      int    `json:"temporal_patch_size"`
	SpatialMergeSize       int    `json:"spatial_merge_size"`
	OutHiddenSize          int    `json:"out_hidden_size"`
	NumPositionEmbeddings  int    `json:"num_position_embeddings"`
	DeepstackVisualIndexes []int  `json:"deepstack_visual_indexes"`
	InChannels             int    `json:"in_channels"`
}

// visionRopeTheta: the vision tower's axial RoPE base (the config's
// default; not stored in config.json).
const visionRopeTheta = 10000

// LoadVisionConfig reads dir/config.json's vision_config.
func LoadVisionConfig(dir string) (VisionConfig, error) {
	var c struct {
		V VisionConfig `json:"vision_config"`
	}
	if err := readConfig(dir, &c); err != nil {
		return c.V, err
	}
	v := c.V
	if v.PatchSize != visPatch || v.SpatialMergeSize != visMerge || v.TemporalPatchSize != visTemporal ||
		v.InChannels != 3 || v.HiddenAct != "gelu_pytorch_tanh" {
		return v, fmt.Errorf("qwenimage: vision tower variant not implemented: %+v", v)
	}
	return v, nil
}

const visionPrefix = "model.visual."

// BuildVision builds the vision tower for one image of gh×gw patches: input
// "pixels" [gh·gw, 1536] (VisionPatches); outputs "merged" [gh·gw/4,
// out_hidden] (the image tokens the language model reads in place of its
// image placeholders) and "deep<i>" (the deepstack features added to the
// first language-model layers' hidden states at those positions).
func BuildVision(cfg VisionConfig, set *safetensors.Set, gh, gw int) (g *graph.Graph, err error) {
	defer catch(&err)
	w := weights{set: set, prefix: visionPrefix}
	b := graph.NewBuilder("qwen3vl_vision")
	N, D, H := gh*gw, cfg.HiddenSize, cfg.NumHeads
	dh := D / H
	pw := w.raw("patch_embed.proj.weight")
	x := b.Input("pixels", tensor.F32, N, visPatchDim)
	x = b.Scope("patch_embed").Linear(x, pw.Reshape(D, visPatchDim), w.f32("patch_embed.proj.bias"))
	x = b.Add(x, b.Const("pos_embed", visionPosEmbed(cfg, w.f32("pos_embed.weight"), gh, gw)))

	cos, sin := visionRope(cfg, gh, gw)
	cosV, sinV := b.Const("rope_cos", cos), b.Const("rope_sin", sin)
	rope := func(b *graph.Builder, t *graph.Value) *graph.Value { // t [N, H, dh]
		x1 := b.Slice(t, 2, 0, int64(dh/2))
		x2 := b.Slice(t, 2, int64(dh/2), int64(dh))
		return b.Add(b.Mul(t, cosV), b.Mul(b.Concat(2, b.Op("Neg", nil, x2), x1), sinV))
	}
	deep := map[int]int{}
	for i, l := range cfg.DeepstackVisualIndexes {
		deep[l] = i
	}
	scale := float32(1 / math.Sqrt(float64(dh)))
	for i := range cfg.Depth {
		bi, wi := b.Scope(fmt.Sprintf("blocks.%d", i)), w.scope(fmt.Sprintf("blocks.%d", i))
		y := bi.LayerNorm(x, D, wi.f32("norm1.weight"), wi.f32("norm1.bias"), 1e-6)
		qkv := bi.Linear(y, wi.raw("attn.qkv.weight"), wi.f32("attn.qkv.bias")) // [N, 3·D] = [q | k | v]
		qkv = bi.Reshape(qkv, int64(N), 3, int64(H), int64(dh))
		q := rope(bi, bi.Reshape(bi.Slice(qkv, 1, 0, 1), int64(N), int64(H), int64(dh)))
		k := rope(bi, bi.Reshape(bi.Slice(qkv, 1, 1, 2), int64(N), int64(H), int64(dh)))
		v := bi.Slice(qkv, 1, 2, 3)
		shape := func(t *graph.Value) *graph.Value { return bi.Reshape(t, 1, int64(N), int64(H), int64(dh)) }
		o := bi.Op("ingot.SDPA", graph.Attr("scale", scale, "a_layout", 1, "b_layout", 1, "v_layout", 1, "stride_out", 1),
			shape(q), shape(k), shape(v))
		o = bi.Linear(bi.Reshape(o, int64(N), int64(D)), wi.raw("attn.proj.weight"), wi.f32("attn.proj.bias"))
		x = bi.Add(x, o)
		y = bi.LayerNorm(x, D, wi.f32("norm2.weight"), wi.f32("norm2.bias"), 1e-6)
		y = bi.GeluTanh(bi.Linear(y, wi.raw("mlp.linear_fc1.weight"), wi.f32("mlp.linear_fc1.bias")))
		x = bi.Add(x, bi.Linear(y, wi.raw("mlp.linear_fc2.weight"), wi.f32("mlp.linear_fc2.bias")))
		if j, ok := deep[i]; ok {
			b.Output(fmt.Sprintf("deep%d", j), merger(b.Scope(fmt.Sprintf("deepstack%d", j)),
				w.scope(fmt.Sprintf("deepstack_merger_list.%d", j)), x, N, D, true))
		}
	}
	b.Output("merged", merger(b.Scope("merger"), w.scope("merger"), x, N, D, false))
	return b.Build()
}

// merger is Qwen3VLVisionPatchMerger: each 2×2 block of patches (4
// consecutive rows) becomes one 4·D vector → LayerNorm (before the shuffle,
// over D, or after it, over 4·D) → Linear → GELU (exact) → Linear.
func merger(b *graph.Builder, w weights, x *graph.Value, n, d int, postShuffle bool) *graph.Value {
	m := visMerge * visMerge
	if !postShuffle {
		x = b.LayerNorm(x, d, w.f32("norm.weight"), w.f32("norm.bias"), 1e-6)
	}
	x = b.Reshape(x, int64(n/m), int64(m*d))
	if postShuffle {
		x = b.LayerNorm(x, m*d, w.f32("norm.weight"), w.f32("norm.bias"), 1e-6)
	}
	x = b.Op("Gelu", nil, b.Linear(x, w.raw("linear_fc1.weight"), w.f32("linear_fc1.bias")))
	return b.Linear(x, w.raw("linear_fc2.weight"), w.f32("linear_fc2.bias"))
}

// visionPatchRC returns patch p's (row, col) in the grid: patches are laid
// out in 2×2 merge-block order.
func visionPatchRC(p, gw int) (int, int) {
	blocksW := gw / visMerge
	inCol, inRow := p%visMerge, (p/visMerge)%visMerge
	blockCol := (p / (visMerge * visMerge)) % blocksW
	blockRow := p / (visMerge * visMerge * blocksW)
	return blockRow*visMerge + inRow, blockCol*visMerge + inCol
}

// visionPosEmbed interpolates the learned side×side position table onto the
// gh×gw grid (bilinear, align_corners=True, border clamp) in patch order:
// get_vision_interpolation_indices_and_weights + the weighted table sum.
func visionPosEmbed(cfg VisionConfig, table *tensor.Tensor, gh, gw int) *tensor.Tensor {
	side := int(math.Sqrt(float64(cfg.NumPositionEmbeddings)))
	D := cfg.HiddenSize
	taps := func(i, size int) ([2]int, [2]float32) {
		src := float32(i) * float32(side-1) / float32(max(size-1, 1))
		fl := float32(math.Floor(float64(src)))
		var t [2]int
		var w [2]float32
		for o := range 2 {
			t[o] = min(max(int(fl)+o, 0), side-1)
			w[o] = max(1-float32(math.Abs(float64(src-fl-float32(o)))), 0)
		}
		return t, w
	}
	out := tensor.New(tensor.F32, gh*gw, D)
	tf, of := table.F32(), out.F32()
	for p := range gh * gw {
		r, c := visionPatchRC(p, gw)
		ht, hw := taps(r, gh)
		wt, ww := taps(c, gw)
		dst := of[p*D : (p+1)*D]
		for a := range 2 {
			for bb := range 2 {
				wgt := hw[a] * ww[bb]
				row := tf[(ht[a]*side+wt[bb])*D : (ht[a]*side+wt[bb]+1)*D]
				for j, v := range row {
					dst[j] += wgt * v
				}
			}
		}
	}
	return out
}

// visionRope returns the axial rotary tables [N, 1, dh] (rotate_half
// layout): frequencies over dh/2 channels, the first half driven by the
// patch row and the second by its column, the whole set repeated twice.
func visionRope(cfg VisionConfig, gh, gw int) (*tensor.Tensor, *tensor.Tensor) {
	dh := cfg.HiddenSize / cfg.NumHeads
	spatial := dh / 2
	inv := make([]float32, spatial/2)
	for i := range inv {
		inv[i] = 1 / float32(math.Pow(visionRopeTheta, float64(float32(2*i)/float32(spatial))))
	}
	N := gh * gw
	cos, sin := tensor.New(tensor.F32, N, 1, dh), tensor.New(tensor.F32, N, 1, dh)
	for p := range N {
		r, c := visionPatchRC(p, gw)
		for i, f := range inv {
			for k, pos := range [2]int{r, c} {
				a := float64(float32(pos) * f)
				cv, sv := float32(math.Cos(a)), float32(math.Sin(a))
				j := k*len(inv) + i
				cos.F32()[p*dh+j], cos.F32()[p*dh+j+spatial] = cv, cv
				sin.F32()[p*dh+j], sin.F32()[p*dh+j+spatial] = sv, sv
			}
		}
	}
	return cos, sin
}
