package trellis2

import (
	"fmt"
	"math"
	"path/filepath"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// DinoConfig is the DINOv3 ViT's config.json (transformers DINOv3ViTModel).
type DinoConfig struct {
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumRegisterTokens int     `json:"num_register_tokens"`
	PatchSize         int     `json:"patch_size"`
	NumChannels       int     `json:"num_channels"`
	LayerNormEps      float32 `json:"layer_norm_eps"`
	RopeTheta         float64 `json:"rope_theta"`
	HiddenAct         string  `json:"hidden_act"`
	UseGatedMLP       bool    `json:"use_gated_mlp"`
	KeyBias           bool    `json:"key_bias"`
}

// LoadDinoConfig reads dir/config.json.
func LoadDinoConfig(dir string) (DinoConfig, error) {
	var c DinoConfig
	if err := readJSON(filepath.Join(dir, "config.json"), &c); err != nil {
		return c, err
	}
	if c.UseGatedMLP || c.HiddenAct != "gelu" || c.KeyBias || c.HiddenSize%c.NumAttentionHeads != 0 ||
		(c.HiddenSize/c.NumAttentionHeads)%4 != 0 {
		return c, fmt.Errorf("trellis2: DINOv3 variant not implemented (gated_mlp=%v act=%q key_bias=%v heads=%d)",
			c.UseGatedMLP, c.HiddenAct, c.KeyBias, c.NumAttentionHeads)
	}
	return c, nil
}

// Tokens is the sequence length for a size×size image: the class token,
// the register tokens, then the patches.
func (c DinoConfig) Tokens(size int) int {
	n := size / c.PatchSize
	return 1 + c.NumRegisterTokens + n*n
}

// ImageMean and ImageStd normalise RGB in [0, 1] for the encoder.
var (
	ImageMean = [3]float32{0.485, 0.456, 0.406}
	ImageStd  = [3]float32{0.229, 0.224, 0.225}
)

// dinoRope returns cos/sin [tokens, headDim/2] of DINOv3ViTRopePositionEmbedding
// for an n×n patch grid: patch centres in [-1, 1], headDim/4 frequencies
// per axis, angle 2π·coord·theta^(-4i/headDim). The class and register
// tokens rotate by zero (the reference skips them).
func dinoRope(c DinoConfig, n int) (cos, sin []float32) {
	dh := c.HiddenSize / c.NumAttentionHeads
	half, quarter := dh/2, dh/4
	inv := make([]float32, quarter)
	for i := range inv {
		inv[i] = 1 / float32(math.Pow(c.RopeTheta, float64(float32(i)*(4/float32(dh)))))
	}
	prefix := 1 + c.NumRegisterTokens
	T := prefix + n*n
	cos, sin = make([]float32, T*half), make([]float32, T*half)
	for t := range prefix {
		for j := range half {
			cos[t*half+j] = 1
		}
	}
	coord := func(i int) float32 { return 2*((float32(i)+0.5)/float32(n)) - 1 }
	for y := range n {
		for x := range n {
			row := (prefix + y*n + x) * half
			for ax, p := range [2]float32{coord(y), coord(x)} {
				for i, f := range inv {
					a := float64(float32(2*math.Pi) * p * f)
					cos[row+ax*quarter+i], sin[row+ax*quarter+i] = float32(math.Cos(a)), float32(math.Sin(a))
				}
			}
		}
	}
	return cos, sin
}

// BuildDino builds the conditioning encoder as the pipeline's
// DinoV3FeatureExtractor runs it: input "image" [1, 3, size, size]
// (normalised RGB); output "features" [Tokens(size), hidden] — every
// layer's output through a final unparameterised layer norm, class and
// register tokens included.
func BuildDino(cfg DinoConfig, f *safetensors.File, size int) (g *graph.Graph, err error) {
	defer catch(&err)
	if size%cfg.PatchSize != 0 {
		return nil, fmt.Errorf("trellis2: image size %d is not a multiple of the patch size %d", size, cfg.PatchSize)
	}
	w := weights{f: f}
	b := graph.NewBuilder("trellis2_dino")
	D, H := cfg.HiddenSize, int64(cfg.NumAttentionHeads)
	dh := int64(D) / H
	n := size / cfg.PatchSize
	T := cfg.Tokens(size)

	img := b.Input("image", tensor.F32, 1, cfg.NumChannels, size, size)
	we := w.scope("embeddings")
	p := b.Scope("embeddings").Conv2d(img, we.f32("patch_embeddings.weight"), we.f32("patch_embeddings.bias"), cfg.PatchSize, 0)
	p = b.Reshape(b.Transpose(b.Reshape(p, 1, int64(D), int64(n*n)), 0, 2, 1), int64(n*n), int64(D))
	x := b.Concat(0,
		b.Const("cls_token", we.f32("cls_token").Reshape(1, D)),
		b.Const("register_tokens", we.f32("register_tokens").Reshape(cfg.NumRegisterTokens, D)), p)

	c, s := dinoRope(cfg, n)
	cos := b.Const("rope_cos", tensor.FromF32(c, T, int(dh)/2))
	sin := b.Const("rope_sin", tensor.FromF32(s, T, int(dh)/2))
	sdpa := graph.Attr("scale", float32(1/math.Sqrt(float64(dh))), "a_layout", 1, "b_layout", 1, "v_layout", 1, "stride_out", 1)
	for i := range cfg.NumHiddenLayers {
		lb, wl := b.Scope(fmt.Sprintf("layer.%d", i)), w.scope(fmt.Sprintf("layer.%d", i))
		ab, wa := lb.Scope("attention"), wl.scope("attention")
		y := lb.LayerNorm(x, D, wl.f32("norm1.weight"), wl.f32("norm1.bias"), cfg.LayerNormEps)
		rot := func(v *graph.Value) *graph.Value {
			v = ab.Op("ingot.RoPE", graph.Attr("layout", 1), ab.Reshape(v, int64(T), H, dh), cos, sin)
			return ab.Reshape(v, 1, int64(T), H, dh)
		}
		q := rot(ab.Linear(y, wa.raw("q_proj.weight"), wa.f32("q_proj.bias")))
		k := rot(ab.Linear(y, wa.raw("k_proj.weight"), nil))
		v := ab.Reshape(ab.Linear(y, wa.raw("v_proj.weight"), wa.f32("v_proj.bias")), 1, int64(T), H, dh)
		o := ab.Reshape(ab.Op("ingot.SDPA", sdpa, q, k, v), int64(T), int64(D))
		o = ab.Linear(o, wa.raw("o_proj.weight"), wa.f32("o_proj.bias"))
		x = lb.Add(x, lb.Mul(o, lb.Const("layer_scale1", wl.f32("layer_scale1.lambda1"))))

		mb, wm := lb.Scope("mlp"), wl.scope("mlp")
		y = lb.LayerNorm(x, D, wl.f32("norm2.weight"), wl.f32("norm2.bias"), cfg.LayerNormEps)
		y = mb.Op("Gelu", nil, mb.Linear(y, wm.raw("up_proj.weight"), wm.f32("up_proj.bias")))
		y = mb.Linear(y, wm.raw("down_proj.weight"), wm.f32("down_proj.bias"))
		x = lb.Add(x, lb.Mul(y, lb.Const("layer_scale2", wl.f32("layer_scale2.lambda1"))))
	}
	b.Output("features", b.LayerNorm(x, D, nil, nil, 1e-5))
	return b.Build()
}
