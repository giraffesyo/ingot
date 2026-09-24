package qwenimage

import (
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// VAEConfig is vae/config.json (AutoencoderKLQwenImage21).
type VAEConfig struct {
	ZDim               int       `json:"z_dim"`
	BaseDim            int       `json:"base_dim"`
	DecoderBaseDim     int       `json:"decoder_base_dim"`
	DimMult            []int     `json:"dim_mult"`
	NumResBlocks       int       `json:"num_res_blocks"`
	AttnScales         []float64 `json:"attn_scales"`
	TemperalDownsample []bool    `json:"temperal_downsample"`
	IsResidual         bool      `json:"is_residual"`
	OutChannels        int       `json:"out_channels"`
	PatchSize          *int      `json:"patch_size"`
	ScaleFactorSpatial int       `json:"scale_factor_spatial"`
	LatentsMean        []float32 `json:"latents_mean"`
	LatentsStd         []float32 `json:"latents_std"`
}

// LoadVAEConfig reads dir/config.json.
func LoadVAEConfig(dir string) (VAEConfig, error) {
	var c VAEConfig
	if err := readConfig(dir, &c); err != nil {
		return c, err
	}
	if c.DecoderBaseDim == 0 {
		c.DecoderBaseDim = c.BaseDim
	}
	return c, nil
}

// BuildVAEDecoder builds the image decoder for an h×w latent: input "z"
// [1, z_dim, h, w] (denormalised latents, as the pipeline passes to
// vae.decode), output "image" [1, out_channels, 16h, 16w] clamped to
// [-1, 1]. The reference's 5-D (b, c, t, h, w) tensors carry a single frame
// here, so the graph is plain NCHW; the temporal paths reduce to what the
// first chunk takes (time_conv skipped, last shortcut frame kept).
func BuildVAEDecoder(cfg VAEConfig, set *safetensors.Set, h, w int) (g *graph.Graph, err error) {
	if !cfg.IsResidual || cfg.PatchSize != nil || len(cfg.AttnScales) != 0 {
		return nil, fmt.Errorf("qwenimage: VAE config variant not implemented (is_residual=%v patch_size=%v attn_scales=%v)",
			cfg.IsResidual, cfg.PatchSize, cfg.AttnScales)
	}
	defer catch(&err)
	b := graph.NewBuilder("qwenimage21_vae_decoder")
	wt := weights{set: set}
	x := b.Input("z", tensor.F32, 1, cfg.ZDim, h, w)
	x = b.Scope("post_quant_conv").Conv2d(x, wt.f32("post_quant_conv.weight"), wt.f32("post_quant_conv.bias"), 1, 0)

	d := wt.scope("decoder")
	mult := cfg.DimMult
	dims := []int{cfg.DecoderBaseDim * mult[len(mult)-1]}
	for i := len(mult) - 1; i >= 0; i-- {
		dims = append(dims, cfg.DecoderBaseDim*mult[i])
	}
	up := make([]bool, len(cfg.TemperalDownsample)) // temperal_upsample = reversed
	for i, v := range cfg.TemperalDownsample {
		up[len(up)-1-i] = v
	}

	x = conv(b.Scope("conv_in"), d.scope("conv_in"), x, 1)
	mid := d.scope("mid_block")
	x = resBlock(b.Scope("mid.res0"), mid.scope("resnets.0"), x, dims[0], dims[0])
	x = attnBlock(b.Scope("mid.attn"), mid.scope("attentions.0"), x, dims[0], h, w)
	x = resBlock(b.Scope("mid.res1"), mid.scope("resnets.1"), x, dims[0], dims[0])

	ch, hh, ww := dims[0], h, w
	for i := range len(dims) - 1 {
		in, out := dims[i], dims[i+1]
		upFlag := i != len(mult)-1
		bi, wi := b.Scope(fmt.Sprintf("up%d", i)), d.scope(fmt.Sprintf("up_blocks.%d", i))
		xCopy := x
		cur := in
		for j := range cfg.NumResBlocks + 1 {
			x = resBlock(bi.Scope(fmt.Sprintf("res%d", j)), wi.scope(fmt.Sprintf("resnets.%d", j)), x, cur, out)
			cur = out
		}
		if upFlag {
			us := bi.Scope("upsample")
			x = us.Op("Resize", graph.Attr("mode", "nearest", "coordinate_transformation_mode", "asymmetric", "nearest_mode", "floor"),
				x, nil, us.Const("scales", tensor.FromF32([]float32{1, 1, 2, 2}, 4)))
			x = conv(us, wi.scope("upsampler.resample.1"), x, 1)
			ft := 1
			if up[i] {
				ft = 2
			}
			x = bi.Add(x, dupUp(bi.Scope("shortcut"), xCopy, in, out, ft, hh, ww))
			hh, ww = 2*hh, 2*ww
		}
		ch = out
	}
	x = rmsNormC(b.Scope("norm_out"), d.f32("norm_out.gamma"), x, ch)
	x = b.SiLU(x)
	x = conv(b.Scope("conv_out"), d.scope("conv_out"), x, 1)
	b.Output("image", b.Op("Clip", nil, x, b.Scalar(-1), b.Scalar(1)))
	return b.Build()
}

// conv applies the scope's weight/bias as a same-padded conv.
func conv(b *graph.Builder, w weights, x *graph.Value, pad int) *graph.Value {
	return b.Conv2d(x, w.f32("weight"), w.f32("bias"), 1, pad)
}

// rmsNormC is QwenImage21RMS_norm over channels (dim 1):
// F.normalize(x, dim=1) · √C · γ, with F.normalize's max(‖x‖, 1e-12).
func rmsNormC(b *graph.Builder, gamma *tensor.Tensor, x *graph.Value, c int) *graph.Value {
	n := b.Op("ReduceL2", graph.Attr("keepdims", 1), x, b.Ints(1))
	n = b.Op("Max", nil, n, b.Scalar(1e-12))
	g := scaled(reshaped(gamma, 1, c, 1, 1), float32(math.Sqrt(float64(c))))
	return b.Mul(b.Div(x, n), b.Const("gamma", g))
}

// resBlock is QwenImage21ResidualBlock: RMS-norm → SiLU → conv3×3, twice,
// plus a 1×1 shortcut when the channel count changes.
func resBlock(b *graph.Builder, w weights, x *graph.Value, in, out int) *graph.Value {
	h := x
	if in != out {
		h = b.Scope("shortcut").Conv2d(x, w.f32("conv_shortcut.weight"), w.f32("conv_shortcut.bias"), 1, 0)
	}
	y := rmsNormC(b.Scope("norm1"), w.f32("norm1.gamma"), x, in)
	y = conv(b.Scope("conv1"), w.scope("conv1"), b.SiLU(y), 1)
	y = rmsNormC(b.Scope("norm2"), w.f32("norm2.gamma"), y, out)
	y = conv(b.Scope("conv2"), w.scope("conv2"), b.SiLU(y), 1)
	return b.Add(y, h)
}

// attnBlock is QwenImage21AttentionBlock: single-head self-attention over
// the h·w pixels, 1×1-conv projections, residual.
func attnBlock(b *graph.Builder, w weights, x *graph.Value, c, h, wd int) *graph.Value {
	t := h * wd
	y := rmsNormC(b.Scope("norm"), w.f32("norm.gamma"), x, c)
	qkv := b.Scope("to_qkv").Conv2d(y, w.f32("to_qkv.weight"), w.f32("to_qkv.bias"), 1, 0)
	qkv = b.Transpose(b.Reshape(qkv, 1, int64(3*c), int64(t)), 0, 2, 1) // [1, t, 3c]
	parts := b.OpN("Split", graph.Attr("axis", 2, "num_outputs", 3), 3, qkv)
	q := b.Reshape(parts[0], 1, 1, int64(t), int64(c))
	k := b.Reshape(parts[1], 1, 1, int64(t), int64(c))
	v := b.Reshape(parts[2], 1, 1, int64(t), int64(c))
	o := b.Op("ingot.SDPA", graph.Attr("scale", float32(1/math.Sqrt(float64(c))), "b_layout", 2), q, k, v)
	o = b.Transpose(b.Reshape(o, 1, int64(t), int64(c)), 0, 2, 1) // [1, c, t]
	o = b.Reshape(o, 1, int64(c), int64(h), int64(wd))
	o = b.Scope("proj").Conv2d(o, w.f32("proj.weight"), w.f32("proj.bias"), 1, 0)
	return b.Add(o, x)
}

// dupUp is QwenImage21DupUp3D for one first-chunk frame: channels are
// repeat-interleaved to out·ft·4, viewed as (out, ft, 2, 2), the last time
// slice kept (first_chunk drops ft-1 leading frames), and the 2×2 factor
// folded into space — one channel Gather plus a pixel shuffle.
func dupUp(b *graph.Builder, x *graph.Value, in, out, ft, h, w int) *graph.Value {
	repeats := out * ft * 4 / in
	idx := make([]int64, 0, out*4)
	for oc := range out {
		for hs := range 2 {
			for ws := range 2 {
				c := ((oc*ft+ft-1)*2+hs)*2 + ws
				idx = append(idx, int64(c/repeats))
			}
		}
	}
	y := b.Op("Gather", graph.Attr("axis", 1), x, b.Const("channels", tensor.FromI64(idx, len(idx))))
	y = b.Reshape(y, 1, int64(out), 2, 2, int64(h), int64(w))
	y = b.Transpose(y, 0, 1, 4, 2, 5, 3)
	return b.Reshape(y, 1, int64(out), int64(2*h), int64(2*w))
}

// UnpackLatents turns the DiT's packed latents [h·w, C] into the VAE's
// input [1, C, h, w], denormalised with the config's latents_mean/std.
func UnpackLatents(cfg VAEConfig, x *tensor.Tensor, h, w int) *tensor.Tensor {
	c := cfg.ZDim
	z := tensor.New(tensor.F32, 1, c, h, w)
	src, dst := x.F32(), z.F32()
	for p := range h * w {
		for ch := range c {
			dst[ch*h*w+p] = src[p*c+ch]*cfg.LatentsStd[ch] + cfg.LatentsMean[ch]
		}
	}
	return z
}
