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
	InChannels         int       `json:"in_channels"`
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

// BuildVAEEncoder builds the image encoder for an H×W image: input "x"
// [1, in_channels, H, W] in [-1, 1]; output "z" [1, z_dim, H/16, W/16], the
// latent distribution's mode (its mean — the pipeline encodes with
// sample_mode="argmax"), not yet normalised. One frame: the temporal
// downsampling reduces to its first-chunk form (time_conv skipped; the
// shortcut averages a zero frame in).
func BuildVAEEncoder(cfg VAEConfig, set *safetensors.Set, H, W int) (g *graph.Graph, err error) {
	if !cfg.IsResidual || cfg.PatchSize != nil || len(cfg.AttnScales) != 0 {
		return nil, fmt.Errorf("qwenimage: VAE config variant not implemented")
	}
	f := 1 << (len(cfg.DimMult) - 1)
	if H%f != 0 || W%f != 0 {
		return nil, fmt.Errorf("qwenimage: VAE encoder input %dx%d not a multiple of %d", W, H, f)
	}
	defer catch(&err)
	b := graph.NewBuilder("qwenimage21_vae_encoder")
	wt := weights{set: set}
	e := wt.scope("encoder")
	dims := []int{cfg.BaseDim}
	for _, m := range cfg.DimMult {
		dims = append(dims, cfg.BaseDim*m)
	}
	x := b.Input("x", tensor.F32, 1, cfg.InChannels, H, W)
	x = conv(b.Scope("conv_in"), e.scope("conv_in"), x, 1)
	h, w := H, W
	last := len(cfg.DimMult) - 1
	for i := range last + 1 {
		in, out := dims[i], dims[i+1]
		down := i != last
		ft := 1
		if down && cfg.TemperalDownsample[i] {
			ft = 2
		}
		bi, wi := b.Scope(fmt.Sprintf("down%d", i)), e.scope(fmt.Sprintf("down_blocks.%d", i))
		xCopy := x
		cur := in
		for j := range cfg.NumResBlocks {
			x = resBlock(bi.Scope(fmt.Sprintf("res%d", j)), wi.scope(fmt.Sprintf("resnets.%d", j)), x, cur, out)
			cur = out
		}
		fs := 1
		if down {
			ds := bi.Scope("downsample")
			rw := wi.scope("downsampler.resample.1")
			// ZeroPad2d((0, 1, 0, 1)) then a stride-2 3×3 conv.
			x = ds.Op("Conv", graph.Attr("strides", []int{2, 2}, "pads", []int{0, 0, 1, 1}),
				x, ds.Const("weight", rw.f32("weight")), ds.Const("bias", rw.f32("bias")))
			fs = 2
		}
		x = bi.Add(x, avgDown(bi.Scope("shortcut"), xCopy, in, out, ft, fs, h, w))
		if down {
			h, w = h/2, w/2
		}
	}
	mid := e.scope("mid_block")
	C := dims[len(dims)-1]
	x = resBlock(b.Scope("mid.res0"), mid.scope("resnets.0"), x, C, C)
	x = attnBlock(b.Scope("mid.attn"), mid.scope("attentions.0"), x, C, h, w)
	x = resBlock(b.Scope("mid.res1"), mid.scope("resnets.1"), x, C, C)
	x = rmsNormC(b.Scope("norm_out"), e.f32("norm_out.gamma"), x, C)
	x = conv(b.Scope("conv_out"), e.scope("conv_out"), b.SiLU(x), 1)
	x = b.Scope("quant_conv").Conv2d(x, wt.f32("quant_conv.weight"), wt.f32("quant_conv.bias"), 1, 0)
	b.Output("z", b.Slice(x, 1, 0, int64(cfg.ZDim)))
	return b.Build()
}

// avgDown is QwenImage21AvgDown3D for one frame: space-to-depth by fs, a
// zero frame prepended in time when ft = 2, then each output channel the
// mean of its group of in·ft·fs²/out consecutive channels.
func avgDown(b *graph.Builder, x *graph.Value, in, out, ft, fs, h, w int) *graph.Value {
	factor := ft * fs * fs
	if factor == 1 && in == out {
		return x
	}
	ho, wo := h/fs, w/fs
	s2d := x
	if fs > 1 {
		s2d = b.Reshape(x, 1, int64(in), int64(ho), int64(fs), int64(wo), int64(fs))
		s2d = b.Transpose(s2d, 0, 1, 3, 5, 2, 4)
		s2d = b.Reshape(s2d, 1, int64(in*fs*fs), int64(ho), int64(wo))
	}
	src := s2d
	zero := int64(in * fs * fs) // index of an appended all-zero channel
	if ft > 1 {
		src = b.Concat(1, s2d, b.Const("zero_frame", tensor.New(tensor.F32, 1, 1, ho, wo)))
	}
	idx := make([]int64, 0, in*factor)
	for c := range in {
		for t := range ft {
			for k := range fs * fs {
				if ft > 1 && t == 0 { // the padded (first) frame
					idx = append(idx, zero)
					continue
				}
				idx = append(idx, int64(c*fs*fs+k))
			}
		}
	}
	y := b.Op("Gather", graph.Attr("axis", 1), src, b.Const("channels", tensor.FromI64(idx, len(idx))))
	group := in * factor / out
	y = b.Reshape(y, 1, int64(out), int64(group), int64(ho), int64(wo))
	return b.Op("ReduceMean", graph.Attr("keepdims", 0), y, b.Ints(2))
}

// PackLatents normalises encoder output z [1, C, h, w] ((z − mean)/std per
// channel, the pipeline's _encode_vae_image) and packs it as the DiT's
// condition tokens [h·w, C].
func PackLatents(cfg VAEConfig, z *tensor.Tensor) *tensor.Tensor {
	s := z.Shape()
	C, hw := s[1], s[2]*s[3]
	out := tensor.New(tensor.F32, hw, C)
	src, dst := z.F32(), out.F32()
	for c := range C {
		for p := range hw {
			dst[p*C+c] = (src[c*hw+p] - cfg.LatentsMean[c]) / cfg.LatentsStd[c]
		}
	}
	return out
}
