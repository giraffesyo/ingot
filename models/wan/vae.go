package wan

import (
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// VAEConfig is vae/config.json (diffusers AutoencoderKLWan).
type VAEConfig struct {
	BaseDim            int       `json:"base_dim"`
	DecoderBaseDim     int       `json:"decoder_base_dim"`
	ZDim               int       `json:"z_dim"`
	DimMult            []int     `json:"dim_mult"`
	NumResBlocks       int       `json:"num_res_blocks"`
	AttnScales         []float64 `json:"attn_scales"`
	TemperalDownsample []bool    `json:"temperal_downsample"`
	IsResidual         bool      `json:"is_residual"`
	InChannels         int       `json:"in_channels"`
	OutChannels        int       `json:"out_channels"`
	PatchSize          int       `json:"patch_size"`
	ScaleSpatial       int       `json:"scale_factor_spatial"`
	ScaleTemporal      int       `json:"scale_factor_temporal"`
	LatentsMean        []float32 `json:"latents_mean"`
	LatentsStd         []float32 `json:"latents_std"`
}

// LoadVAEConfig reads dir/config.json and rejects variants not built here
// (the Wan 2.1 VAE: no residual blocks, no patchify).
func LoadVAEConfig(dir string) (VAEConfig, error) {
	var c VAEConfig
	if err := readJSON(dir, "config.json", &c); err != nil {
		return c, err
	}
	if c.DecoderBaseDim == 0 {
		c.DecoderBaseDim = c.BaseDim
	}
	if !c.IsResidual || c.PatchSize != 2 || len(c.AttnScales) != 0 || len(c.DimMult) != len(c.TemperalDownsample)+1 ||
		c.InChannels != 3*c.PatchSize*c.PatchSize || c.OutChannels != c.InChannels ||
		len(c.LatentsMean) != c.ZDim || len(c.LatentsStd) != c.ZDim {
		return c, fmt.Errorf("wan: VAE variant not implemented (is_residual=%v patch_size=%d attn_scales=%v in=%d out=%d)",
			c.IsResidual, c.PatchSize, c.AttnScales, c.InChannels, c.OutChannels)
	}
	return c, nil
}

// Spatial is the pixel size of one latent cell (patch size × 2^downsamples).
func (c VAEConfig) Spatial() int { return c.PatchSize << (len(c.DimMult) - 1) }

// Frames is the number of video frames a latent of t frames decodes to.
func (c VAEConfig) Frames(t int) int { return 1 + c.temporal()*(t-1) }

// LatentFrames is the latent length for n video frames (n ≡ 1 mod 4).
func (c VAEConfig) LatentFrames(n int) int { return (n-1)/c.temporal() + 1 }

func (c VAEConfig) temporal() int {
	f := 1
	for _, t := range c.TemperalDownsample {
		if t {
			f *= 2
		}
	}
	return f
}

// PatchifyImage turns an image [3, H, W] in [-1, 1] into the encoder's
// input [1, 12, H/2, W/2] (the reference's patchify: channel c·4 + kw·2 +
// kh).
func PatchifyImage(img *tensor.Tensor) *tensor.Tensor {
	s := img.Shape()
	C, H, W := s[0], s[1], s[2]
	h, w := H/2, W/2
	out := tensor.New(tensor.F32, 1, C*4, h, w)
	src, dst := img.F32(), out.F32()
	for c := range C {
		for kw := range 2 {
			for kh := range 2 {
				plane := dst[(c*4+kw*2+kh)*h*w:]
				for i := range h {
					for j := range w {
						plane[i*w+j] = src[(c*H+2*i+kh)*W+2*j+kw]
					}
				}
			}
		}
	}
	return out
}

// UnpatchifyFrames turns decoder output [F, 12, h, w] into video frames
// [F, 3, 2h, 2w].
func UnpatchifyFrames(x *tensor.Tensor) *tensor.Tensor {
	s := x.Shape()
	F, C, h, w := s[0], s[1]/4, s[2], s[3]
	H, W := 2*h, 2*w
	out := tensor.New(tensor.F32, F, C, H, W)
	src, dst := x.F32(), out.F32()
	for f := range F {
		for c := range C {
			for kw := range 2 {
				for kh := range 2 {
					plane := src[((f*C*4)+c*4+kw*2+kh)*h*w:]
					for i := range h {
						for j := range w {
							dst[((f*C+c)*H+2*i+kh)*W+2*j+kw] = plane[i*w+j]
						}
					}
				}
			}
		}
	}
	return out
}

// convTaps returns a causal Conv3d weight [O, I, kt, kh, kw] as kt 2-D
// weights [O, I, kh, kw], tap k applying to frame t−(kt−1)+k.
func convTaps(w *tensor.Tensor) []*tensor.Tensor {
	s := w.Shape()
	O, I, kt, kh, kw := s[0], s[1], s[2], s[3], s[4]
	src := w.F32()
	taps := make([]*tensor.Tensor, kt)
	for k := range kt {
		t := tensor.New(tensor.F32, O, I, kh, kw)
		dst := t.F32()
		for o := range O {
			for i := range I {
				copy(dst[(o*I+i)*kh*kw:(o*I+i+1)*kh*kw], src[((o*I+i)*kt+k)*kh*kw:((o*I+i)*kt+k+1)*kh*kw])
			}
		}
		taps[k] = t
	}
	return taps
}

// vae holds what a VAE graph shares while building. Activations are
// [frames, C, h, w]: frames ride the batch axis, so every spatial op is
// the 2-D one and a causal 3-D conv is a sum of 2-D convs over shifted
// frames.
type vae struct {
	cfg   VAEConfig
	w     weights
	first bool // the first chunk: one frame, temporal paths in their first-chunk form
	emit  bool // output the first chunk's caches (the decoder; the encoder runs one chunk)
	// caches lists the causal convs' cached inputs in call order.
	caches []cacheInfo
}

// cacheInfo is one causal conv's cache: its channels and size, and whether
// the first chunk leaves it zero (an upsampler's time conv, skipped there).
type cacheInfo struct {
	C, H, W int
	zero    bool
}

func cacheIn(i int) string  { return fmt.Sprintf("cache.%d", i) }
func cacheOut(i int) string { return fmt.Sprintf("cache_out.%d", i) }

// causalConv applies a causal Conv3d (kernel kt×k×k, same spatial padding)
// to x [F, C, h, w]. First chunk: the zero cache leaves only the last tap,
// and the cache output is x itself (one frame; the host prepends the zero
// frame). Otherwise the cache input [2, C, h, w] is prepended, each tap
// convolves its shifted F frames, and the last two frames are the new
// cache.
func (m *vae) causalConv(b *graph.Builder, w weights, x *graph.Value, F, C, h, wd int) *graph.Value {
	taps := convTaps(w.f32("weight"))
	bias := w.f32("bias")
	pad := (taps[0].Shape()[2] - 1) / 2
	kt := len(taps)
	if kt == 1 {
		return b.Conv2d(x, taps[0], bias, 1, pad)
	}
	idx := len(m.caches)
	m.caches = append(m.caches, cacheInfo{C: C, H: h, W: wd})
	if m.first {
		if m.emit {
			b.Output(cacheOut(idx), x)
		}
		return b.Conv2d(x, taps[kt-1], bias, 1, pad)
	}
	cache := b.Input(cacheIn(idx), tensor.F32, kt-1, C, h, wd)
	all := b.Concat(0, cache, x) // [F+2, C, h, w]
	var y *graph.Value
	for k := range kt {
		var bk *tensor.Tensor
		if k == kt-1 {
			bk = bias
		}
		yk := b.Conv2d(b.Slice(all, 0, int64(k), int64(k+F)), taps[k], bk, 1, pad)
		if y == nil {
			y = yk
		} else {
			y = b.Add(y, yk)
		}
	}
	b.Output(cacheOut(idx), b.Slice(all, 0, int64(F), int64(F+kt-1)))
	return y
}

// pointConv applies a 1×1(×1) conv.
func pointConv(b *graph.Builder, w weights, x *graph.Value) *graph.Value {
	wt := w.f32("weight")
	s := wt.Shape()
	return b.Conv2d(x, wt.Reshape(s[0], s[1], 1, 1), w.f32("bias"), 1, 0)
}

// rmsNormC is WanRMS_norm over channels (axis 1): F.normalize(x, dim=1) ·
// √C · γ, with F.normalize's max(‖x‖, 1e-12).
func rmsNormC(b *graph.Builder, gamma *tensor.Tensor, x *graph.Value, c int) *graph.Value {
	n := b.Op("ReduceL2", graph.Attr("keepdims", 1), x, b.Ints(1))
	n = b.Op("Max", nil, n, b.Scalar(1e-12))
	s := float32(math.Sqrt(float64(c)))
	g := tensor.New(tensor.F32, 1, c, 1, 1)
	for i, v := range gamma.F32() {
		g.F32()[i] = v * s
	}
	return b.Mul(b.Div(x, n), b.Const("gamma", g))
}

// resBlock is WanResidualBlock: RMS-norm → SiLU → causal conv, twice, plus
// a 1×1 shortcut when the channel count changes.
func (m *vae) resBlock(b *graph.Builder, w weights, x *graph.Value, F, in, out, h, wd int) *graph.Value {
	sc := x
	if in != out {
		sc = pointConv(b.Scope("shortcut"), w.scope("conv_shortcut"), x)
	}
	y := rmsNormC(b.Scope("norm1"), w.f32("norm1.gamma"), x, in)
	y = m.causalConv(b.Scope("conv1"), w.scope("conv1"), b.SiLU(y), F, in, h, wd)
	y = rmsNormC(b.Scope("norm2"), w.f32("norm2.gamma"), y, out)
	y = m.causalConv(b.Scope("conv2"), w.scope("conv2"), b.SiLU(y), F, out, h, wd)
	return b.Add(y, sc)
}

// attnBlock is WanAttentionBlock: per frame, single-head self-attention
// over the h·w pixels, 1×1-conv projections, residual.
func attnBlock(b *graph.Builder, w weights, x *graph.Value, F, c, h, wd int) *graph.Value {
	t := int64(h * wd)
	f, ci := int64(F), int64(c)
	y := rmsNormC(b.Scope("norm"), w.f32("norm.gamma"), x, c)
	qkv := pointConv(b.Scope("to_qkv"), w.scope("to_qkv"), y)
	qkv = b.Transpose(b.Reshape(qkv, f, 3*ci, t), 0, 2, 1) // [F, t, 3c]
	parts := b.OpN("Split", graph.Attr("axis", 2, "num_outputs", 3), 3, qkv)
	q := b.Reshape(parts[0], f, 1, t, ci)
	k := b.Reshape(parts[1], f, 1, t, ci)
	v := b.Reshape(parts[2], f, 1, t, ci)
	o := b.Op("ingot.SDPA", graph.Attr("scale", float32(1/math.Sqrt(float64(c))), "b_layout", 2), q, k, v)
	o = b.Transpose(b.Reshape(o, f, t, ci), 0, 2, 1) // [F, c, t]
	o = b.Reshape(o, f, ci, int64(h), int64(wd))
	return b.Add(pointConv(b.Scope("proj"), w.scope("proj"), o), x)
}

// midBlock is WanMidBlock: residual block, attention, residual block.
func (m *vae) midBlock(b *graph.Builder, w weights, x *graph.Value, F, c, h, wd int) *graph.Value {
	x = m.resBlock(b.Scope("res0"), w.scope("resnets.0"), x, F, c, c, h, wd)
	x = attnBlock(b.Scope("attn"), w.scope("attentions.0"), x, F, c, h, wd)
	return m.resBlock(b.Scope("res1"), w.scope("resnets.1"), x, F, c, c, h, wd)
}

// BuildVAEEncoder builds the image encoder for an H×W image (one frame,
// the first chunk): input "x" [1, 12, H/2, W/2] (PatchifyImage), output
// "z" [1, z_dim, H/16, W/16], the latent distribution's mean (the pipeline
// encodes with sample_mode="argmax"), not yet normalised.
func BuildVAEEncoder(cfg VAEConfig, set *safetensors.Set, H, W int) (g *graph.Graph, err error) {
	if s := cfg.Spatial(); H%s != 0 || W%s != 0 {
		return nil, fmt.Errorf("wan: image %dx%d not a multiple of %d", W, H, s)
	}
	defer catch(&err)
	m := &vae{cfg: cfg, w: weights{set: set}, first: true}
	b := graph.NewBuilder("wan_vae_encoder")
	e := m.w.scope("encoder")
	dims := []int{cfg.BaseDim}
	for _, k := range cfg.DimMult {
		dims = append(dims, cfg.BaseDim*k)
	}
	h, w := H/cfg.PatchSize, W/cfg.PatchSize
	x := b.Input("x", tensor.F32, 1, cfg.InChannels, h, w)
	x = m.causalConv(b.Scope("conv_in"), e.scope("conv_in"), x, 1, cfg.InChannels, h, w)
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
			x = m.resBlock(bi.Scope(fmt.Sprintf("res%d", j)), wi.scope(fmt.Sprintf("resnets.%d", j)), x, 1, cur, out, h, w)
			cur = out
		}
		fs := 1
		if down {
			ds, rw := bi.Scope("downsample"), wi.scope("downsampler.resample.1")
			// ZeroPad2d((0, 1, 0, 1)) then a stride-2 3×3 conv; the
			// first chunk skips the temporal downsampler's time conv.
			x = ds.Op("Conv", graph.Attr("strides", []int{2, 2}, "pads", []int{0, 0, 1, 1}),
				x, ds.Const("weight", rw.f32("weight")), ds.Const("bias", rw.f32("bias")))
			fs = 2
		}
		x = bi.Add(x, avgDown(bi.Scope("shortcut"), xCopy, in, out, ft, fs, h, w))
		if down {
			h, w = h/2, w/2
		}
	}
	C := dims[len(dims)-1]
	x = m.midBlock(b.Scope("mid"), e.scope("mid_block"), x, 1, C, h, w)
	x = rmsNormC(b.Scope("norm_out"), e.f32("norm_out.gamma"), x, C)
	x = m.causalConv(b.Scope("conv_out"), e.scope("conv_out"), b.SiLU(x), 1, C, h, w)
	x = pointConv(b.Scope("quant_conv"), m.w.scope("quant_conv"), x)
	b.Output("z", b.Slice(x, 1, 0, int64(cfg.ZDim)))
	return b.Build()
}

// avgDown is AvgDown3D for one first-chunk frame: space-to-depth by fs, a
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

// dupUp is DupUp3D over x [F, in, h, w]: channels repeat-interleaved to
// out·ft·4 and viewed as (out, ft, 2, 2) — the ft time steps become frames,
// the 2×2 factor folds into space: [F·ft, out, 2h, 2w]. The first chunk
// keeps only the last time step (first_chunk drops ft−1 leading frames).
func (m *vae) dupUp(b *graph.Builder, x *graph.Value, F, in, out, ft, h, w int) *graph.Value {
	repeats := out * ft * 4 / in
	t0, nt := 0, ft
	if m.first {
		t0, nt = ft-1, 1
	}
	idx := make([]int64, 0, nt*out*4)
	for t := t0; t < ft; t++ {
		for oc := range out {
			for hs := range 2 {
				for ws := range 2 {
					idx = append(idx, int64((((oc*ft+t)*2+hs)*2+ws)/repeats))
				}
			}
		}
	}
	y := b.Op("Gather", graph.Attr("axis", 1), x, b.Const("channels", tensor.FromI64(idx, len(idx))))
	Fo := int64(F * nt)
	y = b.Reshape(y, Fo, int64(out), 2, 2, int64(h), int64(w))
	y = b.Transpose(y, 0, 1, 4, 2, 5, 3)
	return b.Reshape(y, Fo, int64(out), int64(2*h), int64(2*w))
}

// VAEDecoder decodes latents chunk by chunk, one latent frame per chunk,
// carrying every causal conv's last two input frames between chunks — the
// reference's streaming decode, so a clip of any length decodes in the
// memory of one chunk. The first chunk yields one frame, each later chunk
// four.
type VAEDecoder struct {
	cfg    VAEConfig
	h, w   int // latent size
	first  *graph.Graph
	steady *graph.Graph
	caches []cacheInfo
}

// BuildVAEDecoder builds the decoder graphs for an h×w latent: input "z"
// [1, z_dim, h, w] (one denormalised latent frame) plus, after the first
// chunk, the caches; output "frames" [F, 12, 8h, 8w] (UnpatchifyFrames
// makes [F, 3, 16h, 16w]) clamped to [-1, 1].
func BuildVAEDecoder(cfg VAEConfig, set *safetensors.Set, h, w int) (d *VAEDecoder, err error) {
	d = &VAEDecoder{cfg: cfg, h: h, w: w}
	var c0 []cacheInfo
	if d.first, c0, err = buildDecoderChunk(cfg, set, h, w, true); err != nil {
		return nil, err
	}
	if d.steady, d.caches, err = buildDecoderChunk(cfg, set, h, w, false); err != nil {
		return nil, err
	}
	if len(c0) != len(d.caches) {
		return nil, fmt.Errorf("wan: VAE decoder chunk graphs disagree on caches (%d vs %d)", len(c0), len(d.caches))
	}
	for i := range c0 {
		d.caches[i].zero = c0[i].zero
	}
	return d, nil
}

// Graphs returns the first-chunk and steady-chunk graphs (for compiling).
func (d *VAEDecoder) Graphs() (first, steady *graph.Graph) { return d.first, d.steady }

func buildDecoderChunk(cfg VAEConfig, set *safetensors.Set, h, w int, first bool) (g *graph.Graph, caches []cacheInfo, err error) {
	defer catch(&err)
	m := &vae{cfg: cfg, w: weights{set: set}, first: first, emit: true}
	b := graph.NewBuilder("wan_vae_decoder")
	dw := m.w.scope("decoder")
	mult := cfg.DimMult
	dims := []int{cfg.DecoderBaseDim * mult[len(mult)-1]}
	for i := len(mult) - 1; i >= 0; i-- {
		dims = append(dims, cfg.DecoderBaseDim*mult[i])
	}
	up := make([]bool, len(cfg.TemperalDownsample)) // temperal_upsample = reversed
	for i, v := range cfg.TemperalDownsample {
		up[len(up)-1-i] = v
	}
	x := b.Input("z", tensor.F32, 1, cfg.ZDim, h, w)
	x = pointConv(b.Scope("post_quant_conv"), m.w.scope("post_quant_conv"), x)
	F := 1
	x = m.causalConv(b.Scope("conv_in"), dw.scope("conv_in"), x, F, cfg.ZDim, h, w)
	x = m.midBlock(b.Scope("mid"), dw.scope("mid_block"), x, F, dims[0], h, w)
	hh, ww := h, w
	for i := range len(dims) - 1 {
		in, out := dims[i], dims[i+1]
		upFlag := i != len(mult)-1
		bi, wi := b.Scope(fmt.Sprintf("up%d", i)), dw.scope(fmt.Sprintf("up_blocks.%d", i))
		xCopy := x
		cur := in
		for j := range cfg.NumResBlocks + 1 {
			x = m.resBlock(bi.Scope(fmt.Sprintf("res%d", j)), wi.scope(fmt.Sprintf("resnets.%d", j)), x, F, cur, out, hh, ww)
			cur = out
		}
		if !upFlag {
			continue
		}
		us := bi.Scope("upsample")
		ft, Fo := 1, F
		if up[i] {
			ft = 2
			// upsample3d's time conv doubles the frames (channel halves
			// become consecutive frames); the first chunk skips it and
			// its cache stays zero.
			if m.first {
				m.caches = append(m.caches, cacheInfo{C: out, H: hh, W: ww, zero: true})
			} else {
				y := m.causalConv(us.Scope("time_conv"), wi.scope("upsampler.time_conv"), x, F, out, hh, ww)
				Fo = 2 * F
				x = us.Reshape(y, int64(Fo), int64(out), int64(hh), int64(ww))
			}
		}
		x = us.Op("Resize", graph.Attr("mode", "nearest", "coordinate_transformation_mode", "asymmetric", "nearest_mode", "floor"),
			x, nil, us.Const("scales", tensor.FromF32([]float32{1, 1, 2, 2}, 4)))
		rw := wi.scope("upsampler.resample.1")
		x = us.Conv2d(x, rw.f32("weight"), rw.f32("bias"), 1, 1)
		x = bi.Add(x, m.dupUp(bi.Scope("shortcut"), xCopy, F, in, out, ft, hh, ww))
		F, hh, ww = Fo, 2*hh, 2*ww
	}
	C := dims[len(dims)-1]
	x = rmsNormC(b.Scope("norm_out"), dw.f32("norm_out.gamma"), x, C)
	x = m.causalConv(b.Scope("conv_out"), dw.scope("conv_out"), b.SiLU(x), F, C, hh, ww)
	b.Output("frames", b.Op("Clip", nil, x, b.Scalar(-1), b.Scalar(1)))
	g, err = b.Build()
	return g, m.caches, err
}

// Decoder runs a VAEDecoder over a latent video through two compiled
// runners (first chunk, steady chunks).
type Decoder struct {
	d             *VAEDecoder
	first, steady graph.Runner
}

// NewDecoder compiles d's graphs on device.
func NewDecoder(d *VAEDecoder, device string) (*Decoder, error) {
	first, err := graph.CompileOn(d.first, device)
	if err != nil {
		return nil, err
	}
	steady, err := graph.CompileOn(d.steady, device)
	if err != nil {
		return nil, err
	}
	return &Decoder{d: d, first: first, steady: steady}, nil
}

// Decode decodes denormalised latents z [z_dim, T, h, w] to video frames
// [1+4(T−1), 3, 16h, 16w] in [-1, 1]. frame, when set, receives each
// chunk's frames ([F, 3, H, W]) as they are decoded.
func (dc *Decoder) Decode(z *tensor.Tensor, frame func(*tensor.Tensor) error) (*tensor.Tensor, error) {
	d := dc.d
	s := z.Shape()
	C, T, h, w := s[0], s[1], s[2], s[3]
	if C != d.cfg.ZDim || h != d.h || w != d.w {
		return nil, fmt.Errorf("wan: decoder built for %dx%d, latents %v", d.h, d.w, s)
	}
	H, W := h*d.cfg.Spatial(), w*d.cfg.Spatial()
	out := tensor.New(tensor.F32, d.cfg.Frames(T), 3, H, W)
	caches := make(map[string]*tensor.Tensor, len(d.caches))
	at := 0
	for t := range T {
		zt := tensor.New(tensor.F32, 1, C, h, w)
		for c := range C {
			copy(zt.F32()[c*h*w:(c+1)*h*w], z.F32()[(c*T+t)*h*w:(c*T+t+1)*h*w])
		}
		feeds := map[string]*tensor.Tensor{"z": zt}
		r := dc.steady
		if t == 0 {
			r = dc.first
		} else {
			for k, v := range caches {
				feeds[k] = v
			}
		}
		res, err := r.Run(feeds)
		if err != nil {
			return nil, fmt.Errorf("wan: VAE decode chunk %d: %w", t, err)
		}
		next := make(map[string]*tensor.Tensor, len(d.caches))
		for i, ci := range d.caches {
			v := res[cacheOut(i)]
			switch {
			case t == 0 && ci.zero:
				next[cacheIn(i)] = tensor.New(tensor.F32, 2, ci.C, ci.H, ci.W)
			case t == 0: // one frame: prepend the zero frame
				c := tensor.New(tensor.F32, 2, ci.C, ci.H, ci.W)
				copy(c.F32()[ci.C*ci.H*ci.W:], v.F32())
				next[cacheIn(i)] = c
			default:
				next[cacheIn(i)] = v.Clone()
			}
		}
		frames := UnpatchifyFrames(res["frames"])
		r.Release(res)
		caches = next
		n := frames.Numel()
		copy(out.F32()[at:at+n], frames.F32())
		at += n
		if frame != nil {
			if err := frame(frames); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}
