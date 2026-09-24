//go:build darwin && arm64

package qwenimage

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// MetalVAE runs the VAE decoder on the GPU in NHWC: 3×3 convs as im2col +
// GEMM (in pixel chunks), 1×1 convs as GEMMs, the channel RMS-norm as a row
// RMSNorm, the mid-block attention with the DiT's GEMM/softmax kernels. The
// f32 conv weights are read in place from the mapped checkpoint. Same math
// as BuildVAEDecoder.
type MetalVAE struct {
	dev    *metal.Device
	cfg    VAEConfig
	set    *safetensors.Set
	shards map[unsafe.Pointer]*metal.Buffer
	small  map[string]*metal.Buffer // biases, norm gammas
}

// NewMetalVAE maps the decoder's weights onto the GPU.
func NewMetalVAE(cfg VAEConfig, set *safetensors.Set) (*MetalVAE, error) {
	if !cfg.IsResidual || cfg.PatchSize != nil || len(cfg.AttnScales) != 0 {
		return nil, fmt.Errorf("qwenimage: VAE config variant not implemented")
	}
	dev, err := metal.Open()
	if err != nil {
		return nil, err
	}
	if err := dev.PrepareConv(); err != nil {
		return nil, err
	}
	return &MetalVAE{dev: dev, cfg: cfg, set: set, shards: map[unsafe.Pointer]*metal.Buffer{}, small: map[string]*metal.Buffer{}}, nil
}

// weight returns name's f32 conv weight [out, in·k·k] in place.
func (v *MetalVAE) weight(name string, out, in int) (metal.Region, error) {
	mem, off, info, err := v.set.Locate(name)
	if err != nil {
		return metal.Region{}, err
	}
	n := 1
	for _, d := range info.Shape {
		n *= d
	}
	if info.DType != "F32" || len(info.Shape) < 2 || info.Shape[0] != out || n != out*in {
		return metal.Region{}, fmt.Errorf("qwenimage: %s is %s%v, want F32 [%d, %d…]", name, info.DType, info.Shape, out, in)
	}
	key := unsafe.Pointer(&mem[0])
	b, ok := v.shards[key]
	if !ok {
		if b, err = v.dev.Wrap(mem); err != nil {
			return metal.Region{}, err
		}
		v.shards[key] = b
	}
	return b.At(off), nil
}

// vec returns a small f32 weight (bias, gamma) as its own buffer.
func (v *MetalVAE) vec(name string) (*metal.Buffer, error) {
	if b, ok := v.small[name]; ok {
		return b, nil
	}
	b, err := f32Buffer(v.dev, v.set, name)
	if err != nil {
		return nil, err
	}
	v.small[name] = b
	return b, nil
}

// Decode turns packed, denormalised latents z [h·w, z_dim] (NHWC) into the
// image [out_channels, 16h, 16w] in [-1, 1].
func (v *MetalVAE) Decode(z *tensor.Tensor, h, w int) (img *tensor.Tensor, err error) {
	cfg := v.cfg
	if z.Numel() != h*w*cfg.ZDim {
		return nil, fmt.Errorf("qwenimage: latents %v, want [%d, %d]", z.Shape(), h*w, cfg.ZDim)
	}
	mult := cfg.DimMult
	dims := []int{cfg.DecoderBaseDim * mult[len(mult)-1]}
	for i := len(mult) - 1; i >= 0; i-- {
		dims = append(dims, cfg.DecoderBaseDim*mult[i])
	}
	up := make([]bool, len(cfg.TemperalDownsample))
	for i, t := range cfg.TemperalDownsample {
		up[len(up)-1-i] = t
	}
	// Largest activation (pixels × channels) over every stage sizes the
	// ping-pong buffers; the upsampled tensor of each up block is the peak.
	maxAct, hw := h*w*max(dims[0], cfg.ZDim), h*w
	for i := range len(dims) - 1 {
		maxAct = max(maxAct, hw*max(dims[i], dims[i+1]))
		if i != len(mult)-1 {
			hw *= 4
			maxAct = max(maxAct, hw*max(dims[i], dims[i+1]))
		}
	}
	const colsBytes = 256 << 20
	var bufs []*metal.Buffer
	defer func() {
		for _, b := range bufs {
			b.Release()
		}
	}()
	nb := func(bytes int) *metal.Buffer {
		b, e := v.dev.NewBuffer(bytes)
		if e != nil && err == nil {
			err = e
		}
		if b != nil {
			bufs = append(bufs, b)
		}
		return b
	}
	x, a, b2, sc, bin := nb(4*maxAct), nb(4*maxAct), nb(4*maxAct), nb(4*maxAct), nb(4*maxAct)
	cols := nb(colsBytes)
	one := nb(4 * 3 * max(dims[0], cfg.ZDim)) // ones: residual adds as GatedAdd(g = 1)
	P0 := h * w
	S := nb(4 * P0 * P0) // mid-block attention scores (latent resolution)
	idxBufs := map[string]*metal.Buffer{}
	if err != nil {
		return nil, err
	}
	copy(f32view(x), z.F32())
	for i := range f32view(one) {
		f32view(one)[i] = 1
	}

	// Resolve every weight before encoding (the encoder callback runs on
	// the device thread and cannot call back into Device).
	type conv struct {
		w          metal.Region
		b          *metal.Buffer
		in, out, k int
	}
	getConv := func(prefix string, in, out, k int) conv {
		c := conv{in: in, out: out, k: k}
		if err == nil {
			c.w, err = v.weight(prefix+".weight", out, in*k*k)
		}
		if err == nil {
			c.b, err = v.vec(prefix + ".bias")
		}
		return c
	}
	getVec := func(name string) *metal.Buffer {
		var r *metal.Buffer
		if err == nil {
			r, err = v.vec(name)
		}
		return r
	}
	type res struct {
		n1, n2  *metal.Buffer
		c1, c2  conv
		short   *conv
		in, out int
	}
	getRes := func(p string, in, out int) res {
		r := res{in: in, out: out, n1: getVec(p + ".norm1.gamma"), n2: getVec(p + ".norm2.gamma"),
			c1: getConv(p+".conv1", in, out, 3), c2: getConv(p+".conv2", out, out, 3)}
		if in != out {
			s := getConv(p+".conv_shortcut", in, out, 1)
			r.short = &s
		}
		return r
	}
	d := "decoder"
	post := getConv("post_quant_conv", cfg.ZDim, cfg.ZDim, 1)
	convIn := getConv(d+".conv_in", cfg.ZDim, dims[0], 3)
	mid0, mid1 := getRes(d+".mid_block.resnets.0", dims[0], dims[0]), getRes(d+".mid_block.resnets.1", dims[0], dims[0])
	attnNorm := getVec(d + ".mid_block.attentions.0.norm.gamma")
	qkv := getConv(d+".mid_block.attentions.0.to_qkv", dims[0], 3*dims[0], 1)
	proj := getConv(d+".mid_block.attentions.0.proj", dims[0], dims[0], 1)
	type upBlock struct {
		res         []res
		resample    *conv
		in, out, ft int
		idx         *metal.Buffer
	}
	var ups []upBlock
	for i := range len(dims) - 1 {
		in, out := dims[i], dims[i+1]
		u := upBlock{in: in, out: out, ft: 1}
		cur := in
		for j := range cfg.NumResBlocks + 1 {
			u.res = append(u.res, getRes(fmt.Sprintf("%s.up_blocks.%d.resnets.%d", d, i, j), cur, out))
			cur = out
		}
		if i != len(mult)-1 {
			c := getConv(fmt.Sprintf("%s.up_blocks.%d.upsampler.resample.1", d, i), out, out, 3)
			u.resample = &c
			if up[i] {
				u.ft = 2
			}
			key := fmt.Sprintf("%d/%d/%d", in, out, u.ft)
			if idxBufs[key] == nil && err == nil {
				ib := nb(4 * out * 4)
				iv := unsafe.Slice((*uint32)(unsafe.Pointer(&ib.Bytes()[0])), out*4)
				repeats := out * u.ft * 4 / in
				for oc := range out {
					for hs := range 2 {
						for ws := range 2 {
							iv[(oc*2+hs)*2+ws] = uint32(((oc*u.ft+u.ft-1)*2+hs)*2+ws) / uint32(repeats)
						}
					}
				}
				idxBufs[key] = ib
			}
			u.idx = idxBufs[key]
		}
		ups = append(ups, u)
	}
	normOut := getVec(d + ".norm_out.gamma")
	convOut := getConv(d+".conv_out", dims[len(dims)-1], cfg.OutChannels, 3)
	if err != nil {
		return nil, err
	}

	// eps: F.normalize divides by max(‖x‖, 1e-12); rsqrt(mean + eps) with a
	// negligible eps is the same map for any pixel that is not all zeros.
	const eps = 1e-30
	err = v.dev.Run(func(e *metal.Encoder) {
		// conv applies c to src [HW, c.in] into dst [HW, c.out].
		applyConv := func(src, dst *metal.Buffer, H, W int, c conv) {
			P := H * W
			if c.k == 1 {
				e.Gemm(metal.Gemm{M: P, N: c.out, K: c.in, A: src.At(0), B: c.w, C: dst.At(0), TransB: true})
			} else {
				chunk := max(1, colsBytes/(4*9*c.in))
				for p0 := 0; p0 < P; p0 += chunk {
					n := min(chunk, P-p0)
					e.Im2Col3x3(src.At(0), cols.At(0), H, W, c.in, p0, n)
					e.Gemm(metal.Gemm{M: n, N: c.out, K: 9 * c.in, A: cols.At(0), B: c.w, C: dst.At(4 * p0 * c.out), TransB: true})
				}
			}
			e.AddBias(dst.At(0), c.b.At(0), P, c.out, c.out)
		}
		// resBlock: x [HW, in] → x [HW, out]. The result is built in b2
		// and the two swap, so x always names the live activation.
		resBlock := func(H, W int, r res) {
			P := H * W
			hsrc := x
			if r.short != nil {
				applyConv(x, sc, H, W, *r.short)
				hsrc = sc
			}
			e.RMSNormRows(x.At(0), a.At(0), r.n1.At(0), P, r.in, r.in, r.in, eps)
			e.SiLU(a.At(0), P*r.in)
			applyConv(a, b2, H, W, r.c1)
			e.RMSNormRows(b2.At(0), a.At(0), r.n2.At(0), P, r.out, r.out, r.out, eps)
			e.SiLU(a.At(0), P*r.out)
			applyConv(a, b2, H, W, r.c2)
			e.GatedAdd(b2.At(0), one.At(0), hsrc.At(0), P, r.out, r.out, r.out)
			x, b2 = b2, x
		}
		H, W := h, w
		applyConv(x, a, H, W, post)
		applyConv(a, x, H, W, convIn)
		resBlock(H, W, mid0)
		{ // mid attention: single head over the H·W pixels, dim C
			C, P := dims[0], H*W
			e.RMSNormRows(x.At(0), a.At(0), attnNorm.At(0), P, C, C, C, eps)
			applyConv(a, b2, H, W, qkv) // b2 = [P, 3C] = [q | k | v]
			e.Gemm(metal.Gemm{M: P, N: P, K: C, A: b2.At(0), LDA: 3 * C, B: b2.At(4 * C), LDB: 3 * C, C: S.At(0), TransB: true})
			e.SoftmaxRows(S.At(0), P, P, P, float32(1/math.Sqrt(float64(C))))
			e.Gemm(metal.Gemm{M: P, N: C, K: P, A: S.At(0), B: b2.At(4 * 2 * C), LDB: 3 * C, C: a.At(0)})
			applyConv(a, b2, H, W, proj)
			e.GatedAdd(x.At(0), one.At(0), b2.At(0), P, C, C, C)
		}
		resBlock(H, W, mid1)
		for _, u := range ups {
			if u.resample != nil {
				e.CopyF32(x.At(0), bin.At(0), H*W*u.in) // block input, for the pixel-shuffle shortcut
			}
			for _, r := range u.res {
				resBlock(H, W, r)
			}
			if u.resample == nil {
				continue
			}
			e.Upsample2x(x.At(0), a.At(0), H, W, u.out)
			applyConv(a, b2, 2*H, 2*W, *u.resample)
			e.DepthToSpace2Map(bin.At(0), x.At(0), u.idx.At(0), H, W, u.in, u.out)
			e.GatedAdd(x.At(0), one.At(0), b2.At(0), 4*H*W, u.out, u.out, u.out)
			H, W = 2*H, 2*W
		}
		C := dims[len(dims)-1]
		e.RMSNormRows(x.At(0), a.At(0), normOut.At(0), H*W, C, C, C, eps)
		e.SiLU(a.At(0), H*W*C)
		applyConv(a, b2, H, W, convOut)
	})
	if err != nil {
		return nil, err
	}
	// NHWC → [C, H, W], clamped.
	Ho, Wo, Co := 16*h, 16*w, cfg.OutChannels
	img = tensor.New(tensor.F32, Co, Ho, Wo)
	src, dst := f32view(b2), img.F32()
	for p := range Ho * Wo {
		for c := range Co {
			dst[c*Ho*Wo+p] = min(max(src[p*Co+c], -1), 1)
		}
	}
	return img, nil
}

// Close releases the GPU buffers (the mapped weights stay the caller's).
func (v *MetalVAE) Close() {
	for _, b := range v.small {
		b.Release()
	}
	for _, b := range v.shards {
		b.Release()
	}
}
