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

// MetalVAE runs the VAE decoder and encoder on the GPU in NHWC: 3×3 convs
// as im2col + GEMM (in pixel chunks), 1×1 convs as GEMMs, the channel
// RMS-norm as a row RMSNorm, the mid-block attention on the GEMM/softmax
// kernels. The f32 conv weights are read in place from the mapped
// checkpoint. Same math as BuildVAEDecoder / BuildVAEEncoder.
type MetalVAE struct {
	dev    *metal.Device
	cfg    VAEConfig
	set    *safetensors.Set
	shards map[unsafe.Pointer]*metal.Buffer
	small  map[string]*metal.Buffer // biases, norm gammas
	// bandBytes, attnBytes override decodeBandBytes and attnBytes (tests).
	bandBytes, attnBytes int
}

// NewMetalVAE maps the VAE's weights onto the GPU.
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

// Weight handles, resolved before encoding (Device.Run callbacks cannot call
// back into the device).
type (
	vaeConv struct {
		w          metal.Region
		b          *metal.Buffer
		in, out, k int
	}
	vaeRes struct {
		n1, n2  *metal.Buffer
		c1, c2  vaeConv
		short   *vaeConv
		in, out int
	}
	vaeAttn struct {
		norm      *metal.Buffer
		qkv, proj vaeConv
		c         int
	}
)

// vaeWeights resolves weights, keeping the first error.
type vaeWeights struct {
	v   *MetalVAE
	err error
}

func (r *vaeWeights) vec(name string) *metal.Buffer {
	if r.err != nil {
		return nil
	}
	if b, ok := r.v.small[name]; ok {
		return b
	}
	b, err := f32Buffer(r.v.dev, r.v.set, name)
	if err != nil {
		r.err = err
		return nil
	}
	r.v.small[name] = b
	return b
}

func (r *vaeWeights) conv(prefix string, in, out, k int) vaeConv {
	c := vaeConv{in: in, out: out, k: k}
	if r.err != nil {
		return c
	}
	mem, off, info, err := r.v.set.Locate(prefix + ".weight")
	if err != nil {
		r.err = err
		return c
	}
	n := 1
	for _, d := range info.Shape {
		n *= d
	}
	if info.DType != "F32" || len(info.Shape) < 2 || info.Shape[0] != out || n != out*in*k*k {
		r.err = fmt.Errorf("qwenimage: %s.weight is %s%v, want F32 [%d, %d, %d, %d]", prefix, info.DType, info.Shape, out, in, k, k)
		return c
	}
	key := unsafe.Pointer(&mem[0])
	b, ok := r.v.shards[key]
	if !ok {
		if b, err = r.v.dev.Wrap(mem); err != nil {
			r.err = err
			return c
		}
		r.v.shards[key] = b
	}
	c.w, c.b = b.At(off), r.vec(prefix+".bias")
	return c
}

func (r *vaeWeights) res(p string, in, out int) vaeRes {
	rb := vaeRes{in: in, out: out, n1: r.vec(p + ".norm1.gamma"), n2: r.vec(p + ".norm2.gamma"),
		c1: r.conv(p+".conv1", in, out, 3), c2: r.conv(p+".conv2", out, out, 3)}
	if in != out {
		s := r.conv(p+".conv_shortcut", in, out, 1)
		rb.short = &s
	}
	return rb
}

func (r *vaeWeights) attn(p string, c int) vaeAttn {
	return vaeAttn{c: c, norm: r.vec(p + ".norm.gamma"), qkv: r.conv(p+".to_qkv", c, 3*c, 1), proj: r.conv(p+".proj", c, c, 1)}
}

// vaeRun holds one pass's buffers and encoder; x always names the live
// activation.
type vaeRun struct {
	e                               *metal.Encoder
	x, a, b2, sc, bin, cols, one, S *metal.Buffer
}

// colsBytes bounds the im2col scratch; convs process pixels in chunks.
const colsBytes = 256 << 20

// eps for the channel RMS-norm: F.normalize divides by max(‖x‖, 1e-12);
// rsqrt(mean + eps) with a negligible eps is the same map for any pixel
// that is not all zeros.
const vaeEps = 1e-30

// conv applies c (stride 1, same padding) to src [H·W, c.in] into dst.
func (r *vaeRun) conv(src, dst *metal.Buffer, H, W int, c vaeConv) {
	e, P := r.e, H*W
	if c.k == 1 {
		e.Gemm(metal.Gemm{M: P, N: c.out, K: c.in, A: src.At(0), B: c.w, C: dst.At(0), TransB: true})
	} else {
		chunk := max(1, colsBytes/(4*9*c.in))
		for p0 := 0; p0 < P; p0 += chunk {
			n := min(chunk, P-p0)
			e.Im2Col3x3(src.At(0), r.cols.At(0), H, W, c.in, p0, n)
			e.Gemm(metal.Gemm{M: n, N: c.out, K: 9 * c.in, A: r.cols.At(0), B: c.w, C: dst.At(4 * p0 * c.out), TransB: true})
		}
	}
	e.AddBias(dst.At(0), c.b.At(0), P, c.out, c.out)
}

// convDown is ZeroPad2d((0, 1, 0, 1)) + a stride-2 3×3 conv: src [H·W,
// c.in] → dst [(H/2)·(W/2), c.out].
func (r *vaeRun) convDown(src, dst *metal.Buffer, H, W int, c vaeConv) {
	e := r.e
	OH, OW := H/2, W/2
	P := OH * OW
	chunk := max(1, colsBytes/(4*9*c.in))
	for p0 := 0; p0 < P; p0 += chunk {
		n := min(chunk, P-p0)
		e.Im2Col3x3Strided(src.At(0), r.cols.At(0), H, W, c.in, OW, 2, 0, 0, p0, n)
		e.Gemm(metal.Gemm{M: n, N: c.out, K: 9 * c.in, A: r.cols.At(0), B: c.w, C: dst.At(4 * p0 * c.out), TransB: true})
	}
	e.AddBias(dst.At(0), c.b.At(0), P, c.out, c.out)
}

// res is QwenImage21ResidualBlock on x (H×W).
func (r *vaeRun) res(H, W int, rb vaeRes) {
	e, P := r.e, H*W
	hsrc := r.x
	if rb.short != nil {
		r.conv(r.x, r.sc, H, W, *rb.short)
		hsrc = r.sc
	}
	e.RMSNormRows(r.x.At(0), r.a.At(0), rb.n1.At(0), P, rb.in, rb.in, rb.in, vaeEps)
	e.SiLU(r.a.At(0), P*rb.in)
	r.conv(r.a, r.b2, H, W, rb.c1)
	e.RMSNormRows(r.b2.At(0), r.a.At(0), rb.n2.At(0), P, rb.out, rb.out, rb.out, vaeEps)
	e.SiLU(r.a.At(0), P*rb.out)
	r.conv(r.a, r.b2, H, W, rb.c2)
	e.GatedAdd(r.b2.At(0), r.one.At(0), hsrc.At(0), P, rb.out, rb.out, rb.out)
	r.x, r.b2 = r.b2, r.x
}

// attn is QwenImage21AttentionBlock on x: one head over the H·W pixels,
// queries in chunks of qc rows (S holds qc×P scores).
func (r *vaeRun) attn(H, W int, at vaeAttn, qc int) {
	e, C, P := r.e, at.c, H*W
	e.RMSNormRows(r.x.At(0), r.a.At(0), at.norm.At(0), P, C, C, C, vaeEps)
	r.conv(r.a, r.b2, H, W, at.qkv) // [P, 3C] = [q | k | v]
	for q0 := 0; q0 < P; q0 += qc {
		n := min(qc, P-q0)
		e.Gemm(metal.Gemm{M: n, N: P, K: C, A: r.b2.At(4 * q0 * 3 * C), LDA: 3 * C, B: r.b2.At(4 * C), LDB: 3 * C, C: r.S.At(0), TransB: true})
		e.SoftmaxRows(r.S.At(0), n, P, P, float32(1/math.Sqrt(float64(C))))
		e.Gemm(metal.Gemm{M: n, N: C, K: P, A: r.S.At(0), B: r.b2.At(4 * 2 * C), LDB: 3 * C, C: r.a.At(4 * q0 * C)})
	}
	r.conv(r.a, r.b2, H, W, at.proj)
	e.GatedAdd(r.x.At(0), r.one.At(0), r.b2.At(0), P, C, C, C)
}

// buffers allocates a pass's scratch: maxAct floats per activation buffer,
// nS attention scores, ones for maxC channels.
func (v *MetalVAE) buffers(maxAct, nS, maxC int) (*vaeRun, func(), error) {
	var bufs []*metal.Buffer
	var err error
	nb := func(bytes int) *metal.Buffer {
		b, e := v.dev.NewBuffer(max(bytes, 4))
		if e != nil && err == nil {
			err = e
		}
		if b != nil {
			bufs = append(bufs, b)
		}
		return b
	}
	r := &vaeRun{x: nb(4 * maxAct), a: nb(4 * maxAct), b2: nb(4 * maxAct), sc: nb(4 * maxAct), bin: nb(4 * maxAct),
		cols: nb(colsBytes), one: nb(4 * maxC), S: nb(4 * nS)}
	free := func() {
		for _, b := range bufs {
			b.Release()
		}
	}
	if err != nil {
		free()
		return nil, nil, err
	}
	for i := range f32view(r.one) {
		f32view(r.one)[i] = 1
	}
	return r, free, nil
}

func (v *MetalVAE) dims() (enc, dec []int, up []bool) {
	cfg := v.cfg
	mult := cfg.DimMult
	enc = []int{cfg.BaseDim}
	for _, m := range mult {
		enc = append(enc, cfg.BaseDim*m)
	}
	dec = []int{cfg.DecoderBaseDim * mult[len(mult)-1]}
	for i := len(mult) - 1; i >= 0; i-- {
		dec = append(dec, cfg.DecoderBaseDim*mult[i])
	}
	up = make([]bool, len(cfg.TemperalDownsample))
	for i, t := range cfg.TemperalDownsample {
		up[len(up)-1-i] = t
	}
	return enc, dec, up
}

// The decoder's memory is bounded independently of the image size: after
// the mid block every op is local (3×3 convs, nearest upsampling, per-pixel
// norms and activations), so each up block runs over horizontal bands of
// its input with halo rows of context on either side, and the rows it keeps
// are exactly the full decode's. decodeBandBytes caps each activation
// buffer; attnBytes caps the mid attention's score block (queries are
// processed in chunks).
const (
	decodeBandBytes = 512 << 20
	attnBytes       = 256 << 20
)

// decodeStage is one up block (the last one with the output head) over
// band-sized pieces of its H×W input.
type decodeStage struct {
	H, W, in, out int
	up            bool
	halo, rows    int // context rows either side; input rows kept per band
}

// decodePlan is a decode's bands and scratch sizes.
type decodePlan struct {
	stages []decodeStage
	act    int // floats per activation buffer
	qChunk int // attention query rows per score block
	maxC   int
}

func (v *MetalVAE) decodePlan(h, w int) decodePlan {
	cfg := v.cfg
	_, dims, _ := v.dims()
	P := h * w
	p := decodePlan{act: P * max(3*dims[0], cfg.ZDim), maxC: 3 * max(dims[0], cfg.ZDim)}
	band, att := decodeBandBytes, attnBytes
	if v.bandBytes > 0 {
		band, att = v.bandBytes, v.attnBytes
	}
	p.qChunk = min(P, max(1, att/(4*P)))
	H, W := h, w
	nres := cfg.NumResBlocks + 1
	for i := range len(dims) - 1 {
		s := decodeStage{H: H, W: W, in: dims[i], out: dims[i+1], up: i != len(dims)-2, halo: 2*nres + 1}
		row := W * max(s.in, s.out) // floats per input row, peak over the stage
		if s.up {
			row = max(row, 4*W*s.out)
		} else {
			row = max(row, W*cfg.OutChannels)
		}
		s.rows = min(H, max(8, band/(4*row)-2*s.halo))
		p.act = max(p.act, min(H, s.rows+2*s.halo)*row)
		p.stages = append(p.stages, s)
		if s.up {
			H, W = 2*H, 2*W
		}
	}
	return p
}

// DecodeBytes is the GPU memory Decode needs for h×w latents beyond the
// mapped weights: its activation, im2col and attention scratch.
func (v *MetalVAE) DecodeBytes(h, w int) int {
	p := v.decodePlan(h, w)
	return 5*4*p.act + colsBytes + 4*p.qChunk*h*w + 4*p.maxC
}

// Decode turns packed, denormalised latents z [h·w, z_dim] (NHWC) into the
// image [out_channels, 16h, 16w] in [-1, 1]. The mid block runs once at
// latent resolution; the up blocks run in bands (decodePlan), their
// full-size intermediates held in host memory between blocks.
func (v *MetalVAE) Decode(z *tensor.Tensor, h, w int) (*tensor.Tensor, error) {
	cfg := v.cfg
	if z.Numel() != h*w*cfg.ZDim {
		return nil, fmt.Errorf("qwenimage: latents %v, want [%d, %d]", z.Shape(), h*w, cfg.ZDim)
	}
	_, dims, up := v.dims()
	plan := v.decodePlan(h, w)
	rw := &vaeWeights{v: v}
	d := "decoder"
	post := rw.conv("post_quant_conv", cfg.ZDim, cfg.ZDim, 1)
	convIn := rw.conv(d+".conv_in", cfg.ZDim, dims[0], 3)
	mid0, mid1 := rw.res(d+".mid_block.resnets.0", dims[0], dims[0]), rw.res(d+".mid_block.resnets.1", dims[0], dims[0])
	midAttn := rw.attn(d+".mid_block.attentions.0", dims[0])
	type upBlock struct {
		res      []vaeRes
		resample *vaeConv
		idx      []uint32
	}
	var ups []upBlock
	for i, s := range plan.stages {
		var u upBlock
		cur := s.in
		for j := range cfg.NumResBlocks + 1 {
			u.res = append(u.res, rw.res(fmt.Sprintf("%s.up_blocks.%d.resnets.%d", d, i, j), cur, s.out))
			cur = s.out
		}
		if s.up {
			c := rw.conv(fmt.Sprintf("%s.up_blocks.%d.upsampler.resample.1", d, i), s.out, s.out, 3)
			u.resample = &c
			ft := 1
			if up[i] {
				ft = 2
			}
			repeats := s.out * ft * 4 / s.in
			for oc := range s.out {
				for hs := range 2 {
					for ws := range 2 {
						u.idx = append(u.idx, uint32(((oc*ft+ft-1)*2+hs)*2+ws)/uint32(repeats))
					}
				}
			}
		}
		ups = append(ups, u)
	}
	normOut := rw.vec(d + ".norm_out.gamma")
	convOut := rw.conv(d+".conv_out", dims[len(dims)-1], cfg.OutChannels, 3)
	if rw.err != nil {
		return nil, rw.err
	}
	r, free, err := v.buffers(plan.act, plan.qChunk*h*w, plan.maxC)
	if err != nil {
		return nil, err
	}
	defer free()
	var idxBufs []*metal.Buffer
	defer func() {
		for _, b := range idxBufs {
			b.Release()
		}
	}()
	idxFor := make([]*metal.Buffer, len(ups))
	for i, u := range ups {
		if u.idx == nil {
			continue
		}
		b, err := v.dev.NewBuffer(4 * len(u.idx))
		if err != nil {
			return nil, err
		}
		idxBufs = append(idxBufs, b)
		copy(unsafe.Slice((*uint32)(unsafe.Pointer(&b.Bytes()[0])), len(u.idx)), u.idx)
		idxFor[i] = b
	}

	// Mid block at latent resolution.
	copy(f32view(r.x), z.F32())
	err = v.dev.Run(func(e *metal.Encoder) {
		r.e = e
		r.conv(r.x, r.a, h, w, post)
		r.conv(r.a, r.x, h, w, convIn)
		r.res(h, w, mid0)
		r.attn(h, w, midAttn, plan.qChunk)
		r.res(h, w, mid1)
	})
	if err != nil {
		return nil, err
	}
	cur := make([]float32, h*w*dims[0])
	copy(cur, f32view(r.x))

	// Up blocks, band by band.
	Co := cfg.OutChannels
	Ho, Wo := 16*h, 16*w
	img := tensor.New(tensor.F32, Co, Ho, Wo)
	for i, s := range plan.stages {
		u := ups[i]
		var next []float32
		if s.up {
			next = make([]float32, 4*s.H*s.W*s.out)
		}
		for r0 := 0; r0 < s.H; r0 += s.rows {
			r1 := min(s.H, r0+s.rows)
			a0, a1 := max(0, r0-s.halo), min(s.H, r1+s.halo)
			bh, W := a1-a0, s.W
			copy(f32view(r.x), cur[a0*W*s.in:a1*W*s.in])
			err = v.dev.Run(func(e *metal.Encoder) {
				r.e = e
				if s.up {
					e.CopyF32(r.x.At(0), r.bin.At(0), bh*W*s.in) // block input, for the pixel-shuffle shortcut
				}
				for _, rb := range u.res {
					r.res(bh, W, rb)
				}
				if s.up {
					e.Upsample2x(r.x.At(0), r.a.At(0), bh, W, s.out)
					r.conv(r.a, r.b2, 2*bh, 2*W, *u.resample)
					e.DepthToSpace2Map(r.bin.At(0), r.x.At(0), idxFor[i].At(0), bh, W, s.in, s.out)
					e.GatedAdd(r.x.At(0), r.one.At(0), r.b2.At(0), 4*bh*W, s.out, s.out, s.out)
					return
				}
				e.RMSNormRows(r.x.At(0), r.a.At(0), normOut.At(0), bh*W, s.out, s.out, s.out, vaeEps)
				e.SiLU(r.a.At(0), bh*W*s.out)
				r.conv(r.a, r.b2, bh, W, convOut)
			})
			if err != nil {
				return nil, err
			}
			if s.up { // keep output rows [2·r0, 2·r1)
				row := 2 * W * s.out
				copy(next[2*r0*row:2*r1*row], f32view(r.x)[2*(r0-a0)*row:2*(r1-a0)*row])
				continue
			}
			// NHWC → [C, H, W], clamped.
			src, dst := f32view(r.b2)[(r0-a0)*W*Co:], img.F32()
			for p := range (r1 - r0) * W {
				o := r0*W + p
				for c := range Co {
					dst[c*Ho*Wo+o] = min(max(src[p*Co+c], -1), 1)
				}
			}
		}
		cur = next
	}
	return img, nil
}

// encodePlan is Encode's floats per activation buffer and attention query
// chunk for an H0×W0 image. (Encode runs whole-image: condition images are
// about Resolution² pixels, and its peak is at 96 channels, not the
// decoder's 288.)
func (v *MetalVAE) encodePlan(H0, W0 int) (act, qc int) {
	dims, _, _ := v.dims()
	last := len(v.cfg.DimMult) - 1
	act = H0 * W0 * max(v.cfg.InChannels, dims[0])
	for i := range last + 1 {
		act = max(act, (H0>>i)*(W0>>i)*max(dims[i], dims[i+1]))
	}
	P := (H0 >> last) * (W0 >> last)
	return act, min(P, max(1, attnBytes/(4*P)))
}

// EncodeBytes is the GPU memory Encode needs for an H0×W0 image beyond the
// mapped weights.
func (v *MetalVAE) EncodeBytes(H0, W0 int) int {
	act, qc := v.encodePlan(H0, W0)
	last := len(v.cfg.DimMult) - 1
	C := v.cfg.BaseDim * v.cfg.DimMult[last]
	return 5*4*act + colsBytes + 4*qc*(H0>>last)*(W0>>last) + 4*max(3*C, v.cfg.BaseDim)
}

// Encode turns an image [in_channels, H, W] in [-1, 1] into normalised,
// packed condition latents [(H/16)·(W/16), z_dim] (the latent mode, as
// PackLatents(BuildVAEEncoder(...))).
func (v *MetalVAE) Encode(px *tensor.Tensor) (*tensor.Tensor, error) {
	cfg := v.cfg
	s := px.Shape()
	C0, H0, W0 := s[0], s[1], s[2]
	f := 1 << (len(cfg.DimMult) - 1)
	if C0 != cfg.InChannels || H0%f != 0 || W0%f != 0 {
		return nil, fmt.Errorf("qwenimage: VAE encoder input %v, want [%d, H, W] with H, W multiples of %d", s, cfg.InChannels, f)
	}
	dims, _, _ := v.dims()
	last := len(cfg.DimMult) - 1
	maxAct, qc := v.encodePlan(H0, W0)
	rw := &vaeWeights{v: v}
	e := "encoder"
	convIn := rw.conv(e+".conv_in", C0, dims[0], 3)
	type downBlock struct {
		res     []vaeRes
		down    *vaeConv
		in, out int
		ft      int
	}
	var downs []downBlock
	for i := range last + 1 {
		in, out := dims[i], dims[i+1]
		b := downBlock{in: in, out: out, ft: 1}
		cur := in
		for j := range cfg.NumResBlocks {
			b.res = append(b.res, rw.res(fmt.Sprintf("%s.down_blocks.%d.resnets.%d", e, i, j), cur, out))
			cur = out
		}
		if i != last {
			c := rw.conv(fmt.Sprintf("%s.down_blocks.%d.downsampler.resample.1", e, i), out, out, 3)
			b.down = &c
			if cfg.TemperalDownsample[i] {
				b.ft = 2
			}
		}
		downs = append(downs, b)
	}
	C := dims[len(dims)-1]
	mid0, mid1 := rw.res(e+".mid_block.resnets.0", C, C), rw.res(e+".mid_block.resnets.1", C, C)
	midAttn := rw.attn(e+".mid_block.attentions.0", C)
	normOut := rw.vec(e + ".norm_out.gamma")
	convOut := rw.conv(e+".conv_out", C, 2*cfg.ZDim, 3)
	quant := rw.conv("quant_conv", 2*cfg.ZDim, 2*cfg.ZDim, 1)
	if rw.err != nil {
		return nil, rw.err
	}
	hl, wl := H0/f, W0/f
	r, free, err := v.buffers(maxAct, qc*hl*wl, max(3*C, dims[0]))
	if err != nil {
		return nil, err
	}
	defer free()
	xs := f32view(r.x)
	for c := range C0 { // [C, H, W] → NHWC
		for p := range H0 * W0 {
			xs[p*C0+c] = px.F32()[c*H0*W0+p]
		}
	}
	err = v.dev.Run(func(e *metal.Encoder) {
		r.e = e
		H, W := H0, W0
		r.conv(r.x, r.a, H, W, convIn)
		r.x, r.a = r.a, r.x
		for _, b := range downs {
			e.CopyF32(r.x.At(0), r.bin.At(0), H*W*b.in) // block input, for the shortcut
			for _, rb := range b.res {
				r.res(H, W, rb)
			}
			if b.down == nil { // last block: identity shortcut (in == out)
				e.GatedAdd(r.x.At(0), r.one.At(0), r.bin.At(0), H*W, b.out, b.out, b.out)
				continue
			}
			r.convDown(r.x, r.a, H, W, *b.down)
			e.AvgDownS2D(r.bin.At(0), r.b2.At(0), H/2, W/2, b.in, b.out, b.ft)
			e.GatedAdd(r.a.At(0), r.one.At(0), r.b2.At(0), (H/2)*(W/2), b.out, b.out, b.out)
			r.x, r.a = r.a, r.x
			H, W = H/2, W/2
		}
		r.res(H, W, mid0)
		r.attn(H, W, midAttn, qc)
		r.res(H, W, mid1)
		e.RMSNormRows(r.x.At(0), r.a.At(0), normOut.At(0), H*W, C, C, C, vaeEps)
		e.SiLU(r.a.At(0), H*W*C)
		r.conv(r.a, r.b2, H, W, convOut)
		r.conv(r.b2, r.a, H, W, quant) // [h·w, 2·z]: the mean is the first z channels
	})
	if err != nil {
		return nil, err
	}
	Z := cfg.ZDim
	out := tensor.New(tensor.F32, hl*wl, Z)
	src := f32view(r.a)
	for p := range hl * wl {
		for c := range Z {
			out.F32()[p*Z+c] = (src[p*2*Z+c] - cfg.LatentsMean[c]) / cfg.LatentsStd[c]
		}
	}
	return out, nil
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
