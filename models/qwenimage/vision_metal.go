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

// MetalVision runs the Qwen3-VL vision tower on the GPU with its bf16
// weights read in place (same math as BuildVision): one command buffer per
// image.
type MetalVision struct {
	dev    *metal.Device
	cfg    VisionConfig
	set    *safetensors.Set
	shards map[unsafe.Pointer]*metal.Buffer
	vecs   map[string]*metal.Buffer
}

// NewMetalVision maps the vision tower's weights onto the GPU.
func NewMetalVision(cfg VisionConfig, set *safetensors.Set) (*MetalVision, error) {
	dev, err := metal.Open()
	if err != nil {
		return nil, err
	}
	if err := dev.PrepareConv(); err != nil {
		return nil, err
	}
	return &MetalVision{dev: dev, cfg: cfg, set: set, shards: map[unsafe.Pointer]*metal.Buffer{}, vecs: map[string]*metal.Buffer{}}, nil
}

func (v *MetalVision) vec(name string) (*metal.Buffer, error) {
	if b, ok := v.vecs[name]; ok {
		return b, nil
	}
	b, err := f32Buffer(v.dev, v.set, visionPrefix+name)
	if err == nil {
		v.vecs[name] = b
	}
	return b, err
}

// Encode runs one image's patches [gh·gw, 1536]: the merged tokens and the
// deepstack features, [gh·gw/4, out_hidden] each.
func (v *MetalVision) Encode(pixels *tensor.Tensor, gh, gw int) (merged *tensor.Tensor, deep []*tensor.Tensor, err error) {
	c := v.cfg
	N, D, H, I, O := gh*gw, c.HiddenSize, c.NumHeads, c.IntermediateSize, c.OutHiddenSize
	dh, M := D/H, visMerge*visMerge
	if pixels.Numel() != N*visPatchDim {
		return nil, nil, fmt.Errorf("qwenimage: vision pixels %v, want [%d %d]", pixels.Shape(), N, visPatchDim)
	}
	// Resolve weights first (Run callbacks cannot call into Device).
	wt := func(name string, out, in int) metal.Region {
		if err != nil {
			return metal.Region{}
		}
		var r metal.Region
		mem, off, info, e := v.set.Locate(visionPrefix + name)
		if e != nil {
			err = e
			return r
		}
		n := 1
		for _, d := range info.Shape {
			n *= d
		}
		if info.DType != "BF16" || n != out*in {
			err = fmt.Errorf("qwenimage: %s is %s%v, want BF16 %d×%d", name, info.DType, info.Shape, out, in)
			return r
		}
		key := unsafe.Pointer(&mem[0])
		b, ok := v.shards[key]
		if !ok {
			if b, err = v.dev.Wrap(mem); err != nil {
				return r
			}
			v.shards[key] = b
		}
		return b.At(off)
	}
	vc := func(name string) *metal.Buffer {
		if err != nil {
			return nil
		}
		var b *metal.Buffer
		b, err = v.vec(name)
		return b
	}
	type blk struct {
		n1w, n1b, n2w, n2b, qkvB, projB, fc1B, fc2B *metal.Buffer
		qkv, proj, fc1, fc2                         metal.Region
	}
	type merg struct {
		nw, nb, b1, b2 *metal.Buffer
		w1, w2         metal.Region
	}
	getMerger := func(p string) merg {
		return merg{nw: vc(p + ".norm.weight"), nb: vc(p + ".norm.bias"), b1: vc(p + ".linear_fc1.bias"),
			b2: vc(p + ".linear_fc2.bias"), w1: wt(p+".linear_fc1.weight", M*D, M*D), w2: wt(p+".linear_fc2.weight", O, M*D)}
	}
	patchW, patchB := wt("patch_embed.proj.weight", D, visPatchDim), vc("patch_embed.proj.bias")
	var blocks []blk
	for i := range c.Depth {
		p := fmt.Sprintf("blocks.%d.", i)
		blocks = append(blocks, blk{
			n1w: vc(p + "norm1.weight"), n1b: vc(p + "norm1.bias"), n2w: vc(p + "norm2.weight"), n2b: vc(p + "norm2.bias"),
			qkvB: vc(p + "attn.qkv.bias"), projB: vc(p + "attn.proj.bias"), fc1B: vc(p + "mlp.linear_fc1.bias"),
			fc2B: vc(p + "mlp.linear_fc2.bias"), qkv: wt(p+"attn.qkv.weight", 3*D, D), proj: wt(p+"attn.proj.weight", D, D),
			fc1: wt(p+"mlp.linear_fc1.weight", I, D), fc2: wt(p+"mlp.linear_fc2.weight", D, I)})
	}
	var deepM []merg
	for j := range c.DeepstackVisualIndexes {
		deepM = append(deepM, getMerger(fmt.Sprintf("deepstack_merger_list.%d", j)))
	}
	final := getMerger("merger")
	if err != nil {
		return nil, nil, err
	}
	posTable, err := v.set.F32(visionPrefix + "pos_embed.weight")
	if err != nil {
		return nil, nil, err
	}

	var bufs []*metal.Buffer
	defer func() {
		for _, b := range bufs {
			b.Release()
		}
	}()
	nb := func(floats int) *metal.Buffer {
		b, e := v.dev.NewBuffer(4 * max(floats, 1))
		if e != nil && err == nil {
			err = e
		}
		if b != nil {
			bufs = append(bufs, b)
		}
		return b
	}
	px, x, y, qkv, o, h := nb(N*visPatchDim), nb(N*D), nb(N*M*D), nb(N*3*D), nb(N*D), nb(N*max(I, M*D))
	s, pos, ones, cos, sin := nb(N*N), nb(N*D), nb(M*D), nb(N*dh/2), nb(N*dh/2)
	out := nb((N / M) * O)
	var deepOut []*metal.Buffer
	for range c.DeepstackVisualIndexes {
		deepOut = append(deepOut, nb((N/M)*O))
	}
	if err != nil {
		return nil, nil, err
	}
	copy(f32view(px), pixels.F32())
	copy(f32view(pos), visionPosEmbed(c, posTable, gh, gw).F32())
	for i := range f32view(ones) {
		f32view(ones)[i] = 1
	}
	fc, fs := visionRope(c, gh, gw) // [N, 1, dh]; the kernel takes the first half
	for p := range N {
		copy(f32view(cos)[p*dh/2:(p+1)*dh/2], fc.F32()[p*dh:p*dh+dh/2])
		copy(f32view(sin)[p*dh/2:(p+1)*dh/2], fs.F32()[p*dh:p*dh+dh/2])
	}
	deepAt := map[int]int{}
	for j, l := range c.DeepstackVisualIndexes {
		deepAt[l] = j
	}
	scale := float32(1 / math.Sqrt(float64(dh)))
	err = v.dev.Run(func(e *metal.Encoder) {
		lin := func(a *metal.Buffer, w metal.Region, bias, dst *metal.Buffer, rows, n, k int) {
			e.Gemm(metal.Gemm{M: rows, N: n, K: k, A: a.At(0), B: w, C: dst.At(0), TransB: true, BF16: true})
			e.AddBias(dst.At(0), bias.At(0), rows, n, n)
		}
		ln := func(src, dst, w, b *metal.Buffer, rows, cols int) {
			e.LayerNormMod(src.At(0), dst.At(0), w.At(0), rows, cols, cols, cols, 1e-6)
			e.AddBias(dst.At(0), b.At(0), rows, cols, cols)
		}
		runMerger := func(m merg, post bool, dst *metal.Buffer) {
			if post { // x viewed as [N/4, 4D], then norm
				ln(x, y, m.nw, m.nb, N/M, M*D)
			} else { // norm over D, then viewed as [N/4, 4D]
				ln(x, y, m.nw, m.nb, N, D)
			}
			lin(y, m.w1, m.b1, h, N/M, M*D, M*D)
			e.GELU(h.At(0), (N/M)*M*D, false)
			lin(h, m.w2, m.b2, dst, N/M, O, M*D)
		}
		lin(px, patchW, patchB, x, N, D, visPatchDim)
		e.GatedAdd(x.At(0), ones.At(0), pos.At(0), N, D, D, D)
		for li, b := range blocks {
			ln(x, y, b.n1w, b.n1b, N, D)
			lin(y, b.qkv, b.qkvB, qkv, N, 3*D, D)
			for _, off := range []int{0, D} { // q, k: rotate_half RoPE (no norm)
				e.RMSNormRoPEMode(qkv.At(4*off), cos.At(0), cos.At(0), sin.At(0), N, H, dh, 3*D, 0,
					metal.RopeHalf|metal.RopeNoNorm)
			}
			for hh := range H {
				ho := 4 * hh * dh
				e.Gemm(metal.Gemm{M: N, N: N, K: dh, A: qkv.At(ho), LDA: 3 * D, B: qkv.At(4*D + ho), LDB: 3 * D, C: s.At(0), TransB: true})
				e.SoftmaxRows(s.At(0), N, N, N, scale)
				e.Gemm(metal.Gemm{M: N, N: dh, K: N, A: s.At(0), B: qkv.At(4*2*D + ho), LDB: 3 * D, C: o.At(ho), LDC: D})
			}
			lin(o, b.proj, b.projB, y, N, D, D)
			e.GatedAdd(x.At(0), ones.At(0), y.At(0), N, D, D, D)
			ln(x, y, b.n2w, b.n2b, N, D)
			lin(y, b.fc1, b.fc1B, h, N, I, D)
			e.GELU(h.At(0), N*I, true)
			lin(h, b.fc2, b.fc2B, y, N, D, I)
			e.GatedAdd(x.At(0), ones.At(0), y.At(0), N, D, D, D)
			if j, ok := deepAt[li]; ok {
				runMerger(deepM[j], true, deepOut[j])
			}
		}
		runMerger(final, false, out)
	})
	if err != nil {
		return nil, nil, err
	}
	get := func(b *metal.Buffer) *tensor.Tensor {
		t := tensor.New(tensor.F32, N/M, O)
		copy(t.F32(), f32view(b))
		return t
	}
	merged = get(out)
	for _, b := range deepOut {
		deep = append(deep, get(b))
	}
	return merged, deep, nil
}

// Close releases the GPU buffers (the mapped weights stay the caller's).
func (v *MetalVision) Close() {
	for _, b := range v.vecs {
		b.Release()
	}
	for _, b := range v.shards {
		b.Release()
	}
}
