package ops

import (
	"math"
	"sync"
	"sync/atomic"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/vek"
	"github.com/giraffesyo/ingot/tensor"
)

const ingotDomainNorm = "ingot"

// layerNormOp: normalise over dims [axis, rank) with per-element scale/bias.
type layerNormOp struct {
	n    NodeInfo
	axis int
	eps  float32
}

func (o *layerNormOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 1 || in[0] == nil {
		return nil, o.n.Errorf("need X")
	}
	x := in[0]
	var scale *tensor.Tensor
	if len(in) > 1 && in[1] != nil {
		scale = in[1]
	}
	if x.DType() != tensor.F32 || (scale != nil && scale.DType() != tensor.F32) {
		sd := "nil"
		if scale != nil {
			sd = scale.DType().String()
		}
		return nil, o.n.Errorf("LayerNorm: want f32 inputs, got x=%s scale=%s", x.DType(), sd)
	}
	var bias []float32
	if len(in) > 2 && in[2] != nil {
		if in[2].DType() != tensor.F32 {
			return nil, o.n.Errorf("LayerNorm: want f32 bias, got %s", in[2].DType())
		}
		bias = in[2].F32()
	}
	xs := x.Shape()
	axis := o.axis
	if axis < 0 {
		axis += len(xs)
	}
	if axis < 0 || axis >= len(xs) {
		return nil, o.n.Errorf("axis %d out of range for %v", o.axis, xs)
	}
	outer, D := 1, 1
	for _, d := range xs[:axis] {
		outer *= d
	}
	for _, d := range xs[axis:] {
		D *= d
	}
	var sf []float32
	if scale != nil {
		sf = scale.F32()
		if len(sf) != D {
			return nil, o.n.Errorf("scale numel %d != normalised size %d", len(sf), D)
		}
	}
	out := ctx.NewUninit(tensor.F32, xs...)
	xf, of := x.F32(), out.F32()
	grain := 2
	if outer*D <= 2*unaryChunk {
		grain = outer // tiny: stay on the caller
	}
	par.For(outer, grain, func(i, _ int) {
		layerNormRow(of[i*D:(i+1)*D], xf[i*D:(i+1)*D], sf, bias, o.eps)
	})
	// Optional Mean / InvStdDev outputs are rarely consumed; nil unless asked.
	return ctx.OutPad(o.n.NumOut, out), nil
}

// layerNormRow normalises one row into dst: dst = (x − mean)·inv·scale + bias.
// Built from vek passes over the (L1-resident) row: the mean from a dot with
// ones, the centred row written into dst, the variance as its self-dot, then
// the affine tail in place — ~4× the scalar float64 loop at D=384 while the
// two-pass variance keeps the scalar version's numerical behaviour.
func layerNormRow(dst, x, scale, bias []float32, eps float32) {
	D := len(x)
	if D < layerNormVekMin {
		layerNormRowScalar(dst, x, scale, bias, eps)
		return
	}
	ones := onesVec(D)
	mean := vek.Dot(x, ones) / float32(D)
	vek.AddScalar(dst, x, -mean)
	vs := vek.Dot(dst, dst) / float32(D)
	inv := float32(1 / math.Sqrt(float64(vs)+float64(eps)))
	vek.MulScalar(dst, dst, inv)
	if scale != nil {
		vek.Mul(dst, dst, scale)
	}
	if bias != nil {
		vek.Add(dst, dst, bias)
	}
}

// layerNormVekMin is the row length from which the vek composition beats the
// scalar loop (six kernel calls per row; at D=48 they cost more than the 48
// scalar iterations — bertish measured +7.5% on Zen 5 with vek at D=48).
const layerNormVekMin = 128

// layerNormRowScalar is the float64-accumulating reference form, kept as the
// short-row path.
func layerNormRowScalar(dst, x, scale, bias []float32, eps float32) {
	D := len(x)
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(D)
	var vs float64
	for _, v := range x {
		d := float64(v) - mean
		vs += d * d
	}
	inv := float32(1 / math.Sqrt(vs/float64(D)+float64(eps)))
	m := float32(mean)
	switch {
	case scale == nil:
		for j, v := range x {
			dst[j] = (v - m) * inv
		}
	case bias != nil:
		for j, v := range x {
			dst[j] = (v-m)*inv*scale[j] + bias[j]
		}
	default:
		for j, v := range x {
			dst[j] = (v - m) * inv * scale[j]
		}
	}
}

// onesVec returns a shared all-ones vector of at least n floats (the mean
// reduction runs as a dot product; the vector is read-only and grows
// monotonically).
func onesVec(n int) []float32 {
	if p := onesBuf.Load(); p != nil && len(*p) >= n {
		return (*p)[:n]
	}
	onesMu.Lock()
	defer onesMu.Unlock()
	if p := onesBuf.Load(); p != nil && len(*p) >= n {
		return (*p)[:n]
	}
	buf := make([]float32, max(n, 1024))
	for i := range buf {
		buf[i] = 1
	}
	onesBuf.Store(&buf)
	return buf[:n]
}

var (
	onesMu  sync.Mutex
	onesBuf atomic.Pointer[[]float32]
)

// addLayerNormOp is the optimizer's fusion of a residual Add feeding a
// LayerNorm (the pre-norm transformer block): inputs a, b (same shape),
// scale, bias; outputs the sum (still needed as the next residual) and the
// normalised sum. One pass per row: the sum is written and normalised while
// it sits in L1, instead of a full Add pass followed by a re-read.
type addLayerNormOp struct {
	n    NodeInfo
	axis int
	eps  float32
}

func (o *addLayerNormOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("AddLayerNorm: need a, b")
	}
	a, b := in[0], in[1]
	if a.DType() != tensor.F32 || b.DType() != tensor.F32 || !a.Shape().Equal(b.Shape()) {
		return nil, o.n.Errorf("AddLayerNorm: want equal-shape f32 a, b, got %s%v %s%v", a.DType(), a.Shape(), b.DType(), b.Shape())
	}
	var sf, bias []float32
	if len(in) > 2 && in[2] != nil {
		if in[2].DType() != tensor.F32 {
			return nil, o.n.Errorf("AddLayerNorm: want f32 scale")
		}
		sf = in[2].F32()
	}
	if len(in) > 3 && in[3] != nil {
		if in[3].DType() != tensor.F32 {
			return nil, o.n.Errorf("AddLayerNorm: want f32 bias")
		}
		bias = in[3].F32()
	}
	xs := a.Shape()
	axis := o.axis
	if axis < 0 {
		axis += len(xs)
	}
	if axis < 0 || axis >= len(xs) {
		return nil, o.n.Errorf("axis %d out of range for %v", o.axis, xs)
	}
	outer, D := 1, 1
	for _, d := range xs[:axis] {
		outer *= d
	}
	for _, d := range xs[axis:] {
		D *= d
	}
	if sf != nil && len(sf) != D {
		return nil, o.n.Errorf("scale numel %d != normalised size %d", len(sf), D)
	}
	sum := ctx.NewUninit(tensor.F32, xs...)
	out := ctx.NewUninit(tensor.F32, xs...)
	af, bf, sm, of := a.F32(), b.F32(), sum.F32(), out.F32()
	grain := 2
	if outer*D <= 2*unaryChunk {
		grain = outer
	}
	par.For(outer, grain, func(i, _ int) {
		r := sm[i*D : (i+1)*D]
		vek.Add(r, af[i*D:(i+1)*D], bf[i*D:(i+1)*D])
		layerNormRow(of[i*D:(i+1)*D], r, sf, bias, o.eps)
	})
	return ctx.Out(sum, out), nil
}

// batchNormOp (inference): per-channel (axis 1) affine.
type batchNormOp struct {
	n   NodeInfo
	eps float32
}

func (o *batchNormOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 5 {
		return nil, o.n.Errorf("need X, scale, B, mean, var")
	}
	for i := 0; i < 5; i++ {
		if in[i] == nil {
			return nil, o.n.Errorf("input %d missing", i)
		}
	}
	x := in[0]
	xs := x.Shape()
	if len(xs) < 2 {
		return nil, o.n.Errorf("need rank>=2")
	}
	N, C := xs[0], xs[1]
	P := 1
	for _, d := range xs[2:] {
		P *= d
	}
	sc, b, mean, vr := in[1].F32(), in[2].F32(), in[3].F32(), in[4].F32()
	out := ctx.New(tensor.F32, xs...)
	xf, of := x.F32(), out.F32()
	par.For(N*C, 1, func(nc, _ int) {
		c := nc % C
		a := sc[c] / float32(math.Sqrt(float64(vr[c])+float64(o.eps)))
		bb := b[c] - mean[c]*a
		src := xf[nc*P : (nc+1)*P]
		dst := of[nc*P : (nc+1)*P]
		for i, v := range src {
			dst[i] = v*a + bb
		}
	})
	return ctx.OutPad(o.n.NumOut, out), nil
}

// instanceNormOp: per (n, c) normalisation over spatial dims.
type instanceNormOp struct {
	n   NodeInfo
	eps float32
}

func (o *instanceNormOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 3 || in[0] == nil || in[1] == nil || in[2] == nil {
		return nil, o.n.Errorf("need X, scale, B")
	}
	x := in[0]
	xs := x.Shape()
	N, C := xs[0], xs[1]
	P := 1
	for _, d := range xs[2:] {
		P *= d
	}
	sc, b := in[1].F32(), in[2].F32()
	out := ctx.New(tensor.F32, xs...)
	xf, of := x.F32(), out.F32()
	par.For(N*C, 1, func(nc, _ int) {
		c := nc % C
		src := xf[nc*P : (nc+1)*P]
		dst := of[nc*P : (nc+1)*P]
		var mean float64
		for _, v := range src {
			mean += float64(v)
		}
		mean /= float64(P)
		var vs float64
		for _, v := range src {
			d := float64(v) - mean
			vs += d * d
		}
		a := sc[c] / float32(math.Sqrt(vs/float64(P)+float64(o.eps)))
		bb := b[c] - float32(mean)*a
		for i, v := range src {
			dst[i] = v*a + bb
		}
	})
	return ctx.Out(out), nil
}

func init() {
	Register(ingotDomainNorm, "LayerNorm", 1, func(n NodeInfo) (Op, error) {
		return &layerNormOp{n: n, axis: int(n.Attrs.Int("axis", -1)), eps: n.Attrs.Float("epsilon", 1e-5)}, nil
	})
	Register(ingotDomainNorm, "AddLayerNorm", 1, func(n NodeInfo) (Op, error) {
		return &addLayerNormOp{n: n, axis: int(n.Attrs.Int("axis", -1)), eps: n.Attrs.Float("epsilon", 1e-5)}, nil
	})
	Register("", "LayerNormalization", 17, func(n NodeInfo) (Op, error) {
		return &layerNormOp{n: n, axis: int(n.Attrs.Int("axis", -1)), eps: n.Attrs.Float("epsilon", 1e-5)}, nil
	})
	Register("", "BatchNormalization", 9, func(n NodeInfo) (Op, error) {
		return &batchNormOp{n: n, eps: n.Attrs.Float("epsilon", 1e-5)}, nil
	})
	Register("", "InstanceNormalization", 6, func(n NodeInfo) (Op, error) {
		return &instanceNormOp{n: n, eps: n.Attrs.Float("epsilon", 1e-5)}, nil
	})
}
