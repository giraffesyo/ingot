package ops

import (
	"math"

	"github.com/giraffesyo/ingot/kernels/par"
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
	out := ctx.New(tensor.F32, xs...)
	xf, of := x.F32(), out.F32()
	eps := o.eps
	grain := 2
	if outer*D <= 2*unaryChunk {
		grain = outer // tiny: stay on the caller
	}
	par.For(outer, grain, func(i, _ int) {
		row := xf[i*D : (i+1)*D]
		dst := of[i*D : (i+1)*D]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(D)
		var vs float64
		for _, v := range row {
			d := float64(v) - mean
			vs += d * d
		}
		inv := float32(1 / math.Sqrt(vs/float64(D)+float64(eps)))
		m := float32(mean)
		switch {
		case sf == nil:
			for j, v := range row {
				dst[j] = (v - m) * inv
			}
		case bias != nil:
			for j, v := range row {
				dst[j] = (v-m)*inv*sf[j] + bias[j]
			}
		default:
			for j, v := range row {
				dst[j] = (v - m) * inv * sf[j]
			}
		}
	})
	outs := []*tensor.Tensor{out}
	// Optional Mean / InvStdDev outputs are rarely consumed; provide if asked.
	for k := 1; k < o.n.NumOut; k++ {
		outs = append(outs, nil)
	}
	return outs, nil
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
	outs := []*tensor.Tensor{out}
	for k := 1; k < o.n.NumOut; k++ {
		outs = append(outs, nil)
	}
	return outs, nil
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
	return []*tensor.Tensor{out}, nil
}

func init() {
	Register(ingotDomainNorm, "LayerNorm", 1, func(n NodeInfo) (Op, error) {
		return &layerNormOp{n: n, axis: int(n.Attrs.Int("axis", -1)), eps: n.Attrs.Float("epsilon", 1e-5)}, nil
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
