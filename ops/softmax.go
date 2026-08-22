package ops

import (
	"math"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/tensor"
)

type softmaxOp struct {
	n      NodeInfo
	axis   int
	log    bool
	legacy bool // opset < 13: coerce to 2-D [prod(<axis), prod(>=axis)]
}

func (o *softmaxOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 1 || in[0] == nil || in[0].DType() != tensor.F32 {
		return nil, o.n.Errorf("need f32 input")
	}
	x := in[0]
	xs := x.Shape()
	axis := o.axis
	if axis < 0 {
		axis += len(xs)
	}
	if axis < 0 || axis >= len(xs) {
		return nil, o.n.Errorf("axis %d out of range for %v", o.axis, xs)
	}
	out := ctx.New(tensor.F32, xs...)
	xf, of := x.F32(), out.F32()
	var outer, inner, D int
	if o.legacy {
		outer, D, inner = 1, 1, 1
		for _, d := range xs[:axis] {
			outer *= d
		}
		for _, d := range xs[axis:] {
			D *= d
		}
	} else {
		outer, inner = 1, 1
		for _, d := range xs[:axis] {
			outer *= d
		}
		D = xs[axis]
		for _, d := range xs[axis+1:] {
			inner *= d
		}
	}
	if inner == 1 {
		par.For(outer, 4, func(i, _ int) {
			softmaxRow(xf[i*D:(i+1)*D], of[i*D:(i+1)*D], o.log)
		})
		return []*tensor.Tensor{out}, nil
	}
	// Strided axis: gather into a temp row per (outer, inner).
	par.For(outer, 1, func(i, _ int) {
		tmp := make([]float32, D)
		res := make([]float32, D)
		for j := 0; j < inner; j++ {
			base := i*D*inner + j
			for d := 0; d < D; d++ {
				tmp[d] = xf[base+d*inner]
			}
			softmaxRow(tmp, res, o.log)
			for d := 0; d < D; d++ {
				of[base+d*inner] = res[d]
			}
		}
	})
	return []*tensor.Tensor{out}, nil
}

func softmaxRow(x, y []float32, logsm bool) {
	m := float32(math.Inf(-1))
	for _, v := range x {
		if v > m {
			m = v
		}
	}
	var sum float32
	for i, v := range x {
		e := float32(math.Exp(float64(v - m)))
		y[i] = e
		sum += e
	}
	if logsm {
		ls := float32(math.Log(float64(sum)))
		for i, v := range x {
			y[i] = v - m - ls
		}
		return
	}
	inv := 1 / sum
	for i := range y {
		y[i] *= inv
	}
}

func init() {
	for _, name := range []string{"Softmax", "LogSoftmax"} {
		isLog := name == "LogSoftmax"
		Register("", name, 13, func(n NodeInfo) (Op, error) {
			return &softmaxOp{n: n, axis: int(n.Attrs.Int("axis", -1)), log: isLog}, nil
		})
		Register("", name, 1, func(n NodeInfo) (Op, error) {
			return &softmaxOp{n: n, axis: int(n.Attrs.Int("axis", 1)), log: isLog, legacy: true}, nil
		})
	}
}
