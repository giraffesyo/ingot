package ops

import "github.com/giraffesyo/ingot/tensor"

// cumSumOp: ONNX CumSum. Inputs: x, axis (int32/int64 scalar). Attrs:
// exclusive, reverse. f32 and int64 data.
type cumSumOp struct {
	n         NodeInfo
	exclusive bool
	reverse   bool
}

func (o *cumSumOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) != 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("CumSum: need x and axis")
	}
	x, at := in[0], in[1]
	if at.Numel() != 1 {
		return nil, o.n.Errorf("CumSum: axis must be a scalar")
	}
	var axis int
	switch at.DType() {
	case tensor.I64:
		axis = int(at.I64()[0])
	case tensor.I32:
		axis = int(at.I32()[0])
	default:
		return nil, o.n.Errorf("CumSum: axis dtype %s", at.DType())
	}
	xs := x.Shape()
	if axis < 0 {
		axis += len(xs)
	}
	if axis < 0 || axis >= len(xs) {
		return nil, o.n.Errorf("CumSum: axis out of range for %v", xs)
	}
	n := xs[axis]
	outer, inner := 1, 1
	for i := 0; i < axis; i++ {
		outer *= xs[i]
	}
	for i := axis + 1; i < len(xs); i++ {
		inner *= xs[i]
	}
	out := ctx.NewUninit(x.DType(), xs...)
	run := func(get func(i int) float64, set func(i int, v float64)) {
		for oi := 0; oi < outer; oi++ {
			for ii := 0; ii < inner; ii++ {
				base := oi*n*inner + ii
				idx := func(j int) int {
					if o.reverse {
						j = n - 1 - j
					}
					return base + j*inner
				}
				var acc float64
				for j := 0; j < n; j++ {
					if o.exclusive {
						set(idx(j), acc)
						acc += get(idx(j))
					} else {
						acc += get(idx(j))
						set(idx(j), acc)
					}
				}
			}
		}
	}
	switch x.DType() {
	case tensor.F32:
		xf, of := x.F32(), out.F32()
		// f32 accumulation (not f64): matches ORT bit-for-bit on the same
		// summation order.
		for oi := 0; oi < outer; oi++ {
			for ii := 0; ii < inner; ii++ {
				base := oi*n*inner + ii
				var acc float32
				for j := 0; j < n; j++ {
					k := j
					if o.reverse {
						k = n - 1 - j
					}
					p := base + k*inner
					if o.exclusive {
						of[p] = acc
						acc += xf[p]
					} else {
						acc += xf[p]
						of[p] = acc
					}
				}
			}
		}
	case tensor.I64:
		xi, oi64 := x.I64(), out.I64()
		run(func(i int) float64 { return float64(xi[i]) }, func(i int, v float64) { oi64[i] = int64(v) })
	default:
		return nil, o.n.Errorf("CumSum: dtype %s unsupported", x.DType())
	}
	return ctx.Out(out), nil
}

func init() {
	Register("", "CumSum", 11, func(n NodeInfo) (Op, error) {
		return &cumSumOp{
			n:         n,
			exclusive: n.Attrs.Int("exclusive", 0) != 0,
			reverse:   n.Attrs.Int("reverse", 0) != 0,
		}, nil
	})
}
