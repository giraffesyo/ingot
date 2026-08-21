package ops

import (
	"math"

	"github.com/giraffesyo/ocr/tensor"
)

type reduceOp struct {
	n         NodeInfo
	kind      string
	attrAxes  []int64 // opset < 18
	keepdims  bool
	noopEmpty bool
}

func (o *reduceOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	if x.DType() != tensor.F32 {
		return nil, o.n.Errorf("only f32 (got %s)", x.DType())
	}
	xs := x.Shape()
	r := len(xs)
	var axes []int64
	switch {
	case o.attrAxes != nil:
		axes = o.attrAxes
	case len(in) > 1 && in[1] != nil:
		axes = asI64(in[1])
	}
	red := make([]bool, r)
	if len(axes) == 0 {
		if o.noopEmpty && o.attrAxes == nil {
			return []*tensor.Tensor{x.Clone()}, nil
		}
		for i := range red {
			red[i] = true
		}
	} else {
		for _, a := range axes {
			ax, err := normAxis(int(a), r)
			if err != nil {
				return nil, o.n.Errorf("%v", err)
			}
			red[ax] = true
		}
	}
	oshape := make(tensor.Shape, 0, r)
	for i, d := range xs {
		if red[i] {
			if o.keepdims {
				oshape = append(oshape, 1)
			}
		} else {
			oshape = append(oshape, d)
		}
	}
	// Output index strides: for each input dim, stride into the (squeezed) output.
	kept := make(tensor.Shape, 0, r)
	for i, d := range xs {
		if !red[i] {
			kept = append(kept, d)
		}
	}
	kst := kept.Strides()
	ostr := make([]int, r)
	ki := 0
	for i := range xs {
		if red[i] {
			ostr[i] = 0
		} else {
			ostr[i] = kst[ki]
			ki++
		}
	}
	out := ctx.New(tensor.F32, oshape...)
	of := out.F32()
	xf := x.F32()
	cnt := 1
	for i, d := range xs {
		if red[i] {
			cnt *= d
		}
	}
	init := float32(0)
	switch o.kind {
	case "max":
		init = float32(math.Inf(-1))
	case "min":
		init = float32(math.Inf(1))
	case "prod":
		init = 1
	}
	for i := range of {
		of[i] = init
	}
	// Fast path: reduce over a contiguous trailing block.
	trailing := true
	for i := 0; i < r; i++ {
		if red[i] {
			for j := i; j < r; j++ {
				if !red[j] {
					trailing = false
				}
			}
			break
		}
	}
	if trailing && len(of) > 0 {
		inner := cnt
		for i := range of {
			row := xf[i*inner : (i+1)*inner]
			of[i] = reduceRow(o.kind, of[i], row)
		}
	} else {
		idx := make([]int, r)
		oi := 0
		for i := 0; i < len(xf); i++ {
			of[oi] = accum(o.kind, of[oi], xf[i])
			for d := r - 1; d >= 0; d-- {
				idx[d]++
				oi += ostr[d]
				if idx[d] < xs[d] {
					break
				}
				oi -= ostr[d] * xs[d]
				idx[d] = 0
			}
		}
	}
	switch o.kind {
	case "mean":
		inv := 1 / float32(cnt)
		for i := range of {
			of[i] *= inv
		}
	case "l2":
		for i := range of {
			of[i] = float32(math.Sqrt(float64(of[i])))
		}
	}
	return []*tensor.Tensor{out}, nil
}

func accum(kind string, acc, v float32) float32 {
	switch kind {
	case "sum", "mean":
		return acc + v
	case "max":
		return max(acc, v)
	case "min":
		return min(acc, v)
	case "prod":
		return acc * v
	case "l2", "sumsq":
		return acc + v*v
	case "l1":
		return acc + float32(math.Abs(float64(v)))
	}
	return acc
}

func reduceRow(kind string, acc float32, row []float32) float32 {
	switch kind {
	case "sum", "mean":
		var s float32
		for _, v := range row {
			s += v
		}
		return acc + s
	case "max":
		for _, v := range row {
			if v > acc {
				acc = v
			}
		}
		return acc
	case "min":
		for _, v := range row {
			if v < acc {
				acc = v
			}
		}
		return acc
	}
	for _, v := range row {
		acc = accum(kind, acc, v)
	}
	return acc
}

// argOp: ArgMax / ArgMin → int64.
type argOp struct {
	n        NodeInfo
	axis     int
	keepdims bool
	last     bool
	isMax    bool
}

func (o *argOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	if x.DType() != tensor.F32 {
		return nil, o.n.Errorf("only f32")
	}
	xs := x.Shape()
	axis, err := normAxis(o.axis, len(xs))
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	outer, inner := 1, 1
	for _, d := range xs[:axis] {
		outer *= d
	}
	for _, d := range xs[axis+1:] {
		inner *= d
	}
	D := xs[axis]
	oshape := make(tensor.Shape, 0, len(xs))
	oshape = append(oshape, xs[:axis]...)
	if o.keepdims {
		oshape = append(oshape, 1)
	}
	oshape = append(oshape, xs[axis+1:]...)
	out := ctx.New(tensor.I64, oshape...)
	xf, of := x.F32(), out.I64()
	for a := 0; a < outer; a++ {
		for b := 0; b < inner; b++ {
			best := 0
			bv := xf[a*D*inner+b]
			for d := 1; d < D; d++ {
				v := xf[(a*D+d)*inner+b]
				better := v > bv
				if !o.isMax {
					better = v < bv
				}
				if better || (o.last && v == bv) {
					best, bv = d, v
				}
			}
			of[a*inner+b] = int64(best)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	kinds := map[string]string{
		"ReduceSum": "sum", "ReduceMean": "mean", "ReduceMax": "max", "ReduceMin": "min",
		"ReduceProd": "prod", "ReduceL2": "l2", "ReduceL1": "l1", "ReduceSumSquare": "sumsq",
	}
	for name, kind := range kinds {
		kind := kind
		// opset 18+: axes as input (ReduceSum since 13)
		since := 18
		if name == "ReduceSum" {
			since = 13
		}
		Register("", name, since, func(n NodeInfo) (Op, error) {
			return &reduceOp{n: n, kind: kind, keepdims: n.Attrs.Int("keepdims", 1) == 1, noopEmpty: n.Attrs.Int("noop_with_empty_axes", 0) == 1}, nil
		})
		Register("", name, 1, func(n NodeInfo) (Op, error) {
			axes := n.Attrs.Ints("axes", nil)
			if axes == nil {
				axes = []int64{} // non-nil, empty → reduce all
			}
			return &reduceOp{n: n, kind: kind, attrAxes: axes, keepdims: n.Attrs.Int("keepdims", 1) == 1}, nil
		})
	}
	for _, name := range []string{"ArgMax", "ArgMin"} {
		isMax := name == "ArgMax"
		Register("", name, 1, func(n NodeInfo) (Op, error) {
			return &argOp{n: n, axis: int(n.Attrs.Int("axis", 0)), keepdims: n.Attrs.Int("keepdims", 1) == 1,
				last: n.Attrs.Int("select_last_index", 0) == 1, isMax: isMax}, nil
		})
	}
}
