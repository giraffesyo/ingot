package ops

import (
	"github.com/giraffesyo/ingot/tensor"
)

// padOp implements Pad in constant, reflect, and edge modes over any rank.
type padOp struct {
	n        NodeInfo
	mode     string
	attrPads []int64 // opset < 11
	attrVal  float32
}

func (o *padOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	if x.DType() != tensor.F32 {
		return nil, o.n.Errorf("only f32")
	}
	xs := x.Shape()
	r := len(xs)
	var pads []int64
	if o.attrPads != nil {
		pads = o.attrPads
	} else if len(in) > 1 && in[1] != nil {
		pads = asI64(in[1])
	} else {
		return nil, o.n.Errorf("missing pads")
	}
	if len(pads) != 2*r {
		return nil, o.n.Errorf("pads len %d != 2*rank %d", len(pads), 2*r)
	}
	val := o.attrVal
	if len(in) > 2 && in[2] != nil && in[2].Numel() > 0 {
		val = in[2].F32()[0]
	}
	// Optional axes input (opset 18): restrict which dims pads apply to.
	begin := make([]int, r)
	end := make([]int, r)
	if len(in) > 3 && in[3] != nil {
		axes := asI64(in[3])
		for i, a := range axes {
			ax, err := normAxis(int(a), r)
			if err != nil {
				return nil, o.n.Errorf("%v", err)
			}
			begin[ax] = int(pads[i])
			end[ax] = int(pads[len(axes)+i])
		}
	} else {
		for i := 0; i < r; i++ {
			begin[i] = int(pads[i])
			end[i] = int(pads[r+i])
		}
	}
	oshape := make(tensor.Shape, r)
	for i := 0; i < r; i++ {
		oshape[i] = xs[i] + begin[i] + end[i]
		if oshape[i] < 0 {
			return nil, o.n.Errorf("negative padded dim %d", i)
		}
	}
	out := ctx.New(tensor.F32, oshape...)
	if o.mode == "constant" && val != 0 {
		of := out.F32()
		for i := range of {
			of[i] = val
		}
	}
	xf, of := x.F32(), out.F32()
	xst := xs.Strides()
	ost := oshape.Strides()
	// Iterate over output positions that map into the input region.
	idx := make([]int, r)
	src := make([]int, r)
	total := out.Numel()
	for oi := 0; oi < total; oi++ {
		// decode oi into idx
		rem := oi
		ok := true
		for d := 0; d < r; d++ {
			idx[d] = rem / ost[d]
			rem %= ost[d]
			s := idx[d] - begin[d]
			switch o.mode {
			case "edge":
				s = clampIdx(s, xs[d])
			case "reflect":
				s = reflectIdx(s, xs[d])
			default: // constant
				if s < 0 || s >= xs[d] {
					ok = false
				}
			}
			src[d] = s
		}
		if !ok {
			continue // constant region already filled
		}
		si := 0
		for d := 0; d < r; d++ {
			si += src[d] * xst[d]
		}
		of[oi] = xf[si]
	}
	return ctx.Out(out), nil
}

func reflectIdx(i, n int) int {
	if n == 1 {
		return 0
	}
	period := 2 * (n - 1)
	i %= period
	if i < 0 {
		i += period
	}
	if i >= n {
		i = period - i
	}
	return i
}

// dropoutOp: inference is identity; emits the (optional) mask as all-true.
type dropoutOp struct{ n NodeInfo }

func (o *dropoutOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	outs := ctx.OutPad(max(o.n.NumOut, 1), x.Clone())
	if o.n.NumOut > 1 {
		mask := ctx.New(tensor.Bool, x.Shape()...)
		mb := mask.Bool()
		for i := range mb {
			mb[i] = true
		}
		outs[1] = mask
	}
	return outs, nil
}

func init() {
	Register("", "Pad", 2, func(n NodeInfo) (Op, error) {
		return &padOp{n: n, mode: n.Attrs.String("mode", "constant"), attrPads: n.Attrs.Ints("pads", nil), attrVal: n.Attrs.Float("value", 0)}, nil
	})
	Register("", "Pad", 11, func(n NodeInfo) (Op, error) {
		return &padOp{n: n, mode: n.Attrs.String("mode", "constant")}, nil
	})
	Register("", "Dropout", 1, func(n NodeInfo) (Op, error) { return &dropoutOp{n}, nil })
}
