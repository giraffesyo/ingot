package ops

import (
	"fmt"

	"github.com/giraffesyo/ingot/tensor"
)

// broadcastShape computes the NumPy-style broadcast of two shapes.
func broadcastShape(a, b tensor.Shape) (tensor.Shape, error) {
	n := max(len(a), len(b))
	out := make(tensor.Shape, n)
	for i := 0; i < n; i++ {
		da, db := 1, 1
		if j := len(a) - n + i; j >= 0 {
			da = a[j]
		}
		if j := len(b) - n + i; j >= 0 {
			db = b[j]
		}
		switch {
		case da == db, db == 1:
			out[i] = da
		case da == 1:
			out[i] = db
		default:
			return nil, fmt.Errorf("cannot broadcast %v with %v", a, b)
		}
	}
	return out, nil
}

// broadcastStrides returns element strides for reading `shape` as if it were
// `out` (zero stride on broadcast dims). len(out) >= len(shape).
func broadcastStrides(shape, out tensor.Shape) []int {
	st := make([]int, len(out))
	acc := 1
	for i := len(shape) - 1; i >= 0; i-- {
		oi := len(out) - len(shape) + i
		if shape[i] == 1 {
			st[oi] = 0
		} else {
			st[oi] = acc
		}
		acc *= shape[i]
	}
	return st
}

// binaryF32 applies fn elementwise with broadcasting over float32 tensors.
// Fast paths: identical shapes; b scalar; b broadcast along last dim(s).
func binaryF32(ctx *Ctx, a, b *tensor.Tensor, fn func(x, y float32) float32) (*tensor.Tensor, error) {
	as, bs := a.Shape(), b.Shape()
	af, bf := a.F32(), b.F32()
	if as.Equal(bs) {
		out := ctx.New(tensor.F32, as...)
		of := out.F32()
		for i := range of {
			of[i] = fn(af[i], bf[i])
		}
		return out, nil
	}
	if len(bf) == 1 {
		out := ctx.New(tensor.F32, as...)
		of := out.F32()
		y := bf[0]
		for i := range of {
			of[i] = fn(af[i], y)
		}
		return out, nil
	}
	if len(af) == 1 {
		out := ctx.NewUninit(tensor.F32, bs...)
		of := out.F32()
		x := af[0]
		for i := range of {
			of[i] = fn(x, bf[i])
		}
		return out, nil
	}
	os, err := broadcastShape(as, bs)
	if err != nil {
		return nil, err
	}
	out := ctx.New(tensor.F32, os...)
	of := out.F32()
	ast := broadcastStrides(as, os)
	bst := broadcastStrides(bs, os)
	// Generic N-d iteration with odometer indices; inner dim specialised.
	nd := len(os)
	if nd == 0 {
		of[0] = fn(af[0], bf[0])
		return out, nil
	}
	inner := os[nd-1]
	ia, ib := ast[nd-1], bst[nd-1]
	idx := make([]int, nd)
	offA, offB := 0, 0
	for oi := 0; oi < len(of); oi += inner {
		row := of[oi : oi+inner]
		switch {
		case ia == 1 && ib == 1:
			for j := range row {
				row[j] = fn(af[offA+j], bf[offB+j])
			}
		case ia == 1 && ib == 0:
			y := bf[offB]
			for j := range row {
				row[j] = fn(af[offA+j], y)
			}
		case ia == 0 && ib == 1:
			x := af[offA]
			for j := range row {
				row[j] = fn(x, bf[offB+j])
			}
		default:
			for j := range row {
				row[j] = fn(af[offA+j*ia], bf[offB+j*ib])
			}
		}
		// advance odometer over dims [0, nd-2]
		for d := nd - 2; d >= 0; d-- {
			idx[d]++
			offA += ast[d]
			offB += bst[d]
			if idx[d] < os[d] {
				break
			}
			offA -= ast[d] * os[d]
			offB -= bst[d] * os[d]
			idx[d] = 0
		}
	}
	return out, nil
}
