package ops

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// pool1dRef is a naive NCW pooling oracle: max, or the mean of the taps
// inside the signal (count_include_pad 0).
func pool1dRef(x []float32, N, C, T, K, s, pl, pr int, isMax bool) ([]float32, int) {
	OT := (T+pl+pr-K)/s + 1
	out := make([]float32, N*C*OT)
	for nc := range N * C {
		for o := range OT {
			m, sum, cnt := math.Inf(-1), 0.0, 0
			for k := range K {
				if i := o*s + k - pl; i >= 0 && i < T {
					v := float64(x[nc*T+i])
					m = max(m, v)
					sum += v
					cnt++
				}
			}
			if isMax {
				out[nc*OT+o] = float32(m)
			} else {
				out[nc*OT+o] = float32(sum / float64(cnt))
			}
		}
	}
	return out, OT
}

// 1-D pooling (a codec's peak follower is MaxPool over time) against the
// oracle, across kernels, strides and asymmetric pads.
func TestPool1D(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for _, c := range []struct{ N, C, T, K, s, pl, pr int }{
		{1, 1, 40, 5, 1, 0, 0},
		{1, 1, 2048, 64, 1, 32, 31}, // the SAME decoder's limiter
		{2, 3, 37, 4, 2, 1, 2},
		{1, 2, 16, 3, 3, 0, 1},
	} {
		x := tensor.New(tensor.F32, c.N, c.C, c.T)
		for i := range x.F32() {
			x.F32()[i] = float32(rng.NormFloat64())
		}
		for _, kind := range []string{"MaxPool", "AveragePool"} {
			op := mkOp(t, kind, 11, Attrs{
				"kernel_shape": {Kind: KindInts, Ints: []int64{int64(c.K)}},
				"strides":      {Kind: KindInts, Ints: []int64{int64(c.s)}},
				"pads":         {Kind: KindInts, Ints: []int64{int64(c.pl), int64(c.pr)}},
			}, 1, 1)
			want, OT := pool1dRef(x.F32(), c.N, c.C, c.T, c.K, c.s, c.pl, c.pr, kind == "MaxPool")
			eqF32(t, kind, run(t, op, x)[0], []int{c.N, c.C, OT}, want)
		}
	}
	// Defaults: stride 1, no pads.
	op := mkOp(t, "MaxPool", 11, Attrs{"kernel_shape": {Kind: KindInts, Ints: []int64{2}}}, 1, 1)
	eqF32(t, "defaults", run(t, op, f32t([]int{1, 1, 4}, 1, 3, 2, 5))[0], []int{1, 1, 3}, []float32{3, 3, 5})
	// 3-D pooling stays a loud error.
	b, _ := Lookup("", "MaxPool", 11)
	if _, err := b(NodeInfo{Name: "p", OpType: "MaxPool", Attrs: Attrs{"kernel_shape": {Kind: KindInts, Ints: []int64{2, 2, 2}}}}); err == nil {
		t.Error("a 3-D MaxPool built")
	}
}

// 1-D Resize: linear and nearest over NCW, by scales and by sizes.
func TestResize1D(t *testing.T) {
	x := f32t([]int{1, 2, 3}, 0, 10, 20, 1, 2, 4)
	lin := mkOp(t, "Resize", 13, Attrs{"mode": {Kind: KindString, S: "linear"}, "coordinate_transformation_mode": {Kind: KindString, S: "align_corners"}}, 4, 1)
	eqF32(t, "linear sizes", run(t, lin, x, nil, nil, i64t([]int{3}, 1, 2, 5))[0], []int{1, 2, 5},
		[]float32{0, 5, 10, 15, 20, 1, 1.5, 2, 3, 4})
	nn := mkOp(t, "Resize", 13, Attrs{"mode": {Kind: KindString, S: "nearest"}, "coordinate_transformation_mode": {Kind: KindString, S: "asymmetric"}, "nearest_mode": {Kind: KindString, S: "floor"}}, 4, 1)
	eqF32(t, "nearest scales", run(t, nn, x, nil, f32t([]int{3}, 1, 1, 2))[0], []int{1, 2, 6},
		[]float32{0, 0, 10, 10, 20, 20, 1, 1, 2, 2, 4, 4})
	// half_pixel linear, the default: out[i] samples (i+0.5)/2 - 0.5.
	hp := mkOp(t, "Resize", 13, Attrs{"mode": {Kind: KindString, S: "linear"}}, 4, 1)
	eqF32(t, "half pixel", run(t, hp, f32t([]int{1, 1, 2}, 0, 8), nil, f32t([]int{3}, 1, 1, 2))[0], []int{1, 1, 4},
		[]float32{0, 2, 6, 8})
}

// Neg and Abs on the integer tensors exported shape arithmetic uses.
func TestNegAbsInt(t *testing.T) {
	neg := mkOp(t, "Neg", 13, nil, 1, 1)
	eqI64(t, "neg i64", run(t, neg, i64t([]int{3}, 4, -2, 0))[0], []int{3}, []int64{-4, 2, 0})
	abs := mkOp(t, "Abs", 13, nil, 1, 1)
	eqI64(t, "abs i64", run(t, abs, i64t([]int{3}, 4, -2, 0))[0], []int{3}, []int64{4, 2, 0})
	x := tensor.New(tensor.I32, 2)
	copy(x.I32(), []int32{-7, 9})
	out := run(t, neg, x)[0]
	if out.DType() != tensor.I32 || out.I32()[0] != 7 || out.I32()[1] != -9 {
		t.Errorf("neg i32 = %v %v", out.DType(), out.I32())
	}
	eqF32(t, "neg f32", run(t, neg, f32t([]int{2}, 1.5, -2))[0], []int{2}, []float32{-1.5, 2})
	// other unary ops stay f32-only
	if _, err := mkOp(t, "Sqrt", 13, nil, 1, 1).Run(&Ctx{Pool: tensor.NewPool()}, []*tensor.Tensor{i64t([]int{1}, 4)}); err == nil {
		t.Error("Sqrt of int64 ran")
	}
}

// Transpose of int32 (a codec's PCM, [N,C,T] to [N,T,C]) must move integers,
// not reinterpret them as floats: both the copy-run and the gather paths.
func TestTransposeInt32(t *testing.T) {
	for _, c := range []struct {
		shape []int
		perm  []int64
	}{
		{[]int{1, 2, 9}, []int64{0, 2, 1}},  // gather path
		{[]int{2, 3, 16}, []int64{1, 0, 2}}, // last axis kept: copy runs
	} {
		x := tensor.New(tensor.I32, c.shape...)
		for i := range x.I32() {
			x.I32()[i] = int32(i*7919 - 30000)
		}
		op := mkOp(t, "Transpose", 13, Attrs{"perm": {Kind: KindInts, Ints: c.perm}}, 1, 1)
		out := run(t, op, x)[0]
		if out.DType() != tensor.I32 {
			t.Fatalf("dtype %v", out.DType())
		}
		os := out.Shape()
		for a := range os[0] {
			for b := range os[1] {
				for d := range os[2] {
					idx := [3]int{}
					idx[c.perm[0]], idx[c.perm[1]], idx[c.perm[2]] = a, b, d
					want := x.I32()[(idx[0]*c.shape[1]+idx[1])*c.shape[2]+idx[2]]
					if got := out.I32()[(a*os[1]+b)*os[2]+d]; got != want {
						t.Fatalf("perm %v [%d,%d,%d] = %d, want %d", c.perm, a, b, d, got, want)
					}
				}
			}
		}
	}
}
