package ops

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// naiveReduce is the float64 oracle over an arbitrary axis set.
func naiveReduce(kind string, x []float32, shape []int, red []bool) []float64 {
	r := len(shape)
	var oshape []int
	for i, d := range shape {
		if !red[i] {
			oshape = append(oshape, d)
		}
	}
	on := 1
	for _, d := range oshape {
		on *= d
	}
	acc := make([]float64, on)
	for i := range acc {
		switch kind {
		case "max":
			acc[i] = math.Inf(-1)
		case "min":
			acc[i] = math.Inf(1)
		case "prod":
			acc[i] = 1
		}
	}
	cnt := 1
	for i, d := range shape {
		if red[i] {
			cnt *= d
		}
	}
	idx := make([]int, r)
	for flat := range x {
		rem := flat
		for d := r - 1; d >= 0; d-- {
			idx[d] = rem % shape[d]
			rem /= shape[d]
		}
		oi := 0
		for d := range r {
			if !red[d] {
				oi = oi*shape[d] + idx[d]
			}
		}
		v := float64(x[flat])
		switch kind {
		case "sum", "mean":
			acc[oi] += v
		case "max":
			acc[oi] = math.Max(acc[oi], v)
		case "min":
			acc[oi] = math.Min(acc[oi], v)
		case "prod":
			acc[oi] *= v
		case "l2", "sumsq":
			acc[oi] += v * v
		case "l1":
			acc[oi] += math.Abs(v)
		}
	}
	for i := range acc {
		switch kind {
		case "mean":
			acc[i] /= float64(cnt)
		case "l2":
			acc[i] = math.Sqrt(acc[i])
		}
	}
	return acc
}

// TestReduceAxes covers the trailing, middle-block and generic paths for
// every reduce kind against the float64 oracle.
func TestReduceAxes(t *testing.T) {
	rng := rand.New(rand.NewPCG(61, 62))
	ops := map[string]string{"ReduceSum": "sum", "ReduceMean": "mean", "ReduceMax": "max", "ReduceMin": "min",
		"ReduceProd": "prod", "ReduceL2": "l2", "ReduceL1": "l1", "ReduceSumSquare": "sumsq"}
	cases := []struct {
		shape []int
		axes  []int64
	}{
		{[]int{2, 3, 4, 5}, []int64{3}},      // trailing
		{[]int{2, 3, 4, 5}, []int64{2, 3}},   // trailing block
		{[]int{2, 3, 4, 5}, []int64{1}},      // middle
		{[]int{2, 3, 4, 5}, []int64{1, 2}},   // middle block
		{[]int{2, 3, 4, 5}, []int64{0}},      // leading
		{[]int{2, 3, 4, 5}, []int64{-4, -3}}, // leading block, negative axes
		{[]int{2, 3, 4, 5}, []int64{0, 2}},   // non-contiguous: generic
		{[]int{1, 9, 70, 70}, []int64{1}},    // NCHW channel norm, inner > reduceChunk
		{[]int{3, 5, 1}, []int64{1}},         // inner of 1
	}
	for name, kind := range ops {
		for _, c := range cases {
			for _, keep := range []int64{0, 1} {
				t.Run(fmt.Sprintf("%s/%v/axes=%v/keep=%d", name, c.shape, c.axes, keep), func(t *testing.T) {
					x := tensor.New(tensor.F32, c.shape...)
					for i := range x.F32() {
						x.F32()[i] = rng.Float32()*1.5 + 0.25 // positive: keeps prod well-scaled
						if kind != "prod" && rng.IntN(2) == 0 {
							x.F32()[i] = -x.F32()[i]
						}
					}
					red := make([]bool, len(c.shape))
					for _, a := range c.axes {
						if a < 0 {
							a += int64(len(c.shape))
						}
						red[a] = true
					}
					want := naiveReduce(kind, x.F32(), c.shape, red)
					op := mkOp(t, name, 18, Attrs{"keepdims": {Kind: KindInt, I: keep}}, 2, 1)
					got := run(t, op, x, i64t([]int{len(c.axes)}, c.axes...))[0]
					var wantShape []int
					cnt := 1
					for i, d := range c.shape {
						switch {
						case !red[i]:
							wantShape = append(wantShape, d)
						case keep == 1:
							wantShape = append(wantShape, 1)
						}
						if red[i] {
							cnt *= d
						}
					}
					if !got.Shape().Equal(tensor.Shape(wantShape)) {
						t.Fatalf("shape %v, want %v", got.Shape(), wantShape)
					}
					tol := 1e-5 * math.Sqrt(float64(cnt))
					for i, w := range want {
						if d := math.Abs(float64(got.F32()[i]) - w); d > tol*(1+math.Abs(w)) {
							t.Fatalf("[%d] = %g, want %g", i, got.F32()[i], w)
						}
					}
				})
			}
		}
	}
}

// BenchmarkReduceL2Channels times the VAE decoder's channel norm (ReduceL2
// over axis 1 of NCHW) at two decoder stages.
func BenchmarkReduceL2Channels(b *testing.B) {
	for _, s := range [][]int{{1, 1152, 32, 32}, {1, 288, 128, 128}} {
		x := tensor.New(tensor.F32, s...)
		for i := range x.F32() {
			x.F32()[i] = float32(i%13) * 0.1
		}
		bld, _ := Lookup("", "ReduceL2", 18)
		op, _ := bld(NodeInfo{Name: "l2", OpType: "ReduceL2", Version: 18, NumIn: 2, NumOut: 1,
			Attrs: Attrs{"keepdims": {Kind: KindInt, I: 1}}})
		ctx := &Ctx{Pool: tensor.NewPool()}
		in := []*tensor.Tensor{x, tensor.FromI64([]int64{1}, 1)}
		b.Run(fmt.Sprintf("shape=%v", s), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(4 * x.Numel()))
			for i := 0; i < b.N; i++ {
				out, err := op.Run(ctx, in)
				if err != nil {
					b.Fatal(err)
				}
				ctx.Pool.Put(out[0])
			}
		})
	}
}
