package ops

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ocr/tensor"
)

func randT(r *rand.Rand, shape ...int) *tensor.Tensor {
	t := tensor.New(tensor.F32, shape...)
	f := t.F32()
	for i := range f {
		f[i] = r.Float32()*4 - 2
	}
	return t
}

// naiveBin computes a OP b with NumPy broadcasting as the oracle.
func naiveBin(op byte, a, b *tensor.Tensor) *tensor.Tensor {
	os, err := broadcastShape(a.Shape(), b.Shape())
	if err != nil {
		panic(err)
	}
	out := tensor.New(tensor.F32, os...)
	of := out.F32()
	ast := broadcastStrides(a.Shape(), os)
	bst := broadcastStrides(b.Shape(), os)
	af, bf := a.F32(), b.F32()
	idx := make([]int, len(os))
	for i := range of {
		oa, ob := 0, 0
		for d := range idx {
			oa += idx[d] * ast[d]
			ob += idx[d] * bst[d]
		}
		x, y := af[oa], bf[ob]
		switch op {
		case '+':
			of[i] = x + y
		case '-':
			of[i] = x - y
		case '*':
			of[i] = x * y
		case '/':
			of[i] = x / y
		}
		for d := len(idx) - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < os[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out
}

func TestBinaryBroadcast(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	ctx := &Ctx{Pool: tensor.NewPool()}
	type pair struct{ a, b []int }
	cases := []pair{
		{[]int{2, 3, 4}, []int{2, 3, 4}},       // same shape
		{[]int{2, 3, 4}, []int{1}},             // scalar rhs
		{[]int{1}, []int{2, 3, 4}},             // scalar lhs
		{[]int{2, 8, 5, 5}, []int{2, 8, 1, 1}}, // squeeze-excite block broadcast
		{[]int{2, 8, 1, 1}, []int{2, 8, 5, 5}}, // reversed
		{[]int{2, 3, 4}, []int{3, 4}},          // trailing broadcast (generic)
		{[]int{2, 1, 4}, []int{2, 3, 4}},       // middle broadcast (generic)
		{[]int{4}, []int{3, 4}},                // rank-expanding
	}
	ops := map[byte]string{'+': "Add", '-': "Sub", '*': "Mul", '/': "Div"}
	for _, c := range cases {
		a := randT(r, c.a...)
		b := randT(r, c.b...)
		// keep divisor away from zero
		bf := b.F32()
		for i := range bf {
			if bf[i] > -0.1 && bf[i] < 0.1 {
				bf[i] += 1
			}
		}
		for sym, name := range ops {
			op := &binaryOp{n: NodeInfo{OpType: name}, kind: sym, fn: nil}
			// fn only needed for generic path; supply it
			switch sym {
			case '+':
				op.fn = func(x, y float32) float32 { return x + y }
			case '-':
				op.fn = func(x, y float32) float32 { return x - y }
			case '*':
				op.fn = func(x, y float32) float32 { return x * y }
			case '/':
				op.fn = func(x, y float32) float32 { return x / y }
			}
			outs, err := op.Run(ctx, []*tensor.Tensor{a, b})
			if err != nil {
				t.Fatalf("%s %v %v: %v", name, c.a, c.b, err)
			}
			got := outs[0]
			want := naiveBin(sym, a, b)
			if !got.Shape().Equal(want.Shape()) {
				t.Fatalf("%s %v %v: shape %v want %v", name, c.a, c.b, got.Shape(), want.Shape())
			}
			gf, wf := got.F32(), want.F32()
			for i := range wf {
				if math.Abs(float64(gf[i]-wf[i])) > 1e-5*(1+math.Abs(float64(wf[i]))) {
					t.Fatalf("%s %v %v [%d]: got %g want %g", name, c.a, c.b, i, gf[i], wf[i])
				}
			}
		}
	}
}
