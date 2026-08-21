package ops

import (
	"math"
	"testing"

	"github.com/giraffesyo/ocr/tensor"
)

// mkOp builds an op via the registry at the given opset with attributes and
// input/output arity, failing the test on error.
func mkOp(t *testing.T, name string, ver int, attrs Attrs, numIn, numOut int) Op {
	t.Helper()
	b, err := Lookup("", name, ver)
	if err != nil {
		t.Fatalf("Lookup %s@%d: %v", name, ver, err)
	}
	op, err := b(NodeInfo{Name: name, OpType: name, Version: ver, Attrs: attrs, NumIn: numIn, NumOut: numOut})
	if err != nil {
		t.Fatalf("build %s: %v", name, err)
	}
	return op
}

func run(t *testing.T, op Op, in ...*tensor.Tensor) []*tensor.Tensor {
	t.Helper()
	out, err := op.Run(&Ctx{Pool: tensor.NewPool()}, in)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	return out
}

func f32t(shape []int, data ...float32) *tensor.Tensor { return tensor.FromF32(data, shape...) }
func i64t(shape []int, data ...int64) *tensor.Tensor   { return tensor.FromI64(data, shape...) }

func iota32(shape ...int) *tensor.Tensor {
	t := tensor.New(tensor.F32, shape...)
	f := t.F32()
	for i := range f {
		f[i] = float32(i)
	}
	return t
}

func eqF32(t *testing.T, name string, got *tensor.Tensor, wantShape []int, want []float32) {
	t.Helper()
	if !got.Shape().Equal(tensor.Shape(wantShape)) {
		t.Fatalf("%s: shape %v, want %v", name, got.Shape(), wantShape)
	}
	gf := got.F32()
	if len(gf) != len(want) {
		t.Fatalf("%s: len %d want %d", name, len(gf), len(want))
	}
	for i := range want {
		if math.Abs(float64(gf[i]-want[i])) > 1e-4*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("%s[%d] = %g, want %g\n got=%v\nwant=%v", name, i, gf[i], want[i], gf, want)
		}
	}
}

func eqI64(t *testing.T, name string, got *tensor.Tensor, wantShape []int, want []int64) {
	t.Helper()
	if !got.Shape().Equal(tensor.Shape(wantShape)) {
		t.Fatalf("%s: shape %v want %v", name, got.Shape(), wantShape)
	}
	gi := got.I64()
	for i := range want {
		if gi[i] != want[i] {
			t.Fatalf("%s[%d] = %d want %d (got %v)", name, i, gi[i], want[i], gi)
		}
	}
}
