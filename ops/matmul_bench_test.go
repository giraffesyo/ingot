package ops

import (
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// BenchmarkMatMulGelu: the transformer fc1 (bias + GELU) fused into the GEMM
// versus MatMul(+bias) followed by the Gelu op.
func BenchmarkMatMulGelu(b *testing.B) {
	a := tensor.New(tensor.F32, 128, 384)
	w := tensor.New(tensor.F32, 384, 1536)
	bias := tensor.New(tensor.F32, 1536)
	for _, t := range []*tensor.Tensor{a, w, bias} {
		for i := range t.F32() {
			t.F32()[i] = float32(i%13)*0.05 - 0.3
		}
	}
	mk := func(attrs Attrs) Op {
		bld, _ := Lookup("", "MatMul", 1)
		op, err := bld(NodeInfo{Name: "mm", OpType: "MatMul", Version: 1, Attrs: attrs, NumIn: 3, NumOut: 1})
		if err != nil {
			b.Fatal(err)
		}
		return op
	}
	gb, _ := Lookup("ingot", "Gelu", 1)
	gelu, _ := gb(NodeInfo{Name: "g", OpType: "Gelu", Domain: "ingot", Version: 1, NumIn: 1, NumOut: 1})
	ctx := &Ctx{Pool: tensor.NewPool()}
	in := []*tensor.Tensor{a, w, bias}
	b.Run("separate", func(b *testing.B) {
		op := mk(nil)
		for i := 0; i < b.N; i++ {
			y, _ := op.Run(ctx, in)
			z, _ := gelu.Run(ctx, y)
			ctx.Pool.Put(y[0])
			ctx.Pool.Put(z[0])
		}
	})
	b.Run("fused", func(b *testing.B) {
		op := mk(Attrs{"ingot_act": {Kind: KindString, S: "gelu"}})
		for i := 0; i < b.N; i++ {
			y, _ := op.Run(ctx, in)
			ctx.Pool.Put(y[0])
		}
	})
}
