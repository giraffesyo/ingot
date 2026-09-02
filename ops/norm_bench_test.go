package ops

import (
	"fmt"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// BenchmarkLayerNorm times LayerNormalization at transformer row shapes.
func BenchmarkLayerNorm(b *testing.B) {
	for _, c := range []struct{ rows, D int }{{128, 384}, {1024, 384}, {16, 48}} {
		x := tensor.New(tensor.F32, 1, c.rows, c.D)
		for i := range x.F32() {
			x.F32()[i] = float32(i%17)*0.1 - 0.8
		}
		sc := tensor.New(tensor.F32, c.D)
		bs := tensor.New(tensor.F32, c.D)
		for i := range sc.F32() {
			sc.F32()[i] = 1 + float32(i%3)*0.1
			bs.F32()[i] = float32(i%5) * 0.01
		}
		bld, _ := Lookup("", "LayerNormalization", 17)
		op, err := bld(NodeInfo{Name: "ln", OpType: "LayerNormalization", Version: 17,
			Attrs: Attrs{"axis": {Kind: KindInt, I: -1}, "epsilon": {Kind: KindFloat, F: 1e-5}}, NumIn: 3, NumOut: 1})
		if err != nil {
			b.Fatal(err)
		}
		ctx := &Ctx{Pool: tensor.NewPool()}
		in := []*tensor.Tensor{x, sc, bs}
		b.Run(fmt.Sprintf("rows=%d/D=%d", c.rows, c.D), func(b *testing.B) {
			b.ReportAllocs()
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
