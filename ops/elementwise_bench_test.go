package ops

import (
	"fmt"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// BenchmarkTrig times Sin/Cos at embedding-table and activation sizes.
func BenchmarkTrig(b *testing.B) {
	for _, name := range []string{"Sin", "Cos"} {
		for _, n := range []int{256, 1 << 20} {
			x := tensor.New(tensor.F32, n)
			for i := range x.F32() {
				x.F32()[i] = float32(i%1000) * 0.37
			}
			bld, _ := Lookup("", name, 7)
			op, err := bld(NodeInfo{Name: name, OpType: name, Version: 7, NumIn: 1, NumOut: 1})
			if err != nil {
				b.Fatal(err)
			}
			ctx := &Ctx{Pool: tensor.NewPool()}
			in := []*tensor.Tensor{x}
			b.Run(fmt.Sprintf("op=%s/n=%d", name, n), func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(8 * n))
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
}
