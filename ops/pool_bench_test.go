package ops

import (
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// BenchmarkMaxPool: resnet-style pools (the stem shape is the one that shows
// up in real CNNs; see docs/PERF.md "Separable vectorised MaxPool").
func BenchmarkMaxPool(b *testing.B) {
	for _, c := range []struct {
		name    string
		C, H, W int
		k, s, p int
	}{
		{"res_16x32x32_k3s2", 16, 32, 32, 3, 2, 1},
		{"stem_64x112x112_k3s2", 64, 112, 112, 3, 2, 1},
		{"k2s2_32x64x64", 32, 64, 64, 2, 2, 0},
	} {
		x := iota32(1, c.C, c.H, c.W)
		op := mkOpB(b, "MaxPool", 1, Attrs{
			"kernel_shape": {Kind: KindInts, Ints: []int64{int64(c.k), int64(c.k)}},
			"strides":      {Kind: KindInts, Ints: []int64{int64(c.s), int64(c.s)}},
			"pads":         {Kind: KindInts, Ints: []int64{int64(c.p), int64(c.p), int64(c.p), int64(c.p)}},
		}, 1, 1)
		ctx := &Ctx{Pool: tensor.NewPool()}
		b.Run(c.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				outs, err := op.Run(ctx, []*tensor.Tensor{x})
				if err != nil {
					b.Fatal(err)
				}
				ctx.Pool.Put(outs[0])
			}
		})
	}
}
