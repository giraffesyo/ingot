package ops

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// BenchmarkConvSmall: the ResNet-style small convs (resnetish zoo model)
// where fixed costs — parallel fan-out, packing — dominate the FLOPs.
func BenchmarkConvSmall(b *testing.B) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, c := range []struct {
		Cin, H, W, M, K, S, P int
	}{
		{3, 64, 64, 16, 7, 2, 3},  // stem
		{16, 16, 16, 16, 3, 1, 1}, // l1
		{16, 16, 16, 32, 3, 2, 1}, // l2 in
		{32, 8, 8, 32, 3, 1, 1},   // l2
		{32, 8, 8, 64, 3, 2, 1},   // l3 in
		{64, 4, 4, 64, 3, 1, 1},   // l3
	} {
		x := tensor.New(tensor.F32, 1, c.Cin, c.H, c.W)
		for i := range x.F32() {
			x.F32()[i] = r.Float32()*2 - 1
		}
		w := tensor.New(tensor.F32, c.M, c.Cin, c.K, c.K)
		for i := range w.F32() {
			w.F32()[i] = r.Float32()*2 - 1
		}
		bias := tensor.New(tensor.F32, c.M)
		op := mkOpB(b, "Conv", 1, Attrs{
			"kernel_shape": {Kind: KindInts, Ints: []int64{int64(c.K), int64(c.K)}},
			"strides":      {Kind: KindInts, Ints: []int64{int64(c.S), int64(c.S)}},
			"pads":         {Kind: KindInts, Ints: []int64{int64(c.P), int64(c.P), int64(c.P), int64(c.P)}},
		}, 3, 1)
		ctx := &Ctx{Pool: tensor.NewPool()}
		oh := (c.H+2*c.P-c.K)/c.S + 1
		macs := c.M * c.Cin * c.K * c.K * oh * oh
		b.Run(fmt.Sprintf("c%d_%dx%d_m%d_k%d_s%d", c.Cin, c.H, c.W, c.M, c.K, c.S), func(b *testing.B) {
			for b.Loop() {
				outs, err := op.Run(ctx, []*tensor.Tensor{x, w, bias})
				if err != nil {
					b.Fatal(err)
				}
				ctx.Pool.Put(outs[0])
			}
			b.ReportMetric(2*float64(macs)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}
