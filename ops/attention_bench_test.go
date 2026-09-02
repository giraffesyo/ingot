package ops

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// BenchmarkSDPA times the fused attention op at representative head
// geometries: PARSeq's encoder (B=1, few heads, T=128), a ViT-B-ish shape,
// and a causal decoder block (masked, flash path).
func BenchmarkSDPA(b *testing.B) {
	for _, c := range []struct {
		B, H, T, dh int
		causal      bool
	}{
		{1, 6, 128, 64, false},
		{8, 6, 128, 64, false},
		{1, 12, 197, 64, false},
		{1, 12, 1024, 64, true},
	} {
		name := fmt.Sprintf("B=%d/H=%d/T=%d/dh=%d", c.B, c.H, c.T, c.dh)
		if c.causal {
			name += "/causal"
		}
		b.Run(name, func(b *testing.B) {
			rng := rand.New(rand.NewPCG(1, 2))
			q := tensor.New(tensor.F32, c.B, c.H, c.T, c.dh)
			k := tensor.New(tensor.F32, c.B, c.H, c.dh, c.T)
			v := tensor.New(tensor.F32, c.B, c.H, c.T, c.dh)
			for _, t := range []*tensor.Tensor{q, k, v} {
				for i := range t.F32() {
					t.F32()[i] = rng.Float32() - 0.5
				}
			}
			in := []*tensor.Tensor{q, k, v}
			if c.causal {
				m := tensor.New(tensor.F32, c.T, c.T)
				for i := 0; i < c.T; i++ {
					for j := i + 1; j < c.T; j++ {
						m.F32()[i*c.T+j] = -1e9
					}
				}
				in = append(in, m)
			}
			bld, err := Lookup("ingot", "SDPA", 1)
			if err != nil {
				b.Fatal(err)
			}
			op, err := bld(NodeInfo{Name: "sdpa", OpType: "SDPA", Domain: "ingot", Version: 1,
				Attrs: Attrs{"scale": {Kind: KindFloat, F: 0.125}}, NumIn: len(in), NumOut: 1})
			if err != nil {
				b.Fatal(err)
			}
			ctx := &Ctx{Pool: tensor.NewPool()}
			flops := 4 * float64(c.B*c.H*c.T*c.T*c.dh)
			b.SetBytes(0)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out, err := op.Run(ctx, in)
				if err != nil {
					b.Fatal(err)
				}
				ctx.Pool.Put(out[0])
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}
