package ops

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// BenchmarkSDPA times the fused attention op at representative head
// geometries: PARSeq's encoder (B=1, few heads, T=128), a ViT-B-ish shape,
// a causal decoder block (masked, flash path), and Qwen-Image-2.1's long
// unmasked shapes: the DiT cached step (4096 target queries over [text
// prefix, target] or [text + reference image, target] keys) and the VAE
// mid-block's single 1152-wide head.
func BenchmarkSDPA(b *testing.B) {
	for _, c := range []struct {
		B, H, T, Tk, dh int
		causal          bool
	}{
		{1, 6, 128, 0, 64, false},
		{8, 6, 128, 0, 64, false},
		{1, 12, 197, 0, 64, false},
		{1, 12, 1024, 0, 64, true},
		{1, 32, 4096, 4352, 128, false},
		{1, 32, 4096, 8448, 128, false},
		{1, 1, 4096, 0, 1152, false},
	} {
		if c.Tk == 0 {
			c.Tk = c.T
		}
		name := fmt.Sprintf("B=%d/H=%d/T=%d/dh=%d", c.B, c.H, c.T, c.dh)
		if c.Tk != c.T {
			name = fmt.Sprintf("B=%d/H=%d/T=%d/Tk=%d/dh=%d", c.B, c.H, c.T, c.Tk, c.dh)
		}
		if c.causal {
			name += "/causal"
		}
		b.Run(name, func(b *testing.B) {
			rng := rand.New(rand.NewPCG(1, 2))
			q := tensor.New(tensor.F32, c.B, c.H, c.T, c.dh)
			k := tensor.New(tensor.F32, c.B, c.H, c.dh, c.Tk)
			v := tensor.New(tensor.F32, c.B, c.H, c.Tk, c.dh)
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
			flops := 4 * float64(c.B*c.H*c.T*c.Tk*c.dh)
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
