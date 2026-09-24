package ops

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// toBlk / fromBlk convert NCHW <-> nChw8c for the tests.
func toBlk(x []float32, N, C, P int) []float32 {
	out := make([]float32, len(x))
	for n := range N {
		for c := range C {
			for p := range P {
				out[((n*(C/8)+c/8)*P+p)*8+c%8] = x[(n*C+c)*P+p]
			}
		}
	}
	return out
}

func fromBlk(x []float32, N, C, P int) []float32 {
	out := make([]float32, len(x))
	for n := range N {
		for c := range C {
			for p := range P {
				out[(n*C+c)*P+p] = x[((n*(C/8)+c/8)*P+p)*8+c%8]
			}
		}
	}
	return out
}

func denseRef(x, w, b []float32, N, C, H, W, M, K, S int, pads [4]int) ([]float64, int, int) {
	OH, OW := (H+pads[0]+pads[2]-K)/S+1, (W+pads[1]+pads[3]-K)/S+1
	out := make([]float64, N*M*OH*OW)
	for n := range N {
		for m := range M {
			for oy := range OH {
				for ox := range OW {
					acc := float64(b[m])
					for c := range C {
						for ky := range K {
							for kx := range K {
								iy, ix := oy*S+ky-pads[0], ox*S+kx-pads[1]
								if iy >= 0 && iy < H && ix >= 0 && ix < W {
									acc += float64(x[((n*C+c)*H+iy)*W+ix]) * float64(w[((m*C+c)*K+ky)*K+kx])
								}
							}
						}
					}
					out[((n*M+m)*OH+oy)*OW+ox] = acc
				}
			}
		}
	}
	return out, OH, OW
}

type denseCase struct{ N, C, H, W, M, K, S, pad int }

func runDense(c denseCase, x, w, b []float32) ([]float32, int, int) {
	pads := [4]int{c.pad, c.pad, c.pad, c.pad}
	g := newDenseGeom(c.H, c.W, c.K, c.K, c.S, pads)
	np, na := denseBlkScratch(c.N, c.C, c.M, g)
	wp := denseBlkWeights(w, c.M, c.C, c.K, c.K)
	pad, acc := make([]float32, np), make([]float32, na)
	out := make([]float32, c.N*c.M*g.OH*g.OW)
	xb := toBlk(x, c.N, c.C, c.H*c.W)
	denseBlkConv(xb, wp, b, out, pad, acc, c.N, c.C, c.H, c.W, c.M, c.K, c.K, pads, g, nil)
	return fromBlk(out, c.N, c.M, g.OH*g.OW), g.OH, g.OW
}

// TestDenseBlkConv: the blocked dense conv vs a float64 oracle (stride 1,
// K 3 and 5, odd output-block counts, 4×4 planes and up, batch 2).
func TestDenseBlkConv(t *testing.T) {
	r := rand.New(rand.NewPCG(31, 31))
	for _, c := range []denseCase{
		{1, 16, 16, 16, 16, 3, 1, 1},
		{2, 32, 8, 8, 24, 3, 1, 1},
		{1, 64, 4, 4, 64, 3, 1, 1},
		{1, 8, 7, 11, 8, 5, 1, 2},
		{1, 16, 9, 5, 40, 3, 1, 0},
		{1, 16, 16, 16, 32, 3, 2, 1},
		{2, 32, 9, 7, 16, 3, 2, 1},
		{1, 16, 16, 16, 32, 1, 2, 0},
		{1, 8, 11, 13, 16, 5, 2, 2},
	} {
		x := make([]float32, c.N*c.C*c.H*c.W)
		w := make([]float32, c.M*c.C*c.K*c.K)
		b := make([]float32, c.M)
		for _, s := range [][]float32{x, w, b} {
			for i := range s {
				s[i] = r.Float32()*2 - 1
			}
		}
		want, _, _ := denseRef(x, w, b, c.N, c.C, c.H, c.W, c.M, c.K, c.S, [4]int{c.pad, c.pad, c.pad, c.pad})
		got, _, _ := runDense(c, x, w, b)
		tol := 1e-5 * math.Sqrt(float64(c.C*c.K*c.K)) * 4
		for i, v := range got {
			if math.Abs(float64(v)-want[i]) > tol*(1+math.Abs(want[i])) {
				t.Fatalf("%+v [%d] = %g, want %g", c, i, v, want[i])
			}
		}
	}
}

// BenchmarkDenseBlkConv: the blocked dense conv on resnetish's shapes (the
// NCHW path is BenchmarkConvSmall).
func BenchmarkDenseBlkConv(b *testing.B) {
	for _, c := range []denseCase{
		{1, 16, 16, 16, 16, 3, 1, 1},
		{1, 16, 16, 16, 32, 3, 2, 1},
		{1, 32, 8, 8, 32, 3, 1, 1},
		{1, 32, 8, 8, 64, 3, 2, 1},
		{1, 64, 4, 4, 64, 3, 1, 1},
	} {
		pads := [4]int{c.pad, c.pad, c.pad, c.pad}
		g := newDenseGeom(c.H, c.W, c.K, c.K, c.S, pads)
		np, na := denseBlkScratch(c.N, c.C, c.M, g)
		x := make([]float32, c.N*c.C*c.H*c.W)
		wp := denseBlkWeights(make([]float32, c.M*c.C*c.K*c.K), c.M, c.C, c.K, c.K)
		bias := make([]float32, c.M)
		pad, acc := make([]float32, np), make([]float32, na)
		out := make([]float32, c.N*c.M*g.OH*g.OW)
		b.Run(fmt.Sprintf("c%d_%dx%d_m%d_s%d", c.C, c.H, c.W, c.M, c.S), func(b *testing.B) {
			for b.Loop() {
				denseBlkConv(x, wp, bias, out, pad, acc, c.N, c.C, c.H, c.W, c.M, c.K, c.K, pads, g, nil)
			}
			b.ReportMetric(2*float64(c.M*c.C*c.K*c.K*g.OH*g.OW)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}
