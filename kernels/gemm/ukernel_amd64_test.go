//go:build amd64

package gemm

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestAVX512MatchesGeneric cross-checks the AVX-512 micro-kernel against the
// portable one (runs only where the instructions are available).
func TestAVX512MatchesGeneric(t *testing.T) {
	if !HasAVX512 {
		t.Skip("no AVX-512F")
	}
	r := rand.New(rand.NewPCG(31, 32))
	for _, kc := range []int{0, 1, 3, 8, 16, 33, 257} {
		for _, acc := range []bool{false, true} {
			for _, withBias := range []bool{false, true} {
				ap := randMat(r, kc*MR)
				bp := randMat(r, kc*NR)
				ldc := NR + 5
				c0 := randMat(r, MR*ldc)
				var bias []float32
				if withBias {
					bias = randMat(r, NR)
				}
				want := append([]float32(nil), c0...)
				got := append([]float32(nil), c0...)
				microKernelGeneric(kc, ap, bp, want, ldc, acc, bias)
				microKernelAVX512(kc, ap, bp, got, ldc, acc, bias)
				for i := range want {
					if math.Abs(float64(want[i]-got[i])) > 1e-4*(1+math.Abs(float64(want[i]))) {
						t.Fatalf("kc=%d acc=%v bias=%v idx %d: want %g got %g", kc, acc, withBias, i, want[i], got[i])
					}
				}
			}
		}
	}
}

// TestPairModeNarrowEdge: with the AVX-512 kernel active, the packed-B
// small-M sweep pairs panels; a pair whose second panel is a narrow edge
// (n = 24: a full panel and an 8-wide one) runs as two singles, and the
// edge must still be written. The init probe picks AVX-512 only where it is
// faster, so the test forces it.
func TestPairModeNarrowEdge(t *testing.T) {
	if !HasAVX512 {
		t.Skip("no AVX-512F")
	}
	defer func(k func(int, []float32, []float32, []float32, int, bool, []float32), name string) {
		microKernel, ActiveKernel = k, name
	}(microKernel, ActiveKernel)
	microKernel, ActiveKernel = microKernelAVX512, "avx512"
	r := rand.New(rand.NewPCG(41, 42))
	for _, c := range []struct{ m, n, k int }{
		// NR+8, 3·NR+5: the last pair ends in a narrow edge; 2·NR+8: a lone
		// edge panel; 4·NR: pairs only. m=64, k=24 is SparseConv's 1-tap case.
		{64, NR + 8, 24}, {7, NR + 8, 5}, {13, 3*NR + 5, 33}, {6, 2*NR + 8, 400},
		{9, 4 * NR, 17}, {64, 1000, 24},
	} {
		a, b, bias := randMat(r, c.m*c.k), randMat(r, c.k*c.n), randMat(r, c.n)
		pb := PackB(false, c.k, c.n, b, c.n)
		for _, withBias := range []bool{false, true} {
			var e Epilogue
			if withBias {
				e.Bias = bias
			}
			want := epiRef(c.m, c.n, c.k, a, b, e.Bias, nil, 0, nil)
			got := make([]float32, c.m*c.n)
			for i := range got {
				got[i] = float32(math.NaN()) // an unwritten element fails
			}
			if withBias {
				SgemmPackedBEpi(c.m, a, c.k, pb, got, c.n, &e)
			} else {
				SgemmPackedB(c.m, 1, a, c.k, pb, 0, got, c.n)
			}
			tol := 1e-5 * math.Sqrt(float64(c.k))
			for i, w := range want {
				if g := float64(got[i]); !(math.Abs(g-w) <= tol*(1+math.Abs(w))) {
					t.Fatalf("m=%d n=%d k=%d bias=%v: [%d,%d] = %g, want %g", c.m, c.n, c.k, withBias, i/c.n, i%c.n, g, w)
				}
			}
		}
	}
}

func BenchmarkMicroKernelVariants(b *testing.B) {
	r := rand.New(rand.NewPCG(9, 10))
	kc := KC
	ap := randMat(r, kc*MR)
	bp := randMat(r, kc*NR)
	c := make([]float32, MR*NR)
	flops := 2 * float64(MR*NR*kc)
	bench := func(name string, fn func(int, []float32, []float32, []float32, int, bool, []float32)) {
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fn(kc, ap, bp, c, NR, true, nil)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
	if HasAVX512 {
		bench("avx512", microKernelAVX512)
	}
	if HasAVX2 {
		bench("avx2", microKernelAVX2)
	}
	bench("generic", microKernelGeneric)
}
