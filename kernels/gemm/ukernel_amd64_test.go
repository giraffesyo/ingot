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
