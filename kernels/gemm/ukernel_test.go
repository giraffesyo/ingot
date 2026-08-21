package gemm

import (
	"math/rand/v2"
	"testing"
)

// TestMicroKernelMatchesGeneric checks the active (possibly asm) micro-kernel
// against the portable one on random panels, both overwrite and accumulate.
func TestMicroKernelMatchesGeneric(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for _, kc := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 33, 100, 257} {
		for _, accumulate := range []bool{false, true} {
			ap := randMat(r, kc*MR)
			bp := randMat(r, kc*NR)
			ldc := NR + 5
			c0 := randMat(r, MR*ldc)
			want := append([]float32(nil), c0...)
			got := append([]float32(nil), c0...)
			microKernelGeneric(kc, ap, bp, want, ldc, accumulate)
			microKernel(kc, ap, bp, got, ldc, accumulate)
			for i := range want {
				d := want[i] - got[i]
				if d < 0 {
					d = -d
				}
				if d > 1e-4*(1+abs32(want[i])) {
					t.Fatalf("kc=%d acc=%v: idx %d want %g got %g", kc, accumulate, i, want[i], got[i])
				}
			}
		}
	}
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func BenchmarkMicroKernel(b *testing.B) {
	r := rand.New(rand.NewPCG(9, 10))
	kc := KC
	ap := randMat(r, kc*MR)
	bp := randMat(r, kc*NR)
	c := make([]float32, MR*NR)
	flops := 2 * float64(MR*NR*kc)
	b.Run("asm_or_active", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			microKernel(kc, ap, bp, c, NR, true)
		}
		b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
	})
	b.Run("generic", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			microKernelGeneric(kc, ap, bp, c, NR, true)
		}
		b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
	})
}
