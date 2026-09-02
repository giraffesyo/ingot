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
				microKernelGeneric(kc, ap, bp, want, ldc, accumulate, bias)
				microKernel(kc, ap, bp, got, ldc, accumulate, bias)
				for i := range want {
					d := want[i] - got[i]
					if d < 0 {
						d = -d
					}
					if d > 1e-4*(1+abs32(want[i])) {
						t.Fatalf("kc=%d acc=%v bias=%v: idx %d want %g got %g", kc, accumulate, withBias, i, want[i], got[i])
					}
				}
			}
		}
	}
}

// TestSgemmGenericFallback forces the portable micro-kernel (the path taken on
// CPUs without AVX2/NEON) and checks Sgemm still matches the float64 oracle.
func TestSgemmGenericFallback(t *testing.T) {
	saved := microKernel
	microKernel = microKernelGeneric
	defer func() { microKernel = saved }()
	r := rand.New(rand.NewPCG(21, 22))
	for _, sh := range [][3]int{{1, 1, 1}, {5, 7, 3}, {MR + 1, NR + 1, KC + 1}, {2*MC + 1, 3*NR + 1, 33}, {64, 100, 2*KC + 3}} {
		m, n, k := sh[0], sh[1], sh[2]
		a := randMat(r, m*k)
		b := randMat(r, k*n)
		c0 := randMat(r, m*n)
		want := oracle(m, n, k, 1, a, k, b, n, 0.5, c0, n)
		got := append([]float32(nil), c0...)
		Sgemm(m, n, k, 1, a, k, b, n, 0.5, got, n)
		if e := maxRelErr(got, n, n, want); e > f32Tol(k) {
			t.Fatalf("generic fallback m=%d n=%d k=%d: rel err %g", m, n, k, e)
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
			microKernel(kc, ap, bp, c, NR, true, nil)
		}
		b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
	})
	b.Run("generic", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			microKernelGeneric(kc, ap, bp, c, NR, true, nil)
		}
		b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
	})
}
