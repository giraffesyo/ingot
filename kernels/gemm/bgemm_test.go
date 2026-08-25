package gemm

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestBgemmVsRef: the kernel's f32 output vs a float64 oracle over
// bf16-quantized inputs. bf16 products are exact in f32, so the only error
// is f32 summation: rel tol 1e-6·√k.
func TestBgemmVsRef(t *testing.T) {
	if !HasBFMMLA() {
		t.Skip("no BFMMLA")
	}
	r := rand.New(rand.NewPCG(21, 22))
	for _, sh := range []struct{ m, n, k int }{
		{8, 12, 4}, {8, 12, 64}, {16, 24, 32}, {24, 12, 8},
		{7, 11, 5}, {9, 13, 17}, {64, 60, 100}, {1, 12, 4}, {33, 50, 129},
	} {
		a := make([]float32, sh.m*sh.k)
		b := make([]float32, sh.k*sh.n)
		for i := range a {
			a[i] = r.Float32()*4 - 2
		}
		for i := range b {
			b[i] = r.Float32()*4 - 2
		}
		c := make([]float32, sh.m*sh.n)
		pa := BPackA(sh.m, sh.k, a, sh.k)
		BgemmPacked(pa, sh.n, b, sh.n, c, sh.n, false)
		tol := 1e-6 * math.Sqrt(float64(sh.k))
		for i := 0; i < sh.m; i++ {
			for j := 0; j < sh.n; j++ {
				var want float64
				for p := 0; p < sh.k; p++ {
					av := float64(BF16ToF32(F32ToBF16(a[i*sh.k+p])))
					bv := float64(BF16ToF32(F32ToBF16(b[p*sh.n+j])))
					want += av * bv
				}
				got := float64(c[i*sh.n+j])
				if d := math.Abs(got - want); d > tol*math.Max(1, math.Abs(want)) {
					t.Fatalf("m%d n%d k%d: C[%d,%d]=%g want %g (Δ %g)", sh.m, sh.n, sh.k, i, j, got, want, d)
				}
			}
		}
	}
}

func BenchmarkBgemm(b *testing.B) {
	if !HasBFMMLA() {
		b.Skip("no BFMMLA")
	}
	r := rand.New(rand.NewPCG(23, 24))
	for _, sh := range []struct {
		name    string
		m, n, k int
	}{
		{"sq512", 512, 512, 512},
		{"conv_m64_n16384_k576", 64, 16384, 576},
		{"det_m24_n25600_k864", 24, 25600, 864},
	} {
		a := make([]float32, sh.m*sh.k)
		bm := make([]float32, sh.k*sh.n)
		for i := range a {
			a[i] = r.Float32() - 0.5
		}
		for i := range bm {
			bm[i] = r.Float32() - 0.5
		}
		c := make([]float32, sh.m*sh.n)
		pa := BPackA(sh.m, sh.k, a, sh.k)
		ops := 2 * float64(sh.m) * float64(sh.n) * float64(sh.k)
		b.Run(sh.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				BgemmPacked(pa, sh.n, bm, sh.n, c, sh.n, true)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}
