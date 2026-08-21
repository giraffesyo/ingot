package gemm

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

func randMat(r *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = r.Float32()*2 - 1
	}
	return x
}

// oracle computes C = alpha*A·B + beta*C with float64 accumulation.
func oracle(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) []float64 {
	out := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var acc float64
			for p := 0; p < k; p++ {
				acc += float64(a[i*lda+p]) * float64(b[p*ldb+j])
			}
			out[i*n+j] = float64(alpha)*acc + float64(beta)*float64(c[i*ldc+j])
		}
	}
	return out
}

// f32Tol is the allowed relative error for an f32 dot product of length k.
func f32Tol(k int) float64 { return 1e-5 * math.Sqrt(float64(max(k, 1))) }

func maxRelErr(got []float32, ldc, n int, want []float64) float64 {
	var worst float64
	for idx, w := range want {
		i, j := idx/n, idx%n
		d := math.Abs(float64(got[i*ldc+j]) - w)
		s := math.Max(math.Abs(w), 1)
		worst = math.Max(worst, d/s)
	}
	return worst
}

func TestSgemmMatchesRef(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	shapes := [][3]int{
		{1, 1, 1}, {3, 5, 7}, {4, 4, 4}, {5, 9, 3}, {MR, NR, KC}, {MR + 1, NR + 1, KC + 1},
		{MC, NC, KC}, {MC + 3, NC + 5, KC + 7}, {2*MC + 1, NC + NR + 1, KC + 3},
		{64, 1000, 576}, {1, 4096, 64}, {4096, 1, 64}, {300, 17, 1},
	}
	for _, sh := range shapes {
		m, n, k := sh[0], sh[1], sh[2]
		for _, cfg := range []struct{ alpha, beta float32 }{{1, 0}, {1, 1}, {0.5, 0}, {2, -0.5}} {
			t.Run(fmt.Sprintf("m=%d,n=%d,k=%d,a=%g,b=%g", m, n, k, cfg.alpha, cfg.beta), func(t *testing.T) {
				a := randMat(r, m*k)
				b := randMat(r, k*n)
				c0 := randMat(r, m*n)
				want := oracle(m, n, k, cfg.alpha, a, k, b, n, cfg.beta, c0, n)
				got := append([]float32(nil), c0...)
				Sgemm(m, n, k, cfg.alpha, a, k, b, n, cfg.beta, got, n)
				if e := maxRelErr(got, n, n, want); e > f32Tol(k) {
					t.Fatalf("max rel err %g > tol %g", e, f32Tol(k))
				}
				ref := append([]float32(nil), c0...)
				Ref(m, n, k, cfg.alpha, a, k, b, n, cfg.beta, ref, n)
				if e := maxRelErr(ref, n, n, want); e > f32Tol(k) {
					t.Fatalf("Ref max rel err %g > tol %g", e, f32Tol(k))
				}
			})
		}
	}
}

func TestSgemmStrided(t *testing.T) {
	// Submatrix views: lda/ldb/ldc larger than logical widths.
	r := rand.New(rand.NewPCG(3, 4))
	m, n, k := 37, 53, 29
	lda, ldb, ldc := k+5, n+3, n+9
	a := randMat(r, m*lda)
	b := randMat(r, k*ldb)
	c0 := randMat(r, m*ldc)
	want := oracle(m, n, k, 1, a, lda, b, ldb, 0.25, c0, ldc)
	got := append([]float32(nil), c0...)
	Sgemm(m, n, k, 1, a, lda, b, ldb, 0.25, got, ldc)
	if e := maxRelErr(got, ldc, n, want); e > f32Tol(k) {
		t.Fatalf("max rel err %g", e)
	}
}

var benchShapes = []struct {
	name    string
	m, n, k int
}{
	{"sq256", 256, 256, 256},
	{"sq512", 512, 512, 512},
	{"sq1024", 1024, 1024, 1024},
	{"sq2048", 2048, 2048, 2048},
	// conv-as-gemm (DBNet-ish): M=out_ch, N=H*W, K=in_ch*3*3
	{"conv_m64_n16384_k576", 64, 16384, 576},
	{"conv_m256_n1024_k2304", 256, 1024, 2304},
	// attention/linear (SVTR-ish): tokens × dim
	{"lin_m256_n768_k192", 256, 768, 192},
	{"attn_m256_n256_k64", 256, 256, 64},
}

func BenchmarkSgemm(b *testing.B) {
	r := rand.New(rand.NewPCG(5, 6))
	for _, sh := range benchShapes {
		a := randMat(r, sh.m*sh.k)
		bm := randMat(r, sh.k*sh.n)
		c := make([]float32, sh.m*sh.n)
		b.Run(sh.name, func(b *testing.B) {
			b.ReportAllocs()
			flops := 2 * float64(sh.m) * float64(sh.n) * float64(sh.k)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Sgemm(sh.m, sh.n, sh.k, 1, a, sh.k, bm, sh.n, 0, c, sh.n)
			}
			b.StopTimer()
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

func BenchmarkRef(b *testing.B) {
	r := rand.New(rand.NewPCG(5, 6))
	sh := benchShapes[0]
	a := randMat(r, sh.m*sh.k)
	bm := randMat(r, sh.k*sh.n)
	c := make([]float32, sh.m*sh.n)
	flops := 2 * float64(sh.m) * float64(sh.n) * float64(sh.k)
	for i := 0; i < b.N; i++ {
		Ref(sh.m, sh.n, sh.k, 1, a, sh.k, bm, sh.n, 0, c, sh.n)
	}
	b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
}
