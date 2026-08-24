package gemm

import (
	"math/rand/v2"
	"testing"
)

func TestQgemmU8S8VsRef(t *testing.T) {
	r := rand.New(rand.NewPCG(17, 18))
	for _, sh := range [][3]int{{1, 1, 1}, {3, 5, 2}, {8, 12, 4}, {9, 13, 5}, {16, 24, 64}, {31, 37, 129}, {64, 100, 300}, {24, 320, 864}, {128, 128, 47}} {
		m, n, k := sh[0], sh[1], sh[2]
		a := make([]uint8, m*k)
		b := make([]int8, k*n)
		for i := range a {
			a[i] = uint8(r.UintN(256))
		}
		for i := range b {
			b[i] = int8(r.UintN(256))
		}
		got := make([]int32, m*n)
		want := make([]int32, m*n)
		QgemmU8S8(m, n, k, a, k, b, n, got, n)
		QgemmRef(m, n, k, a, k, b, n, want, n)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("m=%d n=%d k=%d: c[%d]=%d want %d", m, n, k, i, got[i], want[i])
			}
		}
	}
}

func BenchmarkQgemmU8S8(b *testing.B) {
	r := rand.New(rand.NewPCG(5, 6))
	for _, sh := range []struct {
		name    string
		m, n, k int
	}{
		{"sq512", 512, 512, 512},
		{"conv_m64_n16384_k576", 64, 16384, 576},
		{"rec_m480_n240_k480", 480, 240, 480},
	} {
		a := make([]uint8, sh.m*sh.k)
		bm := make([]int8, sh.k*sh.n)
		for i := range a {
			a[i] = uint8(r.UintN(256))
		}
		for i := range bm {
			bm[i] = int8(r.UintN(256))
		}
		c := make([]int32, sh.m*sh.n)
		ops := 2 * float64(sh.m) * float64(sh.n) * float64(sh.k)
		b.Run(sh.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				QgemmU8S8(sh.m, sh.n, sh.k, a, sh.k, bm, sh.n, c, sh.n)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GOPS")
		})
	}
}

func TestQgemmPackedS8VsRef(t *testing.T) {
	r := rand.New(rand.NewPCG(19, 20))
	for _, sh := range [][3]int{{1, 1, 1}, {8, 12, 8}, {9, 13, 9}, {24, 100, 300}, {64, 37, 129}, {480, 33, 480}} {
		m, n, k := sh[0], sh[1], sh[2]
		a := make([]int8, m*k)
		b := make([]int8, k*n)
		for i := range a {
			a[i] = int8(r.UintN(256))
		}
		for i := range b {
			b[i] = int8(r.UintN(256))
		}
		got := make([]int32, m*n)
		QgemmPackedS8(QPackA(m, k, a, k), n, b, n, got, n, true)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var want int32
				for p := 0; p < k; p++ {
					want += int32(a[i*k+p]) * int32(b[p*n+j])
				}
				if got[i*n+j] != want {
					t.Fatalf("m=%d n=%d k=%d c[%d][%d]=%d want %d", m, n, k, i, j, got[i*n+j], want)
				}
			}
		}
	}
}

func BenchmarkQgemmPackedS8(b *testing.B) {
	r := rand.New(rand.NewPCG(5, 6))
	for _, sh := range []struct {
		name    string
		m, n, k int
	}{
		{"sq512", 512, 512, 512},
		{"conv_m64_n16384_k576", 64, 16384, 576},
		{"det_m24_n25600_k864", 24, 25600, 864},
	} {
		a := make([]int8, sh.m*sh.k)
		bm := make([]int8, sh.k*sh.n)
		for i := range a {
			a[i] = int8(r.UintN(256))
		}
		for i := range bm {
			bm[i] = int8(r.UintN(256))
		}
		c := make([]int32, sh.m*sh.n)
		pa := QPackA(sh.m, sh.k, a, sh.k)
		ops := 2 * float64(sh.m) * float64(sh.n) * float64(sh.k)
		b.Run(sh.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				QgemmPackedS8(pa, sh.n, bm, sh.n, c, sh.n, true)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GOPS")
		})
	}
}
