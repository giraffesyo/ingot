package sme_test

import (
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/kernels/sme"
)

func TestQgemmPackedS8VsRef(t *testing.T) {
	if !sme.Available() {
		t.Skip("SME not available")
	}
	r := rand.New(rand.NewPCG(51, 52))
	for _, sh := range [][3]int{{1, 1, 1}, {32, 32, 4}, {32, 32, 64}, {33, 65, 127}, {31, 33, 513}, {100, 50, 480}, {24, 320, 864}, {480, 40, 480}} {
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
		want := make([]int32, m*n)
		sme.QgemmPackedS8(sme.QPackA(m, k, a, k), n, b, n, got, n, true)
		au := make([]uint8, 0) // QgemmRef wants u8 A; build a s8 reference inline instead
		_ = au
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var s int32
				for p := 0; p < k; p++ {
					s += int32(a[i*k+p]) * int32(b[p*n+j])
				}
				want[i*n+j] = s
			}
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("m=%d n=%d k=%d c[%d]=%d want %d", m, n, k, i, got[i], want[i])
			}
		}
	}
	_ = gemm.QgemmRef
}

func BenchmarkQgemmSME(b *testing.B) {
	if !sme.Available() {
		b.Skip("SME not available")
	}
	r := rand.New(rand.NewPCG(5, 6))
	for _, sh := range []struct {
		name    string
		m, n, k int
	}{
		{"sq512", 512, 512, 512},
		{"det_m24_n25600_k864", 24, 25600, 864},
		{"rec_m480_n240_k480", 480, 240, 480},
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
		ops := 2 * float64(sh.m) * float64(sh.n) * float64(sh.k)
		paS := sme.QPackA(sh.m, sh.k, a, sh.k)
		paN := gemm.QPackA(sh.m, sh.k, a, sh.k)
		b.Run(sh.name+"/sme", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sme.QgemmPackedS8(paS, sh.n, bm, sh.n, c, sh.n, true)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GOPS")
		})
		// Note: gemm.QgemmPackedS8 itself dispatches to SME under the auto/forced
		// policy, so this row measures the dispatcher, not raw NEON — force
		// OCR_GEMM_KERNEL=neon to compare kernels.
		b.Run(sh.name+"/gemm-dispatch", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				gemm.QgemmPackedS8(paN, sh.n, bm, sh.n, c, sh.n, true)
			}
			b.ReportMetric(ops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GOPS")
		})
	}
}
