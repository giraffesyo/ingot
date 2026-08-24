package sme

import (
	"math/rand/v2"
	"testing"
)

func TestOuterK(t *testing.T) {
	if !Available() {
		t.Skip("SME not available")
	}
	lanes := SVL() / 4
	t.Logf("SVL = %d bytes (%d f32 lanes)", SVL(), lanes)
	r := rand.New(rand.NewPCG(1, 2))
	for _, k := range []int{1, 2, 7, 64, 513} {
		a := make([]float32, k*lanes)
		b := make([]float32, k*lanes)
		for i := range a {
			a[i] = r.Float32()*2 - 1
		}
		for i := range b {
			b[i] = r.Float32()*2 - 1
		}
		out := make([]float32, lanes*lanes)
		OuterK(a, b, k, out)
		for i := 0; i < lanes; i++ {
			for j := 0; j < lanes; j++ {
				var want float64
				for p := 0; p < k; p++ {
					want += float64(a[p*lanes+i]) * float64(b[p*lanes+j])
				}
				got := float64(out[i*lanes+j])
				d := want - got
				if d < 0 {
					d = -d
				}
				if d > 1e-4*(1+float64(k)) {
					t.Fatalf("k=%d out[%d][%d] = %g want %g", k, i, j, got, want)
				}
			}
		}
	}
}

func BenchmarkFMOPA(b *testing.B) {
	if !Available() {
		b.Skip("SME not available")
	}
	lanes := SVL() / 4
	src := make([]float32, 4*lanes)
	for i := range src {
		src[i] = 1
	}
	flopsPerTile := 2 * float64(lanes) * float64(lanes)
	b.Run("peak", func(b *testing.B) {
		const inner = 4096
		for i := 0; i < b.N; i++ {
			ProbePeak(inner, src)
		}
		b.ReportMetric(4*inner*flopsPerTile*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
	})
	for _, v := range []int64{1, 2, 16} {
		fm := v
		b.Run(map[int64]string{1: "1tile-chain", 2: "2tiles", 16: "16x-4pertile"}[v], func(b *testing.B) {
			const inner = 4096
			for i := 0; i < b.N; i++ {
				ProbeN(inner, src, fm)
			}
			nf := float64(fm)
			b.ReportMetric(nf*inner*flopsPerTile*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			b.ReportMetric(b.Elapsed().Seconds()*1e9/(nf*inner*float64(b.N)), "ns/fmopa")
		})
	}
	b.Run("with-loads", func(b *testing.B) {
		const inner = 4096
		for i := 0; i < b.N; i++ {
			ProbeLoad(inner, src, src)
		}
		b.ReportMetric(4*inner*flopsPerTile*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
	})
}

func TestSgemmVsRef(t *testing.T) {
	if !Available() {
		t.Skip("SME not available")
	}
	r := rand.New(rand.NewPCG(3, 4))
	for _, sh := range [][3]int{{32, 32, 1}, {32, 32, 64}, {64, 64, 7}, {5, 7, 3}, {33, 65, 127}, {31, 33, 512}, {100, 50, 480}, {24, 320, 864}, {256, 256, 256}} {
		m, n, k := sh[0], sh[1], sh[2]
		a := make([]float32, m*k)
		b := make([]float32, k*n)
		for i := range a {
			a[i] = r.Float32()*2 - 1
		}
		for i := range b {
			b[i] = r.Float32()*2 - 1
		}
		c := make([]float32, m*n)
		Sgemm(m, n, k, a, k, b, n, c, n)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var want float64
				for p := 0; p < k; p++ {
					want += float64(a[i*k+p]) * float64(b[p*n+j])
				}
				got := float64(c[i*n+j])
				d := want - got
				if d < 0 {
					d = -d
				}
				scale := 1.0
				if want < 0 {
					scale = -want + 1
				} else {
					scale = want + 1
				}
				if d > 2e-6*float64(k)*scale/10+1e-5 {
					t.Fatalf("m=%d n=%d k=%d c[%d][%d] = %g want %g", m, n, k, i, j, got, want)
				}
			}
		}
	}
}
