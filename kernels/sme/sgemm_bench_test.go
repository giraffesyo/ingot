package sme_test

import (
	"math/rand/v2"
	"runtime"
	"sync"
	"testing"

	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/kernels/sme"
)

func BenchmarkSgemm(b *testing.B) {
	if !sme.Available() {
		b.Skip("SME not available")
	}
	r := rand.New(rand.NewPCG(5, 6))
	for _, sh := range []struct {
		name    string
		m, n, k int
	}{
		{"sq256", 256, 256, 256},
		{"sq512", 512, 512, 512},
		{"sq1024", 1024, 1024, 1024},
		{"rec_m480_n240_k480", 480, 240, 480},
		{"conv3x3_m24_n25600_k864", 24, 25600, 864},
		{"pw_m96_n25600_k96", 96, 25600, 96},
	} {
		a := make([]float32, sh.m*sh.k)
		bm := make([]float32, sh.k*sh.n)
		for i := range a {
			a[i] = r.Float32()
		}
		for i := range bm {
			bm[i] = r.Float32()
		}
		c := make([]float32, sh.m*sh.n)
		flops := 2 * float64(sh.m) * float64(sh.n) * float64(sh.k)
		b.Run(sh.name+"/sme", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sme.Sgemm(sh.m, sh.n, sh.k, a, sh.k, bm, sh.n, c, sh.n)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
		b.Run(sh.name+"/sme-prepacked", func(b *testing.B) {
			pa := sme.PackA(sh.m, sh.k, a, sh.k)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sme.SgemmPacked(pa, sh.n, bm, sh.n, c, sh.n, true)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
		b.Run(sh.name+"/neon", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				gemm.Sgemm(sh.m, sh.n, sh.k, 1, a, sh.k, bm, sh.n, 0, c, sh.n)
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

// BenchmarkFMOPAContention: aggregate FMOPA throughput with G goroutines
// hammering the (per-cluster, shared) matrix units simultaneously.
func BenchmarkFMOPAContention(b *testing.B) {
	if !sme.Available() {
		b.Skip("SME not available")
	}
	lanes := sme.SVL() / 4
	flopsPerIter := 4 * 2 * float64(lanes*lanes)
	for _, g := range []int{1, 2, 4, 6, 12, 18} {
		if g > runtime.GOMAXPROCS(0) {
			continue
		}
		b.Run("g"+itoa(g), func(b *testing.B) {
			const inner = 4096
			var wg sync.WaitGroup
			for i := 0; i < b.N; i++ {
				wg.Add(g)
				for t := 0; t < g; t++ {
					go func() {
						src := make([]float32, 4*lanes)
						for j := range src {
							src[j] = 1
						}
						sme.ProbePeak(inner, src)
						wg.Done()
					}()
				}
				wg.Wait()
			}
			b.ReportMetric(float64(g)*inner*flopsPerIter*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

func itoa(v int) string {
	if v >= 10 {
		return string(rune('0'+v/10)) + string(rune('0'+v%10))
	}
	return string(rune('0' + v))
}
