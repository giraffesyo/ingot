//go:build arm64

package gemm

import (
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/sme"
)

// hybridSgemm prototype: columns [0, nSme) on the SME unit via cSme coarse
// serial tasks, columns [nSme, n) on the NEON small-M sweep — one par region,
// so both compute engines run at once.
func hybridSgemm(pa *PackedA, ps *sme.PackedA, n int, b []float32, ldb int, c []float32, ldc int, cSme int, frac float64) {
	m, k := pa.m, pa.k
	nSme := int(frac*float64(n)) &^ 31
	if nSme > n {
		nSme = n
	}
	// NEON side: set up a ctx for columns [nSme, n).
	nNeon := n - nSme
	g := getCtx()
	defer putCtx(g)
	g.alpha, g.ldb, g.ldc = 1, ldb, ldc
	g.transA, g.transB = false, false
	g.mc, g.nc, g.kc = m, nNeon, k
	g.mPanels = (m + MR - 1) / MR
	g.nPanels = (nNeon + NR - 1) / NR
	g.acc = false
	g.bsrc, g.cblk = b[nSme:], c[nSme:]
	saved := g.asm
	g.asm = pa.data
	defer func() { g.asm = saved }()

	colsPer := (nSme/32 + cSme - 1) / cSme * 32
	total := cSme + g.nPanels
	grain := max(1, macroTaskMACs/(NR*m*k))
	par.For(total, grain, func(t, w int) {
		if t < cSme {
			j0 := t * colsPer
			if j0 >= nSme {
				return
			}
			cols := min(colsPer, nSme-j0)
			sme.SgemmPacked(ps, cols, b[j0:], ldb, c[j0:], ldc, false)
			return
		}
		g.smallMTask(t-cSme, w)
	})
}

func BenchmarkHybrid(b *testing.B) {
	if !sme.Available() {
		b.Skip("SME not available")
	}
	r := rand.New(rand.NewPCG(5, 6))
	for _, sh := range []struct {
		name    string
		m, n, k int
	}{
		{"sq1024", 1024, 1024, 1024},
		{"rec_m240_n960_k240", 240, 960, 240},
		{"pw_m96_n25600_k96", 96, 25600, 96},
		{"conv_m64_n16384_k576", 64, 16384, 576},
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
		if sh.m > MC || (sh.m+MR-1)/MR*MR*((sh.k+KC-1)/KC)*KC > smallMMaxA {
			continue // prototype uses the small-M sweep only
		}
		pa := PackA(false, sh.m, sh.k, a, sh.k)
		ps := sme.PackA(sh.m, sh.k, a, sh.k)
		report := func(b *testing.B) {
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		}
		b.Run(sh.name+"/neon", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				SgemmPackedA(pa, sh.n, bm, sh.n, 0, c, sh.n, true)
			}
			report(b)
		})
		b.Run(sh.name+"/sme-par", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sme.SgemmPacked(ps, sh.n, bm, sh.n, c, sh.n, true)
			}
			report(b)
		})
		for _, cfg := range []struct {
			c    int
			frac float64
		}{{4, 0.45}, {4, 0.55}, {4, 0.65}, {6, 0.55}, {6, 0.65}, {3, 0.45}} {
			b.Run(sh.name+"/hyb-c"+itoa(cfg.c)+"-f"+itoa(int(cfg.frac*100)), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					hybridSgemm(pa, ps, sh.n, bm, sh.n, c, sh.n, cfg.c, cfg.frac)
				}
				report(b)
			})
		}
	}
}

func itoa(v int) string {
	s := ""
	if v == 0 {
		return "0"
	}
	for v > 0 {
		s = string(rune('0'+v%10)) + s
		v /= 10
	}
	return s
}
