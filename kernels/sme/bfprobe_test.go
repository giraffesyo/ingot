package sme_test

import (
	"testing"
	"time"

	"github.com/giraffesyo/ingot/kernels/sme"
)

func TestBF16PeakProbe(t *testing.T) {
	if !sme.Available() {
		t.Skip("no SME")
	}
	src := make([]uint16, 4*sme.SVL()/2)
	sme.ProbeBF16Peak(1000, src) // warm
	const iters = 2_000_000
	t0 := time.Now()
	sme.ProbeBF16Peak(iters, src)
	dt := time.Since(t0).Seconds()
	flops := float64(iters) * 4 * 1024 // 4 BFMOPA × (16×16 f32 × 2-deep × 2)
	t.Logf("BFMOPA peak: %.2f TFLOPS/core", flops/dt/1e12)
}
