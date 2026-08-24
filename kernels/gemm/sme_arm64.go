//go:build arm64

package gemm

import (
	"os"

	"github.com/giraffesyo/ingot/kernels/sme"
)

// smeOn: opt-in SME dispatch (OCR_GEMM_KERNEL=sme on SME2 hardware). When on,
// eligible shapes (no transpose, alpha 1, beta 0, and big enough that 32-row
// ZA panels don't waste the unit) run on the matrix coprocessor; everything
// else stays on the NEON path. ActiveKernel reports the selection.
var smeOn = func() bool {
	if os.Getenv("OCR_GEMM_KERNEL") == "sme" && sme.Available() {
		ActiveKernel = "neon+sme"
		return true
	}
	return false
}()

// ActiveKernel names the selected kernel set ("neon" or "neon+sme").
var ActiveKernel = "neon"

// smeEligible: shapes where the ZA kernel beats NEON (measured: small M loses
// — partial 32-row panels waste FMOPA lanes; tiny k never amortises packing).
func smeEligible(m, n, k int) bool {
	return smeOn && m >= 32 && k >= 48 && n >= 16
}

func smeSgemm(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, parallel bool) {
	pa := sme.PackA(m, k, a, lda)
	sme.SgemmPacked(pa, n, b, ldb, c, ldc, parallel)
}

func smePackA(m, k int, a []float32, lda int) *sme.PackedA { return sme.PackA(m, k, a, lda) }

func smeSgemmPacked(pa *sme.PackedA, n int, b []float32, ldb int, c []float32, ldc int, parallel bool) {
	sme.SgemmPacked(pa, n, b, ldb, c, ldc, parallel)
}
