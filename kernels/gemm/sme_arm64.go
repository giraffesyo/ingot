//go:build arm64

package gemm

import (
	"os"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/sme"
)

// SME dispatch policy. Measured (docs/PERF.md): the ZA kernel is a ~2× win for
// eligible shapes when the process runs single-threaded (rec_320 1T 19.8 →
// 8.8 ms), but a mild loss at full parallelism — the matrix units are shared
// per cluster, so SME tasks contend while the NEON pipelines sit idle.
//
//	OCR_GEMM_KERNEL unset: auto — SME for eligible shapes when the worker
//	                       pool is single-threaded (GOMAXPROCS=1), else NEON.
//	OCR_GEMM_KERNEL=sme:   force SME for eligible shapes at any parallelism.
//	OCR_GEMM_KERNEL=neon:  never dispatch to SME.
var smeMode = func() int {
	switch os.Getenv("OCR_GEMM_KERNEL") {
	case "sme":
		if sme.Available() {
			ActiveKernel = "neon+sme(forced)"
			return 1
		}
	case "neon", "generic":
		return -1
	default:
		if sme.Available() && par.Workers() == 1 {
			ActiveKernel = "neon+sme(auto-1T)"
			return 2
		}
	}
	return -1
}()

// ActiveKernel names the selected kernel set.
var ActiveKernel = "neon"

// smeEligible: shapes where the ZA kernel beats NEON (measured: small M loses
// — below 24 rows the wasted FMOPA lanes outweigh the unit's speed, measured on the det head; tiny k never amortises packing).
func smeEligible(m, n, k int) bool {
	return smeMode > 0 && m >= 24 && k >= 48 && n >= 16
}

func smeSgemm(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, parallel bool) {
	pa := sme.PackA(m, k, a, lda)
	sme.SgemmPacked(pa, n, b, ldb, c, ldc, parallel)
}

func smePackA(m, k int, a []float32, lda int) *sme.PackedA { return sme.PackA(m, k, a, lda) }

func smeSgemmPacked(pa *sme.PackedA, n int, b []float32, ldb int, c []float32, ldc int, parallel bool) {
	sme.SgemmPacked(pa, n, b, ldb, c, ldc, parallel)
}

// PrefersSME reports whether the current dispatch policy would send an
// m×n×k GEMM to the SME unit — used by higher-level fusions (Winograd) to
// yield: measured 1T order is SME > Winograd > im2col.
func PrefersSME(m, n, k int) bool { return smeEligible(m, n, k) }

// int8 on the SME unit follows the same policy: measured 3.4-3.5× the NEON
// int8 GEMM single-threaded (sq512 1316 vs 374 GOPS), contended at MT;
// small M wastes the 32-row ZA panels.
func qsmeEligible(m, k int) bool { return smeMode > 0 && m >= 32 && k >= 64 }

func qsmePackA(m, k int, a []int8, lda int) *sme.QPackedA { return sme.QPackA(m, k, a, lda) }

func qsmeGemm(pa *sme.QPackedA, n int, b []int8, ldb int, c []int32, ldc int, parallel bool) {
	sme.QgemmPackedS8(pa, n, b, ldb, c, ldc, parallel)
}
