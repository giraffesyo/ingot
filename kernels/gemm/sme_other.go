//go:build !arm64

package gemm

import "github.com/giraffesyo/ingot/kernels/sme"

const smeOn = false

func smeEligible(m, n, k int) bool { return false }

func smeSgemm(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, parallel bool) {
}

func smePackA(m, k int, a []float32, lda int) *sme.PackedA { return nil }

func smeSgemmPacked(pa *sme.PackedA, n int, b []float32, ldb int, c []float32, ldc int, parallel bool) {
}

// PrefersSME reports whether the SME unit would take this GEMM (never here).
func PrefersSME(m, n, k int) bool { return false }

func qsmeEligible(m, k int) bool { return false }

func qsmePackA(m, k int, a []int8, lda int) *sme.QPackedA { return nil }

func qsmeGemm(pa *sme.QPackedA, n int, b []int8, ldb int, c []int32, ldc int, parallel bool) {}
