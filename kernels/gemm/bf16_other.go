//go:build !arm64

package gemm

const hasBFMMLA = false

func bkernelBF16(kg int64, ap *uint16, bp *uint16, ct *float32) {
	panic("gemm: bf16 kernel not available")
}
