//go:build !amd64

package gemm

func bf16Row(dst []uint16, src []float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = F32ToBF16(src[i])
	}
}

func bkernelBF16Rows(kg int64, rows *[8]*uint16, bp *uint16, ct *float32) {
	panic("gemm: bf16 rows kernel not available")
}
