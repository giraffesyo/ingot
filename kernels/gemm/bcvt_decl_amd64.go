//go:build amd64

package gemm

//go:noescape
func bcvt32(dst *uint16, src *float32, n int64)

// bf16Row converts src to bf16 into dst (hardware RNE when available).
func bf16Row(dst []uint16, src []float32) {
	n := min(len(dst), len(src))
	m := n &^ 31
	if hasBF16DP && m > 0 {
		bcvt32(&dst[0], &src[0], int64(m))
	} else {
		m = 0
	}
	for i := m; i < n; i++ {
		dst[i] = F32ToBF16(src[i])
	}
}

//go:noescape
func bkernelBF16Rows(kg int64, rows *[8]*uint16, bp *uint16, ct *float32)
