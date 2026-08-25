//go:build !arm64

package gemm

const hasQpackAsm = false

func qpackb(dst *int8, src *int8, ldb int64, groups int64) {
	panic("gemm: qpackb asm not available")
}
