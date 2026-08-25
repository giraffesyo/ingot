//go:build !arm64

package gemm

const hasQpackAsm = false

func qpackb(dst *int8, src *int8, ldb int64, groups int64) {
	panic("gemm: qpackb asm not available")
}

func qscatter(ct *int32, c *int32, ldc int64) {
	panic("gemm: qscatter asm not available")
}

func qpackbq(dst *int8, src *int8, ldb int64, groups int64) {
	panic("gemm: qpackbq asm not available")
}
