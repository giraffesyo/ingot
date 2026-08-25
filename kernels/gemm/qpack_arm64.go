//go:build arm64

package gemm

// zip-based B-panel pack (baseline NEON, no I8MM needed).
const hasQpackAsm = true

//go:noescape
func qpackb(dst *int8, src *int8, ldb int64, groups int64)

//go:noescape
func qscatter(ct *int32, c *int32, ldc int64)

//go:noescape
func qpackbq(dst *int8, src *int8, ldb int64, groups int64)
