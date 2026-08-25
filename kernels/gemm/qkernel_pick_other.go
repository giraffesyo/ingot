//go:build !arm64 && !amd64

package gemm

// qkernel / qkernelS8: portable fallbacks.
var (
	qkernel     = qkernelGeneric
	qkernelS8   = qkernelS8Generic
	qpackQuad   = false
	qctRowMajor = false
)
