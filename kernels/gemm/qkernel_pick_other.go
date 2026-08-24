//go:build !arm64

package gemm

// qkernel / qkernelS8: portable fallbacks (amd64 VNNI kernels are TODO).
var (
	qkernel   = qkernelGeneric
	qkernelS8 = qkernelS8Generic
)
