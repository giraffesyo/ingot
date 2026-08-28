//go:build !arm64 && !amd64

package gemm

// microKernel is the active micro-kernel for this architecture.
var microKernel = microKernelGeneric

func pairKernel() bool { return false }

func microKernel2AVX512(kc int, ap, bp0, bp1 []float32, c []float32, ldc int, accumulate bool) {
	panic("gemm: paired kernel not available")
}
