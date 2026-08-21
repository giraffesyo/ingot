//go:build !arm64

package gemm

// microKernel is the active micro-kernel for this architecture.
var microKernel = microKernelGeneric
