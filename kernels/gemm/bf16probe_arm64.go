//go:build arm64

package gemm

//go:noescape
func bfdotProbe(iters int64, src *uint16)
