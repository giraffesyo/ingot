//go:build amd64

package gemm

import "golang.org/x/sys/cpu"

//go:noescape
func qkernelU8S8VNNI(kg int64, ap *uint8, bp *int8, ct *int32)

//go:noescape
func qkernelS8S8VNNI(kg int64, ap *int8, bp *int8, ct *int32)

// The VNNI kernels need AVX-512 F+VL+VNNI (Cascade Lake / Ice Lake / Zen 4+).
// Everything older takes the portable kernels. The layout flags travel with
// the kernel choice: VNNI consumes a quad-major B panel and writes C
// row-major (the arm64 MMLA kernels use group-major B and 2×2-block C).
var (
	useVNNI = cpu.X86.HasAVX512F && cpu.X86.HasAVX512VL && cpu.X86.HasAVX512VNNI

	qkernel = func() func(int64, *uint8, *int8, *int32) {
		if useVNNI {
			return qkernelU8S8VNNI
		}
		return qkernelGeneric
	}()
	qkernelS8 = func() func(int64, *int8, *int8, *int32) {
		if useVNNI {
			return qkernelS8S8VNNI
		}
		return qkernelS8Generic
	}()
	qpackQuad   = useVNNI
	qctRowMajor = useVNNI
)
