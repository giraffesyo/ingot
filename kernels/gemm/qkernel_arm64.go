//go:build arm64

package gemm

import "golang.org/x/sys/cpu"

// The MMLA kernels need FEAT_I8MM (Armv8.6: Apple M2+, Neoverse V1/N2+).
// Apple M1 and Graviton2-class cores have only dot products — they take the
// portable kernel until an SDOT mid-tier lands (int8 phase 2). Detection via
// x/sys/cpu (hwcap on linux, sysctl on darwin).
// arm64 keeps the MMLA layouts: group-major B, 2×2-block C.
var (
	qpackQuad   = false
	qctRowMajor = false
)

var (
	qkernel = func() func(int64, *uint8, *int8, *int32) {
		if cpu.ARM64.HasI8MM {
			return qkernelU8S8
		}
		return qkernelGeneric
	}()
	qkernelS8 = func() func(int64, *int8, *int8, *int32) {
		if cpu.ARM64.HasI8MM {
			return qkernelS8S8
		}
		return qkernelS8Generic
	}()
)

//go:noescape
func qkernelU8S8(kg int64, ap *uint8, bp *int8, ct *int32)

//go:noescape
func qkernelS8S8(kg int64, ap *int8, bp *int8, ct *int32)
