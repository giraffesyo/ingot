//go:build arm64

package gemm

import "golang.org/x/sys/cpu"

// Three kernel tiers, best available first:
//
//	FEAT_I8MM (Armv8.6: Apple M2+, Neoverse V1/N2+) — MMLA kernels,
//	  group-major B, 2×2-block C.
//	FEAT_DotProd (Armv8.2: Apple M1, Graviton2, most phone cores) — SDOT
//	  kernels, quad-major B and row-major C (the amd64 VNNI layouts).
//	otherwise — portable Go.
//
// Detection via x/sys/cpu (hwcap on linux, sysctl on darwin). The layout
// flags travel with the kernel choice.
var (
	useSDOT = !cpu.ARM64.HasI8MM && cpu.ARM64.HasASIMDDP

	qkernel = func() func(int64, *uint8, *int8, *int32) {
		switch {
		case cpu.ARM64.HasI8MM:
			return qkernelU8S8
		case useSDOT:
			return qkernelU8S8SDOT
		}
		return qkernelGeneric
	}()
	qkernelS8 = func() func(int64, *int8, *int8, *int32) {
		switch {
		case cpu.ARM64.HasI8MM:
			return qkernelS8S8
		case useSDOT:
			return qkernelS8S8SDOT
		}
		return qkernelS8Generic
	}()
	qpackQuad   = useSDOT
	qctRowMajor = useSDOT
)

//go:noescape
func qkernelU8S8(kg int64, ap *uint8, bp *int8, ct *int32)

//go:noescape
func qkernelS8S8(kg int64, ap *int8, bp *int8, ct *int32)

//go:noescape
func qkernelU8S8SDOT(kg int64, ap *uint8, bp *int8, ct *int32)

//go:noescape
func qkernelS8S8SDOT(kg int64, ap *int8, bp *int8, ct *int32)
