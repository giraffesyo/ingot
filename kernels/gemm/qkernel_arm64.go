//go:build arm64

package gemm

// qkernel is the active u8×s8 micro-kernel (USDOT; all shipped arm64 targets
// have FEAT_I8MM — Apple M1+ and Armv8.6+ servers. Revisit with runtime
// detection if pre-8.6 arm64 ever matters).
var qkernel = qkernelU8S8

//go:noescape
func qkernelU8S8(kg int64, ap *uint8, bp *int8, ct *int32)

// qkernelS8 is the active s8×s8 micro-kernel (SMMLA).
var qkernelS8 = qkernelS8S8

//go:noescape
func qkernelS8S8(kg int64, ap *int8, bp *int8, ct *int32)
