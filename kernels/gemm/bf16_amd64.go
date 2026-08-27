//go:build amd64

package gemm

import "golang.org/x/sys/cpu"

// hasBFMMLA (the arm64 kernel) is never available here; the amd64 bf16
// kernel is VDPBF16PS-based with its own layouts (bPairB, bctRowMajor).
const hasBFMMLA = false

// hasBF16DP: AVX-512 BF16 dot-product kernel (Zen 4/5, Cooper Lake+).
// Measured on Zen 5: 1.45x the f32 FMA peak per core.
var hasBF16DP = cpu.X86.HasAVX512F && cpu.X86.HasAVX512VL && cpu.X86.HasAVX512BF16

// bPairB / bctRowMajor: the DP kernel wants B packed pair-major
// ([g][2p][12j][2] bf16, same byte geometry as the VNNI quads) and writes C
// row-major; the MMLA kernel uses group-major B and 2x2-block C.
var (
	bPairB      = hasBF16DP
	bctRowMajor = hasBF16DP
)

func bkernelBF16(kg int64, ap *uint16, bp *uint16, ct *float32) {
	bkernelBF16DP(kg, ap, bp, ct)
}

//go:noescape
func bkernelBF16DP(kg int64, ap *uint16, bp *uint16, ct *float32)
