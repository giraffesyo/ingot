//go:build arm64 && !darwin

package gemm

// hasBFMMLA: BF16 detection is only wired up for darwin so far
// (linux needs HWCAP2_BF16, untested without hardware).
const hasBFMMLA = false

const (
	hasBF16DP   = false
	bPairB      = false
	bctRowMajor = false
)
