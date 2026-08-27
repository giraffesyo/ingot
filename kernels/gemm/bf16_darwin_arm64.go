//go:build arm64 && darwin

package gemm

import "golang.org/x/sys/unix"

// hasBFMMLA: FEAT_BF16 (Apple M2+). x/sys/cpu has no arm64 BF16 field yet,
// so probe the sysctl directly, like kernels/sme does for SME.
var hasBFMMLA = func() bool {
	v, err := unix.SysctlUint32("hw.optional.arm.FEAT_BF16")
	return err == nil && v != 0
}()

const (
	hasBF16DP   = false
	bPairB      = false
	bctRowMajor = false
)
