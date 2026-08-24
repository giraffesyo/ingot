package sme

import "golang.org/x/sys/unix"

// Available reports whether the CPU exposes SME2 with f32 outer products.
func Available() bool {
	v, err := unix.SysctlUint32("hw.optional.arm.FEAT_SME2")
	if err != nil || v == 0 {
		return false
	}
	f, err := unix.SysctlUint32("hw.optional.arm.SME_F32F32")
	return err == nil && f != 0 && svl() == 64 // kernel geometry assumes 16-lane tiles
}
