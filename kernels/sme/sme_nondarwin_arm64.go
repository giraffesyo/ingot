//go:build arm64 && !darwin

package sme

// Available: SME detection is only wired up for darwin so far (Apple M4+).
// Linux arm64 would use HWCAP2_SME; add when there is hardware to test on.
func Available() bool { return false }
