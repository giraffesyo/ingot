//go:build !(darwin && arm64)

package sme

// guard: no signal masking needed where SME is unavailable.
func guard(fn func()) { fn() }
