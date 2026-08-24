//go:build !arm64

package sme

// Available reports whether the CPU exposes SME2 (never on this platform).
func Available() bool { return false }

func svl() int64                                 { return 0 }
func probePeak(int64, *float32)                  {}
func probeLoad(int64, *float32, *float32)        {}
func outerK(*float32, *float32, int64, *float32) {}
