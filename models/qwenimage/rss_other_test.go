//go:build !unix

package qwenimage

// peakRSSGB is not measured off unix.
func peakRSSGB() float64 { return 0 }
