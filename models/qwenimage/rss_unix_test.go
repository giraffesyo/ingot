//go:build unix

package qwenimage

import (
	"runtime"
	"syscall"
)

// peakRSSGB is the process's maximum resident set size.
func peakRSSGB() float64 {
	var ru syscall.Rusage
	if syscall.Getrusage(syscall.RUSAGE_SELF, &ru) != nil {
		return 0
	}
	if runtime.GOOS == "darwin" { // bytes on darwin, KiB elsewhere
		return float64(ru.Maxrss) / 1e9
	}
	return float64(ru.Maxrss) * 1024 / 1e9
}
