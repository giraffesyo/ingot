//go:build darwin && arm64

package qwenimage

import "github.com/giraffesyo/ingot/kernels/metal"

func metalAvailable() bool { return metal.Available() }
