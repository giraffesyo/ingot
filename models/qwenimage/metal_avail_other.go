//go:build !(darwin && arm64)

package qwenimage

func metalAvailable() bool { return false }
