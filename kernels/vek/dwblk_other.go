//go:build !amd64 && !arm64

package vek

// DwBlk8S1: portable channel-blocked depthwise row (see amd64 asm).
func DwBlk8S1(dst, src, w []float32, ncols, wp int) {
	dwBlk8Ref(dst, src, w, ncols, wp)
}
