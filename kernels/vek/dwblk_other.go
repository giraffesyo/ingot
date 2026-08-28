//go:build !amd64 && !arm64

package vek

// DwBlk8S1: portable channel-blocked depthwise row (see amd64 asm).
func DwBlk8S1(dst, src, w []float32, ncols, wp int) {
	dwBlk8Ref(dst, src, w, ncols, wp, 3, 1)
}

// DwBlk8: portable KxK stride-S channel-blocked depthwise row.
func DwBlk8(dst, src, w []float32, ncols, wp, K, S int) {
	dwBlk8Ref(dst, src, w, ncols, wp, K, S)
}
