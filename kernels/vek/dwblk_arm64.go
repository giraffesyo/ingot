//go:build arm64

package vek

//go:noescape
func dwblk8s1_asm(dst, src, w []float32, ncols, wp int)

// DwBlk8S1 computes one output row of a 3x3 stride-1 depthwise conv in
// channel-blocked layout (see dwblk.go for the reference).
func DwBlk8S1(dst, src, w []float32, ncols, wp int) {
	dwblk8s1_asm(dst, src, w, ncols, wp)
}
