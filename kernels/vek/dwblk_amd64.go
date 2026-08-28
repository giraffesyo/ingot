//go:build amd64

package vek

//go:noescape
func dwblk8s1_asm(dst, src, w []float32, ncols, wp int)

// DwBlk8S1 computes one output row of a 3x3 stride-1 depthwise conv in
// channel-blocked layout: dst[j][c8] = Σ taps w[kh*3+kw][c8] · src rows.
// src points at the first padded row for this output row ([3][wp][8] window);
// w holds 9 tap vectors of 8 channels.
func DwBlk8S1(dst, src, w []float32, ncols, wp int) {
	dwblk8s1_asm(dst, src, w, ncols, wp)
}
