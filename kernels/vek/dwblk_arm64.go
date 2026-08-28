//go:build arm64

package vek

//go:noescape
func dwblk8k3s1_asm(dst, src, w []float32, ncols, wp int)

//go:noescape
func dwblk8k3s2_asm(dst, src, w []float32, ncols, wp int)

//go:noescape
func dwblk8k5s1_asm(dst, src, w []float32, ncols, wp int)

//go:noescape
func dwblk8k5s2_asm(dst, src, w []float32, ncols, wp int)

// DwBlk8S1 computes one output row of a 3x3 stride-1 depthwise conv in
// channel-blocked layout (see dwblk.go for the reference semantics).
func DwBlk8S1(dst, src, w []float32, ncols, wp int) {
	dwblk8k3s1_asm(dst, src, w, ncols, wp)
}

// DwBlk8 computes one output row of a KxK stride-S depthwise conv in
// channel-blocked layout. K in {3,5}, S in {1,2}; other shapes take the
// scalar reference.
func DwBlk8(dst, src, w []float32, ncols, wp, K, S int) {
	switch {
	case K == 3 && S == 1:
		dwblk8k3s1_asm(dst, src, w, ncols, wp)
	case K == 3 && S == 2:
		dwblk8k3s2_asm(dst, src, w, ncols, wp)
	case K == 5 && S == 1:
		dwblk8k5s1_asm(dst, src, w, ncols, wp)
	case K == 5 && S == 2:
		dwblk8k5s2_asm(dst, src, w, ncols, wp)
	default:
		dwBlk8Ref(dst, src, w, ncols, wp, K, S)
	}
}
