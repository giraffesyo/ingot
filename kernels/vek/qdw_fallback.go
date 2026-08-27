//go:build !arm64 && !amd64

package vek

// QDwRowS1: scalar fallback (full width).
func QDwRowS1(acc []int32, src, wp []int16, ncols, W, KH, KW int) {
	qdwTail(acc, src, wp, 0, ncols, W, KH, KW)
}
