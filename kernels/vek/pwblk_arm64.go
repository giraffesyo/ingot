//go:build arm64

package vek

//go:noescape
func pwblk6x16_asm(dst0, dst1, x, w []float32, cin, xbstride int)

// PwBlk6x16 computes a 6-position × 16-output-channel tile of a blocked
// (nChw8c) 1x1 convolution. See pwblk.go for the reference semantics.
func PwBlk6x16(dst0, dst1, x, w []float32, cin, xbstride int) {
	pwblk6x16_asm(dst0, dst1, x, w, cin, xbstride)
}

//go:noescape
func pwblk6x16t_asm(dst0, dst1, x, w []float32, cin, xbstride, tiles int)

// PwBlk6x16Tiles runs `tiles` consecutive 6-position tiles (dst0/dst1/x
// advance 48 floats per tile); the loop lives in asm to shed the per-tile
// call overhead.
func PwBlk6x16Tiles(dst0, dst1, x, w []float32, cin, xbstride, tiles int) {
	pwblk6x16t_asm(dst0, dst1, x, w, cin, xbstride, tiles)
}
