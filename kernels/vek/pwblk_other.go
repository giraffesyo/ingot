//go:build !amd64 && !arm64

package vek

// PwBlk6x16: portable blocked 1x1 conv tile (see amd64 asm).
func PwBlk6x16(dst0, dst1, x, w []float32, cin, xbstride int) {
	pwBlkRef(dst0, dst1, x, w, cin, xbstride)
}

// PwBlk6x16Tiles: portable multi-tile loop.
func PwBlk6x16Tiles(dst0, dst1, x, w []float32, cin, xbstride, tiles int) {
	for t := 0; t < tiles; t++ {
		pwBlkRef(dst0[t*48:], dst1[t*48:], x[t*48:], w, cin, xbstride)
	}
}
