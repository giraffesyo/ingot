package vek

// pwBlkRef: scalar reference for the blocked pointwise tile.
// dst0[p][8] / dst1[p][8] for 6 positions; x[block][pos][8] with xbstride
// float32 elements... xbstride is in BYTES to match the asm; w[ci][16].
func pwBlkRef(dst0, dst1, x, w []float32, cin, xbstride int) {
	xs := xbstride / 4
	for p := 0; p < 6; p++ {
		for o := 0; o < 8; o++ {
			var a0, a1 float32
			for ci := 0; ci < cin; ci++ {
				xv := x[(ci/8)*xs+p*8+ci%8]
				a0 += xv * w[ci*16+o]
				a1 += xv * w[ci*16+8+o]
			}
			dst0[p*8+o] = a0
			dst1[p*8+o] = a1
		}
	}
}
