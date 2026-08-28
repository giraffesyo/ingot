package vek

// dwBlk8Ref: scalar reference for the channel-blocked depthwise row.
func dwBlk8Ref(dst, src, w []float32, ncols, wp int) {
	for j := 0; j < ncols; j++ {
		for c := 0; c < 8; c++ {
			var acc float32
			for kh := 0; kh < 3; kh++ {
				for kw := 0; kw < 3; kw++ {
					acc += w[(kh*3+kw)*8+c] * src[(kh*wp+j+kw)*8+c]
				}
			}
			dst[j*8+c] = acc
		}
	}
}
