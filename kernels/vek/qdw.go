package vek

// qdwTail: scalar tail/reference for the int8-depthwise row kernels.
func qdwTail(acc []int32, src, wp []int16, from, ncols, W, KH, KW int) {
	for c := from; c < ncols; c++ {
		var s int32
		for kh := 0; kh < KH; kh++ {
			base := kh*W + c
			for kw := 0; kw < KW; kw++ {
				s += int32(wp[kh*KW+kw]) * int32(src[base+kw])
			}
		}
		acc[c] += s
	}
}
