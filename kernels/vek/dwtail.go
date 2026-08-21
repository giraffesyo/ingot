package vek

// dwTail computes output columns [from, ncols) of a stride-1 KxK depthwise conv
// scalar-wise, adding into dst. It lets the DwRow* wrappers own their full width
// regardless of the SIMD lane count, so callers need not know it.
func dwTail(dst, src, wpacked []float32, from, ncols, W, K int) {
	for c := from; c < ncols; c++ {
		var acc float32
		for kh := 0; kh < K; kh++ {
			base := kh*W + c
			for kw := 0; kw < K; kw++ {
				acc += wpacked[kh*K+kw] * src[base+kw]
			}
		}
		dst[c] += acc
	}
}
