//go:build arm64

package vek

//go:noescape
func qdw3x3s1_asm(acc []int32, src, wp []int16, ncols, W int)

//go:noescape
func qdw5x5s1_asm(acc []int32, src, wp []int16, ncols, W int)

//go:noescape
func qdw3x2s1_asm(acc []int32, src, wp []int16, ncols, W int)

//go:noescape
func qdw3x1s1_asm(acc []int32, src, wp []int16, ncols, W int)

//go:noescape
func qdw5x3s1_asm(acc []int32, src, wp []int16, ncols, W int)

//go:noescape
func qdw5x2s1_asm(acc []int32, src, wp []int16, ncols, W int)

// QDwRowS1 accumulates a stride-1 KHxKW depthwise conv row into acc (s32,
// pre-filled by the caller) from the widened s16 source plane. wp holds KH*KW
// s16 weights row-major, padded to a multiple of 8. Shapes with a kernel run
// SIMD; anything else scalar.
func QDwRowS1(acc []int32, src, wp []int16, ncols, W, KH, KW int) {
	m := ncols &^ 7
	var asm func(acc []int32, src, wp []int16, ncols, W int)
	switch {
	case KH == 3 && KW == 3:
		asm = qdw3x3s1_asm
	case KH == 5 && KW == 5:
		asm = qdw5x5s1_asm
	case KH == 3 && KW == 2:
		asm = qdw3x2s1_asm
	case KH == 3 && KW == 1:
		asm = qdw3x1s1_asm
	case KH == 5 && KW == 3:
		asm = qdw5x3s1_asm
	case KH == 5 && KW == 2:
		asm = qdw5x2s1_asm
	default:
		qdwTail(acc, src, wp, 0, ncols, W, KH, KW)
		return
	}
	asm(acc, src, wp, m, W)
	if t := ncols - m; t > 0 {
		if ncols >= 8 {
			// Re-run the kernel on the last (overlapping) 8-column window
			// into zeroed scratch and add only the fresh tail lanes — the
			// scalar tail was 25-tap × 4-col work on every deep row
			// (OW=20 layers), ~7% of det_int8 1T.
			var tmp [8]int32
			asm(tmp[:], src[ncols-8:], wp, 8, W)
			for i := 0; i < t; i++ {
				acc[m+i] += tmp[8-t+i]
			}
			return
		}
		qdwTail(acc, src, wp, m, ncols, W, KH, KW)
	}
}
