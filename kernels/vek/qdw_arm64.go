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
	switch {
	case KH == 3 && KW == 3:
		qdw3x3s1_asm(acc, src, wp, m, W)
	case KH == 5 && KW == 5:
		qdw5x5s1_asm(acc, src, wp, m, W)
	case KH == 3 && KW == 2:
		qdw3x2s1_asm(acc, src, wp, m, W)
	case KH == 3 && KW == 1:
		qdw3x1s1_asm(acc, src, wp, m, W)
	case KH == 5 && KW == 3:
		qdw5x3s1_asm(acc, src, wp, m, W)
	case KH == 5 && KW == 2:
		qdw5x2s1_asm(acc, src, wp, m, W)
	default:
		m = 0
	}
	qdwTail(acc, src, wp, m, ncols, W, KH, KW)
}
