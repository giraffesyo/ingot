//go:build amd64

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
// SIMD; anything else scalar. Dispatch is a concrete call per shape so the
// overlapped-window tail scratch stays on the stack (an indirect func value
// defeats //go:noescape and heap-allocates the scratch on every row).
func QDwRowS1(acc []int32, src, wp []int16, ncols, W, KH, KW int) {
	switch {
	case KH == 3 && KW == 3:
		m := ncols &^ 7
		qdw3x3s1_asm(acc, src, wp, m, W)
		if t := ncols - m; t > 0 {
			if ncols >= 8 {
				var tmp [8]int32
				qdw3x3s1_asm(tmp[:], src[ncols-8:], wp, 8, W)
				for i := 0; i < t; i++ {
					acc[m+i] += tmp[8-t+i]
				}
				return
			}
			qdwTail(acc, src, wp, m, ncols, W, KH, KW)
		}
	case KH == 5 && KW == 5:
		m := ncols &^ 7
		qdw5x5s1_asm(acc, src, wp, m, W)
		if t := ncols - m; t > 0 {
			if ncols >= 8 {
				var tmp [8]int32
				qdw5x5s1_asm(tmp[:], src[ncols-8:], wp, 8, W)
				for i := 0; i < t; i++ {
					acc[m+i] += tmp[8-t+i]
				}
				return
			}
			qdwTail(acc, src, wp, m, ncols, W, KH, KW)
		}
	case KH == 3 && KW == 2:
		m := ncols &^ 7
		qdw3x2s1_asm(acc, src, wp, m, W)
		if t := ncols - m; t > 0 {
			if ncols >= 8 {
				var tmp [8]int32
				qdw3x2s1_asm(tmp[:], src[ncols-8:], wp, 8, W)
				for i := 0; i < t; i++ {
					acc[m+i] += tmp[8-t+i]
				}
				return
			}
			qdwTail(acc, src, wp, m, ncols, W, KH, KW)
		}
	case KH == 3 && KW == 1:
		m := ncols &^ 7
		qdw3x1s1_asm(acc, src, wp, m, W)
		if t := ncols - m; t > 0 {
			if ncols >= 8 {
				var tmp [8]int32
				qdw3x1s1_asm(tmp[:], src[ncols-8:], wp, 8, W)
				for i := 0; i < t; i++ {
					acc[m+i] += tmp[8-t+i]
				}
				return
			}
			qdwTail(acc, src, wp, m, ncols, W, KH, KW)
		}
	case KH == 5 && KW == 3:
		m := ncols &^ 7
		qdw5x3s1_asm(acc, src, wp, m, W)
		if t := ncols - m; t > 0 {
			if ncols >= 8 {
				var tmp [8]int32
				qdw5x3s1_asm(tmp[:], src[ncols-8:], wp, 8, W)
				for i := 0; i < t; i++ {
					acc[m+i] += tmp[8-t+i]
				}
				return
			}
			qdwTail(acc, src, wp, m, ncols, W, KH, KW)
		}
	case KH == 5 && KW == 2:
		m := ncols &^ 7
		qdw5x2s1_asm(acc, src, wp, m, W)
		if t := ncols - m; t > 0 {
			if ncols >= 8 {
				var tmp [8]int32
				qdw5x2s1_asm(tmp[:], src[ncols-8:], wp, 8, W)
				for i := 0; i < t; i++ {
					acc[m+i] += tmp[8-t+i]
				}
				return
			}
			qdwTail(acc, src, wp, m, ncols, W, KH, KW)
		}
	default:
		qdwTail(acc, src, wp, 0, ncols, W, KH, KW)
	}
}
