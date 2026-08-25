//go:build arm64

package vek

//go:noescape
func quantu8_asm(dst []uint8, src []float32, n int, scale, zp float32)

//go:noescape
func quanti8_asm(dst []int8, src []float32, n int, scale, zp float32)

//go:noescape
func requantu8_asm(dst []uint8, src []int32, n int, mult, off float32, corr int32)

//go:noescape
func requanti8_asm(dst []int8, src []int32, n int, mult, off float32, corr int32)

//go:noescape
func shiftu8s8_asm(dst []int8, src []uint8, n int)

//go:noescape
func widens8_asm(dst []int16, src []int8, n int)

//go:noescape
func deint16_asm(ev, od []int16, src []int16, n int)

//go:noescape
func dequantu8_asm(dst []float32, src []uint8, n int, scale float32, zp int32)

//go:noescape
func dequanti8_asm(dst []float32, src []int8, n int, scale float32, zp int32)

// QuantU8 computes dst = sat_u8(round_even(src/scale + zp)).
func QuantU8(dst []uint8, src []float32, scale, zp float32) {
	n := min(len(dst), len(src))
	m := n &^ 15
	quantu8_asm(dst, src, m, scale, zp)
	for i := m; i < n; i++ {
		dst[i] = qSatU8(src[i]/scale + zp)
	}
}

// QuantI8 computes dst = sat_i8(round_even(src/scale + zp)).
func QuantI8(dst []int8, src []float32, scale, zp float32) {
	n := min(len(dst), len(src))
	m := n &^ 15
	quanti8_asm(dst, src, m, scale, zp)
	for i := m; i < n; i++ {
		dst[i] = qSatI8(src[i]/scale + zp)
	}
}

// RequantU8 computes dst = sat_u8(round_even(f32(src+corr)·mult + off)).
func RequantU8(dst []uint8, src []int32, mult, off float32, corr int32) {
	n := min(len(dst), len(src))
	m := n &^ 15
	requantu8_asm(dst, src, m, mult, off, corr)
	for i := m; i < n; i++ {
		dst[i] = qSatU8(float32(src[i]+corr)*mult + off)
	}
}

// RequantI8 computes dst = sat_i8(round_even(f32(src+corr)·mult + off)).
func RequantI8(dst []int8, src []int32, mult, off float32, corr int32) {
	n := min(len(dst), len(src))
	m := n &^ 15
	requanti8_asm(dst, src, m, mult, off, corr)
	for i := m; i < n; i++ {
		dst[i] = qSatI8(float32(src[i]+corr)*mult + off)
	}
}

// WidenS8S16 computes dst = int16(src).
func WidenS8S16(dst []int16, src []int8) {
	n := min(len(dst), len(src))
	m := n &^ 15
	if m > 0 {
		widens8_asm(dst, src, m)
	}
	for i := m; i < n; i++ {
		dst[i] = int16(src[i])
	}
}

// DeinterleaveS16 splits src into even and odd elements:
// ev[i] = src[2i], od[i] = src[2i+1], for i < min(len(ev), len(od), len(src)/2).
func DeinterleaveS16(ev, od, src []int16) {
	n := min(len(ev), len(od), len(src)/2)
	m := n &^ 7
	if m > 0 {
		deint16_asm(ev, od, src, m)
	}
	for i := m; i < n; i++ {
		ev[i] = src[2*i]
		od[i] = src[2*i+1]
	}
}

// ShiftU8S8 computes dst = int8(src ^ 0x80) — the u8→s8 shift (x−128).
func ShiftU8S8(dst []int8, src []uint8) {
	n := min(len(dst), len(src))
	m := n &^ 63
	shiftu8s8_asm(dst, src, m)
	for i := m; i < n; i++ {
		dst[i] = int8(src[i] ^ 0x80)
	}
}

// DequantU8 computes dst = f32(src − zp)·scale.
func DequantU8(dst []float32, src []uint8, scale float32, zp int32) {
	n := min(len(dst), len(src))
	m := n &^ 15
	dequantu8_asm(dst, src, m, scale, zp)
	for i := m; i < n; i++ {
		dst[i] = float32(int32(src[i])-zp) * scale
	}
}

// DequantI8 computes dst = f32(src − zp)·scale.
func DequantI8(dst []float32, src []int8, scale float32, zp int32) {
	n := min(len(dst), len(src))
	m := n &^ 15
	dequanti8_asm(dst, src, m, scale, zp)
	for i := m; i < n; i++ {
		dst[i] = float32(int32(src[i])-zp) * scale
	}
}
