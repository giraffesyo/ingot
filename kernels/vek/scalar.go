package vek

import "math"

// Exp saturation bounds shared by the SIMD kernels and the scalar tail/fallback.
const (
	expLo = -87.33654 // exp(expLo) ≈ smallest normal f32
	expHi = 88.37626  // exp(expHi) ≈ 2.3e38
)

func expScalar(x float32) float32 {
	return float32(math.Exp(float64(min(max(x, expLo), expHi))))
}

// I8Group is the group length of the Q8 kernels: one f32 scale per 64
// int8 values.
const I8Group = 64

// dotQ8Ref is DotQ8's portable form: exact int32 group sums, scaled.
func dotQ8Ref(w, xh, xl []int8, sx, sw []float32) float32 {
	var acc float32
	for g := range len(w) / I8Group {
		var p int32
		for i := g * I8Group; i < (g+1)*I8Group; i++ {
			p += int32(w[i]) * (128*int32(xh[i]) + int32(xl[i]))
		}
		acc += sx[g] * sw[g] * float32(p)
	}
	return acc
}

// x16Max is the largest |x/s| QuantizeX16 produces: 127·128, so the high
// part fits int8 after rounding to the nearest multiple of 128.
const x16Max = 127 * 128

// QuantizeX16 splits x (length a multiple of I8Group) into ~14-bit
// fixed point per group: s = max|x|/x16Max, q = round(x/s) = 128·xh + xl
// with xh, xl int8 (xl in [-64, 63]).
func QuantizeX16(xh, xl []int8, sx []float32, x []float32) {
	for g := range len(x) / I8Group {
		var amax float32
		for _, v := range x[g*I8Group : (g+1)*I8Group] {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		s := amax / x16Max
		sx[g] = s
		if s == 0 {
			clear(xh[g*I8Group : (g+1)*I8Group])
			clear(xl[g*I8Group : (g+1)*I8Group])
			continue
		}
		inv := 1 / s
		for i := g * I8Group; i < (g+1)*I8Group; i++ {
			q := int32(math.RoundToEven(float64(x[i] * inv)))
			q = max(-x16Max, min(x16Max, q))
			h := (q + 64) >> 7 // nearest multiple of 128 (arithmetic shift floors)
			xh[i], xl[i] = int8(h), int8(q-h<<7)
		}
	}
}
