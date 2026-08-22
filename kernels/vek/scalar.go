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
