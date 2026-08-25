package vek

import "math"

// Scalar reference/tail implementations for the quantization kernels; the
// SIMD versions mirror these bit for bit.

const qmagic = 1.5 * (1 << 23)

func qRoundEven(v float32) float32 {
	if v > 1<<22 || v < -(1<<22) {
		return float32(math.RoundToEven(float64(v)))
	}
	return v + qmagic - qmagic
}

func qSatU8(v float32) uint8 {
	r := qRoundEven(v)
	if r <= 0 {
		return 0
	}
	if r >= 255 {
		return 255
	}
	return uint8(r)
}

func qSatI8(v float32) int8 {
	r := qRoundEven(v)
	if r <= -128 {
		return -128
	}
	if r >= 127 {
		return 127
	}
	return int8(r)
}
