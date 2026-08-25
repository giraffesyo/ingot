//go:build !arm64

package vek

// Portable quantization kernels (amd64 SIMD is future work).

func QuantU8(dst []uint8, src []float32, scale, zp float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = qSatU8(src[i]/scale + zp)
	}
}

func QuantI8(dst []int8, src []float32, scale, zp float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = qSatI8(src[i]/scale + zp)
	}
}

func RequantU8(dst []uint8, src []int32, mult, off float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = qSatU8(float32(src[i])*mult + off)
	}
}

func RequantI8(dst []int8, src []int32, mult, off float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = qSatI8(float32(src[i])*mult + off)
	}
}

func DequantU8(dst []float32, src []uint8, scale float32, zp int32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = float32(int32(src[i])-zp) * scale
	}
}

func DequantI8(dst []float32, src []int8, scale float32, zp int32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = float32(int32(src[i])-zp) * scale
	}
}
