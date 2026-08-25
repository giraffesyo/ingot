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

func RequantU8(dst []uint8, src []int32, mult, off float32, corr int32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = qSatU8(float32(src[i]+corr)*mult + off)
	}
}

func RequantI8(dst []int8, src []int32, mult, off float32, corr int32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = qSatI8(float32(src[i]+corr)*mult + off)
	}
}

func WidenS8S16(dst []int16, src []int8) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = int16(src[i])
	}
}

func DeinterleaveS16(ev, od, src []int16) {
	n := min(len(ev), len(od), len(src)/2)
	for i := 0; i < n; i++ {
		ev[i] = src[2*i]
		od[i] = src[2*i+1]
	}
}

func ShiftU8S8(dst []int8, src []uint8) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = int8(src[i] ^ 0x80)
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
