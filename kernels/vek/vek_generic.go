//go:build !arm64

package vek

func vecLen(dst, a, b []float32) int {
	n := len(dst)
	if len(a) < n {
		n = len(a)
	}
	if len(b) < n {
		n = len(b)
	}
	return n
}

func Add(dst, a, b []float32) {
	for i := 0; i < vecLen(dst, a, b); i++ {
		dst[i] = a[i] + b[i]
	}
}
func Sub(dst, a, b []float32) {
	for i := 0; i < vecLen(dst, a, b); i++ {
		dst[i] = a[i] - b[i]
	}
}
func Mul(dst, a, b []float32) {
	for i := 0; i < vecLen(dst, a, b); i++ {
		dst[i] = a[i] * b[i]
	}
}
func Div(dst, a, b []float32) {
	for i := 0; i < vecLen(dst, a, b); i++ {
		dst[i] = a[i] / b[i]
	}
}
func MaxPair(dst, a, b []float32) {
	for i := 0; i < vecLen(dst, a, b); i++ {
		dst[i] = max(a[i], b[i])
	}
}
func MinPair(dst, a, b []float32) {
	for i := 0; i < vecLen(dst, a, b); i++ {
		dst[i] = min(a[i], b[i])
	}
}
func Relu(dst, src []float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = max(src[i], 0)
	}
}
func HardSwish(dst, src []float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = src[i] * clamp01(src[i]/6+0.5)
	}
}
func HardSigmoid(dst, src []float32, alpha, beta float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = clamp01(alpha*src[i] + beta)
	}
}
func Clip(dst, src []float32, lo, hi float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = min(max(src[i], lo), hi)
	}
}
func LeakyRelu(dst, src []float32, alpha float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		v := src[i]
		if v < 0 {
			v *= alpha
		}
		dst[i] = v
	}
}
func AddScalar(dst, src []float32, s float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = src[i] + s
	}
}
func MulScalar(dst, src []float32, s float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = src[i] * s
	}
}
func clamp01(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}
