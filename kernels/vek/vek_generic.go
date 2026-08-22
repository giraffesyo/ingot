//go:build !arm64 && !amd64

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
func Axpy(dst, src []float32, a float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] += a * src[i]
	}
}

// DwRow3x3S1: scalar fallback (full width).
func DwRow3x3S1(dst, src, wpacked []float32, ncols, W int) {
	dwTail(dst, src, wpacked, 0, ncols, W, 3, 3)
}

// DwRow5x5S1: scalar fallback (full width).
func DwRow5x5S1(dst, src, wpacked []float32, ncols, W int) {
	dwTail(dst, src, wpacked, 0, ncols, W, 5, 5)
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

// Exp computes dst = e^src (saturating at [-87.3, 88.4] like the SIMD kernels).
func Exp(dst, src []float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = expScalar(src[i])
	}
}

// Sigmoid computes dst = 1/(1+e^-src).
func Sigmoid(dst, src []float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = 1 / (1 + expScalar(-src[i]))
	}
}

// DwRowS1: scalar fallback (full width).
func DwRowS1(dst, src, wpacked []float32, ncols, W, KH, KW int) {
	dwTail(dst, src, wpacked, 0, ncols, W, KH, KW)
}

// Dot returns Σ a[i]*b[i] over min lengths (4 independent accumulators).
func Dot(a, b []float32) float32 {
	n := min(len(a), len(b))
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}
