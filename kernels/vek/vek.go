//go:build arm64

package vek

//go:noescape
func add_asm(dst, a, b []float32, n int)

//go:noescape
func sub_asm(dst, a, b []float32, n int)

//go:noescape
func mul_asm(dst, a, b []float32, n int)

//go:noescape
func div_asm(dst, a, b []float32, n int)

//go:noescape
func maxpair_asm(dst, a, b []float32, n int)

//go:noescape
func minpair_asm(dst, a, b []float32, n int)

//go:noescape
func relu_asm(dst, src []float32, n int)

//go:noescape
func hardswish_asm(dst, src []float32, n int)

//go:noescape
func hardsigmoid_asm(dst, src []float32, n int, alpha, beta float32)

//go:noescape
func clip_asm(dst, src []float32, n int, lo, hi float32)

//go:noescape
func leakyrelu_asm(dst, src []float32, n int, alpha float32)

//go:noescape
func addscalar_asm(dst, src []float32, n int, s float32)

//go:noescape
func mulscalar_asm(dst, src []float32, n int, s float32)

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

// Add computes dst = a + b elementwise over min lengths.
func Add(dst, a, b []float32) {
	n := vecLen(dst, a, b)
	m := n &^ 3
	add_asm(dst, a, b, m)
	for i := m; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}

// Sub computes dst = a - b.
func Sub(dst, a, b []float32) {
	n := vecLen(dst, a, b)
	m := n &^ 3
	sub_asm(dst, a, b, m)
	for i := m; i < n; i++ {
		dst[i] = a[i] - b[i]
	}
}

// Mul computes dst = a * b.
func Mul(dst, a, b []float32) {
	n := vecLen(dst, a, b)
	m := n &^ 3
	mul_asm(dst, a, b, m)
	for i := m; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

// Div computes dst = a / b.
func Div(dst, a, b []float32) {
	n := vecLen(dst, a, b)
	m := n &^ 3
	div_asm(dst, a, b, m)
	for i := m; i < n; i++ {
		dst[i] = a[i] / b[i]
	}
}

// MaxPair computes dst = max(a, b).
func MaxPair(dst, a, b []float32) {
	n := vecLen(dst, a, b)
	m := n &^ 3
	maxpair_asm(dst, a, b, m)
	for i := m; i < n; i++ {
		if b[i] > a[i] {
			dst[i] = b[i]
		} else {
			dst[i] = a[i]
		}
	}
}

// MinPair computes dst = min(a, b).
func MinPair(dst, a, b []float32) {
	n := vecLen(dst, a, b)
	m := n &^ 3
	minpair_asm(dst, a, b, m)
	for i := m; i < n; i++ {
		if b[i] < a[i] {
			dst[i] = b[i]
		} else {
			dst[i] = a[i]
		}
	}
}

// Relu computes dst = max(src, 0).
func Relu(dst, src []float32) {
	n := min(len(dst), len(src))
	m := n &^ 3
	relu_asm(dst, src, m)
	for i := m; i < n; i++ {
		if src[i] > 0 {
			dst[i] = src[i]
		} else {
			dst[i] = 0
		}
	}
}

// HardSwish computes dst = src * clamp(src/6+0.5, 0, 1).
func HardSwish(dst, src []float32) {
	n := min(len(dst), len(src))
	m := n &^ 3
	hardswish_asm(dst, src, m)
	for i := m; i < n; i++ {
		dst[i] = src[i] * clamp01(src[i]/6+0.5)
	}
}

// HardSigmoid computes dst = clamp(alpha*src+beta, 0, 1).
func HardSigmoid(dst, src []float32, alpha, beta float32) {
	n := min(len(dst), len(src))
	m := n &^ 3
	hardsigmoid_asm(dst, src, m, alpha, beta)
	for i := m; i < n; i++ {
		dst[i] = clamp01(alpha*src[i] + beta)
	}
}

// Clip computes dst = min(max(src, lo), hi).
func Clip(dst, src []float32, lo, hi float32) {
	n := min(len(dst), len(src))
	m := n &^ 3
	clip_asm(dst, src, m, lo, hi)
	for i := m; i < n; i++ {
		v := src[i]
		if v < lo {
			v = lo
		}
		if v > hi {
			v = hi
		}
		dst[i] = v
	}
}

// LeakyRelu computes dst = src>0 ? src : alpha*src.
func LeakyRelu(dst, src []float32, alpha float32) {
	n := min(len(dst), len(src))
	m := n &^ 3
	leakyrelu_asm(dst, src, m, alpha)
	for i := m; i < n; i++ {
		v := src[i]
		if v < 0 {
			v *= alpha
		}
		dst[i] = v
	}
}

// AddScalar computes dst = src + s.
func AddScalar(dst, src []float32, s float32) {
	n := min(len(dst), len(src))
	m := n &^ 3
	addscalar_asm(dst, src, m, s)
	for i := m; i < n; i++ {
		dst[i] = src[i] + s
	}
}

// MulScalar computes dst = src * s.
func MulScalar(dst, src []float32, s float32) {
	n := min(len(dst), len(src))
	m := n &^ 3
	mulscalar_asm(dst, src, m, s)
	for i := m; i < n; i++ {
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
