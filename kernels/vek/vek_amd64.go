//go:build amd64

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

//go:noescape
func axpy_asm(dst, src []float32, n int, a float32)

//go:noescape
func dwconv3x3s1_asm(dst, src, wpacked []float32, ncols, W int)

//go:noescape
func dwconv5x5s1_asm(dst, src, wpacked []float32, ncols, W int)

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

// AVX2 kernels process 8 lanes at a time; the Go tail handles n%8.

func Add(dst, a, b []float32) {
	n := vecLen(dst, a, b)
	m := n &^ 7
	add_asm(dst, a, b, m)
	for i := m; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}
func Sub(dst, a, b []float32) {
	n := vecLen(dst, a, b)
	m := n &^ 7
	sub_asm(dst, a, b, m)
	for i := m; i < n; i++ {
		dst[i] = a[i] - b[i]
	}
}
func Mul(dst, a, b []float32) {
	n := vecLen(dst, a, b)
	m := n &^ 7
	mul_asm(dst, a, b, m)
	for i := m; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}
func Div(dst, a, b []float32) {
	n := vecLen(dst, a, b)
	m := n &^ 7
	div_asm(dst, a, b, m)
	for i := m; i < n; i++ {
		dst[i] = a[i] / b[i]
	}
}
func MaxPair(dst, a, b []float32) {
	n := vecLen(dst, a, b)
	m := n &^ 7
	maxpair_asm(dst, a, b, m)
	for i := m; i < n; i++ {
		if b[i] > a[i] {
			dst[i] = b[i]
		} else {
			dst[i] = a[i]
		}
	}
}
func MinPair(dst, a, b []float32) {
	n := vecLen(dst, a, b)
	m := n &^ 7
	minpair_asm(dst, a, b, m)
	for i := m; i < n; i++ {
		if b[i] < a[i] {
			dst[i] = b[i]
		} else {
			dst[i] = a[i]
		}
	}
}
func Relu(dst, src []float32) {
	n := min(len(dst), len(src))
	m := n &^ 7
	relu_asm(dst, src, m)
	for i := m; i < n; i++ {
		if src[i] > 0 {
			dst[i] = src[i]
		} else {
			dst[i] = 0
		}
	}
}
func HardSwish(dst, src []float32) {
	n := min(len(dst), len(src))
	m := n &^ 7
	hardswish_asm(dst, src, m)
	for i := m; i < n; i++ {
		dst[i] = src[i] * clamp01(src[i]/6+0.5)
	}
}
func HardSigmoid(dst, src []float32, alpha, beta float32) {
	n := min(len(dst), len(src))
	m := n &^ 7
	hardsigmoid_asm(dst, src, m, alpha, beta)
	for i := m; i < n; i++ {
		dst[i] = clamp01(alpha*src[i] + beta)
	}
}
func Clip(dst, src []float32, lo, hi float32) {
	n := min(len(dst), len(src))
	m := n &^ 7
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
func LeakyRelu(dst, src []float32, alpha float32) {
	n := min(len(dst), len(src))
	m := n &^ 7
	leakyrelu_asm(dst, src, m, alpha)
	for i := m; i < n; i++ {
		v := src[i]
		if v < 0 {
			v *= alpha
		}
		dst[i] = v
	}
}
func AddScalar(dst, src []float32, s float32) {
	n := min(len(dst), len(src))
	m := n &^ 7
	addscalar_asm(dst, src, m, s)
	for i := m; i < n; i++ {
		dst[i] = src[i] + s
	}
}
func MulScalar(dst, src []float32, s float32) {
	n := min(len(dst), len(src))
	m := n &^ 7
	mulscalar_asm(dst, src, m, s)
	for i := m; i < n; i++ {
		dst[i] = src[i] * s
	}
}
func Axpy(dst, src []float32, a float32) {
	n := min(len(dst), len(src))
	m := n &^ 7
	axpy_asm(dst, src, m, a)
	for i := m; i < n; i++ {
		dst[i] += a * src[i]
	}
}

// DwRow3x3S1 adds a stride-1 3x3 depthwise conv into dst over ncols output
// columns (all taps in-bounds); the AVX2 kernel does ncols&^7, the caller's
// remainder loop handles the rest.
func DwRow3x3S1(dst, src, wpacked []float32, ncols, W int) {
	m := ncols &^ 7
	dwconv3x3s1_asm(dst, src, wpacked, m, W)
	dwTail(dst, src, wpacked, m, ncols, W, 3)
}
func DwRow5x5S1(dst, src, wpacked []float32, ncols, W int) {
	m := ncols &^ 7
	dwconv5x5s1_asm(dst, src, wpacked, m, W)
	dwTail(dst, src, wpacked, m, ncols, W, 5)
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
