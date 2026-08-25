//go:build arm64

package vek

import "math"

//go:noescape
func add_asm(dst, a, b []float32, n int)

//go:noescape
func zip2_asm(dst, a, b []float32, n int, c float32)

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
func exp_asm(dst, src []float32, n int)

//go:noescape
func sigmoid_asm(dst, src []float32, n int)

//go:noescape
func silu_asm(dst, src []float32, n int)

//go:noescape
func erf_asm(dst, src []float32, n int)

//go:noescape
func gelu_asm(dst, src []float32, n int)

//go:noescape
func dot_asm(a, b []float32, n int, out []float32)

//go:noescape
func dotbf16_asm(a []float32, b []uint16, n int, out []float32)

//go:noescape
func dwconv3x3s1_asm(dst, src, wpacked []float32, ncols, W int)

//go:noescape
func dwconv5x5s1_asm(dst, src, wpacked []float32, ncols, W int)

//go:noescape
func dwconv3x2s1_asm(dst, src, wpacked []float32, ncols, W int)

//go:noescape
func dwconv3x1s1_asm(dst, src, wpacked []float32, ncols, W int)

//go:noescape
func dwconv5x3s1_asm(dst, src, wpacked []float32, ncols, W int)

//go:noescape
func dwconv5x2s1_asm(dst, src, wpacked []float32, ncols, W int)

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

// Zip2 interleaves two rows with a scalar add: dst[2i] = a[i]+c,
// dst[2i+1] = b[i]+c (the 2×-upsample / stride-2 col2im primitive).
func Zip2(dst, a, b []float32, c float32) {
	n := min(len(a), len(b), len(dst)/2)
	m := n &^ 7
	if m > 0 {
		zip2_asm(dst, a, b, m, c)
	}
	for i := m; i < n; i++ {
		dst[2*i] = a[i] + c
		dst[2*i+1] = b[i] + c
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

// Axpy computes dst += a*src elementwise (in place on dst).
func Axpy(dst, src []float32, a float32) {
	n := min(len(dst), len(src))
	m := n &^ 3
	axpy_asm(dst, src, m, a)
	for i := m; i < n; i++ {
		dst[i] += a * src[i]
	}
}

// DwRow3x3S1 adds a stride-1 3x3 depthwise conv into dst for ncols output
// columns, all taps in-bounds. wpacked holds 9 weights one-per-lane, padded to
// 12. src points at the top-left input element for output column 0. dst is
// pre-filled with bias. Only ncols&^3 columns are done here; the caller handles
// the <4 remainder. W is the input row stride in elements.
func DwRow3x3S1(dst, src, wpacked []float32, ncols, W int) {
	m := ncols &^ 3
	dwconv3x3s1_asm(dst, src, wpacked, m, W)
	dwTail(dst, src, wpacked, m, ncols, W, 3, 3)
}

// DwRow5x5S1 is DwRow3x3S1 for a 5x5 kernel (25 weights padded to 28).
func DwRow5x5S1(dst, src, wpacked []float32, ncols, W int) {
	m := ncols &^ 3
	dwconv5x5s1_asm(dst, src, wpacked, m, W)
	dwTail(dst, src, wpacked, m, ncols, W, 5, 5)
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

// Exp computes dst = e^src. Inputs saturate at [-87.3, 88.4] (smallest normal
// / ~2.3e38) instead of flushing to 0 / +Inf; relative error ~1e-7.
func Exp(dst, src []float32) {
	n := min(len(dst), len(src))
	m := n &^ 3
	exp_asm(dst, src, m)
	for i := m; i < n; i++ {
		dst[i] = expScalar(src[i])
	}
}

// Sigmoid computes dst = 1/(1+e^-src).
func Sigmoid(dst, src []float32) {
	n := min(len(dst), len(src))
	m := n &^ 3
	sigmoid_asm(dst, src, m)
	for i := m; i < n; i++ {
		dst[i] = 1 / (1 + expScalar(-src[i]))
	}
}

// DwRowS1 adds a stride-1 KHxKW depthwise conv into dst for ncols output
// columns (all taps in-bounds; dst pre-filled). wpacked holds KH*KW weights
// row-major, padded to a multiple of the lane count. Supported shapes have a
// SIMD kernel (3x3, 5x5 and the stride-2 sub-kernels 3x2, 3x1, 5x3, 5x2);
// anything else runs scalar.
func DwRowS1(dst, src, wpacked []float32, ncols, W, KH, KW int) {
	m := ncols &^ 3
	switch {
	case KH == 3 && KW == 3:
		dwconv3x3s1_asm(dst, src, wpacked, m, W)
	case KH == 5 && KW == 5:
		dwconv5x5s1_asm(dst, src, wpacked, m, W)
	case KH == 3 && KW == 2:
		dwconv3x2s1_asm(dst, src, wpacked, m, W)
	case KH == 3 && KW == 1:
		dwconv3x1s1_asm(dst, src, wpacked, m, W)
	case KH == 5 && KW == 3:
		dwconv5x3s1_asm(dst, src, wpacked, m, W)
	case KH == 5 && KW == 2:
		dwconv5x2s1_asm(dst, src, wpacked, m, W)
	default:
		m = 0
	}
	dwTail(dst, src, wpacked, m, ncols, W, KH, KW)
}

// Dot returns Σ a[i]*b[i] over min lengths, accumulated in several
// independent SIMD lanes (the summation order differs from a sequential loop).
// DotBF16 computes Σ a[i]·widen(b[i]) for bf16 weights b (bits, f32 = bits<<16).
func DotBF16(a []float32, b []uint16) float32 {
	n := min(len(a), len(b))
	m := n &^ 15
	var s float32
	if m > 0 {
		var parts [16]float32
		dotbf16_asm(a, b, m, parts[:])
		for _, v := range parts {
			s += v
		}
	}
	for i := m; i < n; i++ {
		s += a[i] * math.Float32frombits(uint32(b[i])<<16)
	}
	return s
}

func Dot(a, b []float32) float32 {
	n := min(len(a), len(b))
	m := n &^ 15
	var s float32
	if m > 0 {
		var parts [16]float32
		dot_asm(a, b, m, parts[:])
		for _, v := range parts {
			s += v
		}
	}
	for i := m; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}

// SiLU computes dst = src · sigmoid(src) (a.k.a. Swish).
func SiLU(dst, src []float32) {
	n := min(len(dst), len(src))
	m := n &^ 3
	silu_asm(dst, src, m)
	for i := m; i < n; i++ {
		x := src[i]
		dst[i] = x / (1 + expScalar(-x))
	}
}

// Erf computes dst = erf(src) (abs error ≤ ~2e-7).
func Erf(dst, src []float32) {
	n := min(len(dst), len(src))
	m := n &^ 3
	erf_asm(dst, src, m)
	for i := m; i < n; i++ {
		dst[i] = float32(math.Erf(float64(src[i])))
	}
}

// Gelu computes dst = 0.5·src·(1+erf(src/√2)) (the exact, erf-based GELU).
func Gelu(dst, src []float32) {
	n := min(len(dst), len(src))
	m := n &^ 3
	gelu_asm(dst, src, m)
	for i := m; i < n; i++ {
		x := src[i]
		dst[i] = 0.5 * x * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
	}
}
