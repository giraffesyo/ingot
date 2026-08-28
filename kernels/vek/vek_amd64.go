//go:build amd64

package vek

import "math"

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
	dwTail(dst, src, wpacked, m, ncols, W, 3, 3)
}
func DwRow5x5S1(dst, src, wpacked []float32, ncols, W int) {
	m := ncols &^ 7
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
	m := n &^ 7
	exp_asm(dst, src, m)
	for i := m; i < n; i++ {
		dst[i] = expScalar(src[i])
	}
}

// Sigmoid computes dst = 1/(1+e^-src).
func Sigmoid(dst, src []float32) {
	n := min(len(dst), len(src))
	m := n &^ 7
	sigmoid_asm(dst, src, m)
	for i := m; i < n; i++ {
		dst[i] = flushSub(1 / (1 + expScalar(-src[i])))
	}
}

// DwRowS1 adds a stride-1 KHxKW depthwise conv into dst for ncols output
// columns (all taps in-bounds; dst pre-filled). wpacked holds KH*KW weights
// row-major, padded to a multiple of the lane count. Supported shapes have a
// SIMD kernel (3x3, 5x5 and the stride-2 sub-kernels 3x2, 3x1, 5x3, 5x2);
// anything else runs scalar.
func DwRowS1(dst, src, wpacked []float32, ncols, W, KH, KW int) {
	m := ncols &^ 7
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
func Dot(a, b []float32) float32 {
	n := min(len(a), len(b))
	m := n &^ 31
	var s float32
	if m > 0 {
		var parts [32]float32
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
	m := n &^ 7
	silu_asm(dst, src, m)
	for i := m; i < n; i++ {
		x := src[i]
		dst[i] = x * flushSub(1/(1+expScalar(-x)))
	}
}

// Erf computes dst = erf(src) (abs error ≤ ~2e-7).
func Erf(dst, src []float32) {
	n := min(len(dst), len(src))
	m := n &^ 7
	erf_asm(dst, src, m)
	for i := m; i < n; i++ {
		dst[i] = float32(math.Erf(float64(src[i])))
	}
}

// Gelu computes dst = 0.5·src·(1+erf(src/√2)) (the exact, erf-based GELU).
func Gelu(dst, src []float32) {
	n := min(len(dst), len(src))
	m := n &^ 7
	gelu_asm(dst, src, m)
	for i := m; i < n; i++ {
		x := src[i]
		dst[i] = 0.5 * x * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
	}
}

// Zip2 interleaves two rows with a scalar add: dst[2i] = a[i]+c,
// dst[2i+1] = b[i]+c.
func Zip2(dst, a, b []float32, c float32) {
	n := min(len(a), len(b), len(dst)/2)
	for i := 0; i < n; i++ {
		dst[2*i] = a[i] + c
		dst[2*i+1] = b[i] + c
	}
}

//go:noescape
func dotbf16_asm(a []float32, b []uint16, n int) float32

//go:noescape
func axpybf16_asm(dst []float32, src []uint16, n int, a float32)

// DotBF16 computes Σ a[i]·widen(b[i]) for bf16 weights b (bits<<16).
func DotBF16(a []float32, b []uint16) float32 {
	n := min(len(a), len(b))
	m := n &^ 15
	var s float32
	if m > 0 {
		s = dotbf16_asm(a, b, m)
	}
	for i := m; i < n; i++ {
		s += a[i] * math.Float32frombits(uint32(b[i])<<16)
	}
	return s
}

// AxpyBF16 computes dst += a·widen(src) for bf16 src.
func AxpyBF16(dst []float32, src []uint16, a float32) {
	n := min(len(dst), len(src))
	m := n &^ 15
	if m > 0 {
		axpybf16_asm(dst, src, m, a)
	}
	for i := m; i < n; i++ {
		dst[i] += a * math.Float32frombits(uint32(src[i])<<16)
	}
}

// flushSub flushes a subnormal sigmoid output to zero, matching the SIMD
// kernels (a subnormal operand in a downstream multiply is ~100 cycles on
// x86 — the softmax ambush's cousin).
func flushSub(v float32) float32 {
	if v < 1.17549435e-38 {
		return 0
	}
	return v
}

//go:noescape
func mulblk8_asm(dst, src []float32, n int, s []float32)

//go:noescape
func sumblk8_asm(dst, src []float32, n int)

// MulBlk8 computes dst = src * s, with the 8-lane channel pattern s repeated
// across an nChw8c plane (per-channel scale). Tail elements use s[i%8].
func MulBlk8(dst, src, s []float32) {
	n := min(len(dst), len(src))
	m := n &^ 7
	if m > 0 {
		mulblk8_asm(dst, src, m, s)
	}
	for i := m; i < n; i++ {
		dst[i] = src[i] * s[i&7]
	}
}

// SumBlk8 overwrites dst[0:8] with the per-lane sums of the 8-lane pattern
// across an nChw8c plane src.
func SumBlk8(dst, src []float32) {
	m := len(src) &^ 7
	sumblk8_asm(dst[:8], src, m)
	for i := m; i < len(src); i++ {
		dst[i&7] += src[i]
	}
}
