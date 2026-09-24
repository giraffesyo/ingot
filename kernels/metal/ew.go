//go:build darwin && arm64

package metal

import (
	"fmt"
	"sync"
)

// Generic elementwise, transpose and row-reduction kernels for executing
// ONNX graphs on the GPU (f32).
const ewSrc = `
#include <metal_stdlib>
using namespace metal;

// x^n for integer n (exponentiation by squaring): pow() is NaN for negative
// bases, ONNX Pow with an integral exponent is not.
static float powi(float x, int n) {
	float r = 1.0f, b = x;
	uint e = uint(abs(n));
	while (e) {
		if (e & 1u) r *= b;
		b *= b;
		e >>= 1;
	}
	return n < 0 ? 1.0f / r : r;
}

// out[i] = op(a[(i / da) % ma], b[(i / db) % mb]) over n elements;
// p = (n, da, ma, op), q = (db, mb, 0, 0): each operand is a contiguous
// block of the output's dims repeated around it (trailing vector, per-row
// scalar, [1,H,1,1], ...).
kernel void binary_bcast(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                         device float* o [[buffer(2)]], constant uint4& p [[buffer(3)]],
                         constant uint4& q [[buffer(4)]], uint i [[thread_position_in_grid]]) {
	if (i >= p.x) return;
	const float x = a[(i / p.y) % p.z], y = b[(i / q.x) % q.y];
	const uint op = p.w;
	float r;
	switch (op) {
	case 0: r = x + y; break;
	case 1: r = x - y; break;
	case 2: r = x * y; break;
	case 3: r = x / y; break;
	case 4: r = (y == floor(y) && fabs(y) < 64.0f) ? powi(x, int(y)) : pow(x, y); break;
	case 5: r = max(x, y); break;
	default: r = min(x, y); break;
	}
	o[i] = r;
}

static float erf_ew(float x) {
	const float z = fabs(x), t = 1.0f / (1.0f + 0.5f * z);
	const float r = t * exp(-z * z - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f +
		t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f +
		t * (-0.82215223f + t * 0.17087277f)))))))));
	return x >= 0 ? 1.0f - r : r - 1.0f;
}

// out[i] = op(x[i]); p = (n, op).
kernel void unary_ew(device const float* x [[buffer(0)]], device float* o [[buffer(1)]],
                     constant uint4& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
	if (i >= p.x) return;
	const float v = x[i];
	float r;
	switch (p.y) {
	case 0: r = max(v, 0.0f); break;                                   // relu
	case 1: r = 1.0f / (1.0f + exp(-v)); break;                        // sigmoid
	case 2: r = precise::tanh(v); break;                               // tanh
	case 3: r = 0.5f * v * (1.0f + erf_ew(v * 0.70710678f)); break;    // gelu (erf)
	case 4: r = 0.5f * v * (1.0f + precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v))); break;
	case 5: r = v / (1.0f + exp(-v)); break;                           // silu
	case 6: r = sqrt(v); break;
	case 7: r = 1.0f / v; break;
	case 8: r = exp(v); break;
	case 9: r = -v; break;
	case 10: r = fabs(v); break;
	default: r = erf_ew(v); break;                                     // erf
	}
	o[i] = r;
}

// Transpose (rank ≤ 6): out index → output coords (dims) → input offset
// (strides of the input axis each output axis reads). t[0..5] = output dims,
// t[6..11] = input strides by output axis, t[12] = rank, t[13] = n.
kernel void transpose_nd(device const float* x [[buffer(0)]], device float* o [[buffer(1)]],
                         constant uint* t [[buffer(2)]], uint i [[thread_position_in_grid]]) {
	if (i >= t[13]) return;
	uint rem = i, off = 0;
	for (int d = int(t[12]) - 1; d >= 0; d--) {
		off += (rem % t[d]) * t[6 + d];
		rem /= t[d];
	}
	o[i] = x[off];
}

// dst[r, c] = src[r, c] over [rows, cols] with row strides; p = (cols, lds, ldd, 0).
kernel void copy2d(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                   constant uint4& p [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
	if (i.x >= p.x) return;
	dst[i.y * p.z + i.x] = src[i.y * p.y + i.x];
}

// dst[r, :] = src[idx[r], :] with indices passed as constant data.
kernel void gather_rows_c(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                          constant uint* idx [[buffer(2)]], constant uint4& p [[buffer(3)]],
                          uint2 i [[thread_position_in_grid]]) {
	if (i.x >= p.x) return;
	dst[i.y * p.x + i.x] = src[idx[i.y] * p.x + i.x];
}

// o[r] = mean or sum of x[r, :cols]; p = (cols, ld, mean, 0).
kernel void reduce_rows(device const float* x [[buffer(0)]], device float* o [[buffer(1)]],
                        constant uint4& p [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
	threadgroup float scratch[8];
	device const float* xr = x + row * p.y;
	float acc = 0;
	for (uint i = tid; i < p.x; i += 256) acc += xr[i];
	acc = simd_sum(acc);
	if (lane == 0) scratch[sg] = acc;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (tid == 0) {
		float s = 0;
		for (int k = 0; k < 8; k++) s += scratch[k];
		o[row] = p.z != 0 ? s / p.x : s;
	}
}
`

// Binary and unary op codes.
const (
	OpAdd = iota
	OpSub
	OpMul
	OpDiv
	OpPow
	OpMax
	OpMin
)

const (
	UnRelu = iota
	UnSigmoid
	UnTanh
	UnGeluErf
	UnGeluTanh
	UnSiLU
	UnSqrt
	UnReciprocal
	UnExp
	UnNeg
	UnAbs
	UnErf
)

var ewPSO struct {
	once                                          sync.Once
	binary, unary, transp, redux, copy2d, gatherC *Pipeline
	err                                           error
}

// PrepareEW compiles the elementwise kernels.
func (d *Device) PrepareEW() error {
	ewPSO.once.Do(func() {
		for _, k := range []struct {
			name string
			dst  **Pipeline
		}{{"binary_bcast", &ewPSO.binary}, {"unary_ew", &ewPSO.unary}, {"transpose_nd", &ewPSO.transp}, {"reduce_rows", &ewPSO.redux},
			{"copy2d", &ewPSO.copy2d}, {"gather_rows_c", &ewPSO.gatherC}} {
			if *k.dst, ewPSO.err = d.Compile(ewSrc, k.name); ewPSO.err != nil {
				return
			}
		}
	})
	return ewPSO.err
}

// Binary writes out[i] = op(a[i % na], b[i % nb]) for i < n (na, nb divide n:
// either operand may repeat as a trailing block or be a scalar).
func (e *Encoder) Binary(op int, a, b, out Region, n, na, nb int) {
	if na <= 0 || nb <= 0 || n%na != 0 || n%nb != 0 {
		e.err = fmt.Errorf("metal: Binary n=%d na=%d nb=%d", n, na, nb)
		return
	}
	e.BinaryBcast(op, a, b, out, n, 1, na, 1, nb)
}

// BinaryBcast writes out[i] = op(a[(i/da) % ma], b[(i/db) % mb]) for i < n.
func (e *Encoder) BinaryBcast(op int, a, b, out Region, n, da, ma, db, mb int) {
	if da <= 0 || ma <= 0 || db <= 0 || mb <= 0 {
		e.err = fmt.Errorf("metal: BinaryBcast strides %d %d %d %d", da, ma, db, mb)
		return
	}
	if e.ready(ewPSO.binary) {
		e.Dispatch(ewPSO.binary, [3]int{n, 1, 1}, [3]int{256, 1, 1}, a, b, out, u32s(n, da, ma, op), u32s(db, mb, 0, 0))
	}
}

// Copy2D copies [rows, cols] from src (row stride lds) to dst (ldd).
func (e *Encoder) Copy2D(src, dst Region, rows, cols, lds, ldd int) {
	if e.ready(ewPSO.copy2d) {
		e.Dispatch(ewPSO.copy2d, [3]int{cols, rows, 1}, [3]int{256, 1, 1}, src, dst, u32s(cols, lds, ldd, 0))
	}
}

// GatherRowsConst writes dst[r] = src[idx[r]] (rows of cols floats) with at
// most 1024 indices, passed as constant data.
func (e *Encoder) GatherRowsConst(src, dst Region, idx []uint32, cols int) {
	if len(idx) == 0 || len(idx) > 1024 {
		e.err = fmt.Errorf("metal: GatherRowsConst with %d indices", len(idx))
		return
	}
	b := make([]byte, 4*len(idx))
	for i, v := range idx {
		b[4*i], b[4*i+1], b[4*i+2], b[4*i+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	}
	if e.ready(ewPSO.gatherC) {
		e.Dispatch(ewPSO.gatherC, [3]int{cols, len(idx), 1}, [3]int{256, 1, 1}, src, dst, b, u32s(cols, 0, 0, 0))
	}
}

// Unary writes out[i] = op(x[i]) for i < n (out may alias x).
func (e *Encoder) Unary(op int, x, out Region, n int) {
	if e.ready(ewPSO.unary) {
		e.Dispatch(ewPSO.unary, [3]int{n, 1, 1}, [3]int{256, 1, 1}, x, out, u32s(n, op, 0, 0))
	}
}

// Transpose writes x (shape dims, row-major) permuted by perm into out.
func (e *Encoder) Transpose(x, out Region, dims, perm []int) {
	r := len(dims)
	if r == 0 || r > 6 || len(perm) != r {
		e.err = fmt.Errorf("metal: Transpose rank %d", r)
		return
	}
	stride := make([]int, r)
	acc := 1
	for d := r - 1; d >= 0; d-- {
		stride[d] = acc
		acc *= dims[d]
	}
	t := make([]int, 16)
	for d := range r {
		t[d] = dims[perm[d]]
		t[6+d] = stride[perm[d]]
	}
	t[12], t[13] = r, acc
	if e.ready(ewPSO.transp) {
		e.Dispatch(ewPSO.transp, [3]int{acc, 1, 1}, [3]int{256, 1, 1}, x, out, u32s(t...))
	}
}

// ReduceRows writes the mean (or sum) of each of rows rows of cols.
func (e *Encoder) ReduceRows(x, out Region, rows, cols, ld int, mean bool) {
	m := 0
	if mean {
		m = 1
	}
	if e.ready(ewPSO.redux) {
		e.Dispatch(ewPSO.redux, [3]int{rows * 256, 1, 1}, [3]int{256, 1, 1}, x, out, u32s(cols, ld, m, 0))
	}
}
