//go:build darwin && arm64

package metal

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
)

// Row-wise and elementwise kernels for transformer blocks. Row kernels run
// one threadgroup (256 threads) per row and reduce with simd_sum across 8
// simdgroups; all arithmetic is f32.
const nnSrc = `
#include <metal_stdlib>
using namespace metal;
constant uint NT = 256;

// Sum over the threadgroup (8 simdgroups of 32).
static float tg_sum(float v, threadgroup float* scratch, uint tid, uint sg, uint lane) {
	v = simd_sum(v);
	if (lane == 0) scratch[sg] = v;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float t = lane < NT / 32 ? scratch[lane] : 0.0f;
	t = simd_sum(t);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return t;
}
static float tg_max(float v, threadgroup float* scratch, uint tid, uint sg, uint lane) {
	v = simd_max(v);
	if (lane == 0) scratch[sg] = v;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float t = lane < NT / 32 ? scratch[lane] : -INFINITY;
	t = simd_max(t);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return t;
}

// p = (cols, ldx, ldy, 0); f = (eps, ...). y = (x - mean) / sqrt(var + eps) * s,
// written as T (f32, or bf16 straight into a GEMM operand).
template <typename T>
void layernorm_mod_t(device const float* x, device T* y, device const float* s, constant uint4& p,
                     constant float4& f, uint row, uint tid, uint sg, uint lane, threadgroup float* scratch) {
	const uint n = p.x;
	device const float* xr = x + row * p.y;
	device T* yr = y + row * p.z;
	float acc = 0;
	for (uint i = tid; i < n; i += NT) acc += xr[i];
	const float mean = tg_sum(acc, scratch, tid, sg, lane) / n;
	acc = 0;
	for (uint i = tid; i < n; i += NT) { float d = xr[i] - mean; acc += d * d; }
	const float inv = rsqrt(tg_sum(acc, scratch, tid, sg, lane) / n + f.x);
	for (uint i = tid; i < n; i += NT) yr[i] = T((xr[i] - mean) * inv * s[i]);
}
#define LN(name, T) \
kernel void name(device const float* x [[buffer(0)]], device T* y [[buffer(1)]], device const float* s [[buffer(2)]], \
                 constant uint4& p [[buffer(3)]], constant float4& f [[buffer(4)]], uint row [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]]) { \
	threadgroup float scratch[NT / 32]; \
	layernorm_mod_t<T>(x, y, s, p, f, row, tid, sg, lane, scratch); }
LN(layernorm_mod, float)
LN(layernorm_mod_bf16, bfloat)

// In place on x [T, ld]: each (t, head) row of dh at column head*dh is
// RMS-normalised (·w) then rotated as complex pairs (2j, 2j+1) by
// cos/sin [T, dh/2]. One threadgroup of dh threads per (t, head).
// p = (heads, dh, ld, mode); mode 0 rotates interleaved pairs, 1 halves
// (rotate_half, Llama/Qwen); f = (eps, ...).
template <typename T>
void rmsnorm_rope_t(device float* x, device T* out, device const float* w, device const float* cs,
                    device const float* sn, constant uint4& p, constant float4& f, uint2 g, uint tid,
                    uint sg, uint lane, uint nthr, threadgroup float* scratch, threadgroup float* row) {
	const uint dh = p.y, t = g.y, h = g.x;
	device float* xr = x + t * p.z + h * dh;
	device T* orow = out + t * p.z + h * dh;
	float v = tid < dh ? xr[tid] : 0.0f;
	float ss = simd_sum(v * v);
	if (lane == 0) scratch[sg] = ss;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float tot = 0;
	for (uint i = 0; i < (nthr + 31) / 32; i++) tot += scratch[i];
	const float inv = rsqrt(tot / dh + f.x);
	if (tid < dh) row[tid] = v * inv * w[tid];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (tid < dh) {
		const uint hd = dh / 2;
		if (p.w == 0) { // interleaved pairs (2j, 2j+1)
			const uint j = tid / 2;
			const float c = cs[t * hd + j], s = sn[t * hd + j];
			const float re = row[2 * j], im = row[2 * j + 1];
			orow[tid] = T((tid % 2 == 0) ? re * c - im * s : re * s + im * c);
		} else { // rotate_half: (j, j + dh/2)
			const uint j = tid % hd;
			const float c = cs[t * hd + j], s = sn[t * hd + j];
			orow[tid] = T(tid < hd ? row[j] * c - row[j + hd] * s : row[j + hd] * c + row[j] * s);
		}
	}
}
#define RR(name, T) \
kernel void name(device float* x [[buffer(0)]], device const float* w [[buffer(1)]], device const float* cs [[buffer(2)]], \
                 device const float* sn [[buffer(3)]], constant uint4& p [[buffer(4)]], constant float4& f [[buffer(5)]], \
                 device T* out [[buffer(6)]], uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
                 uint2 nthr2 [[threads_per_threadgroup]]) { \
	threadgroup float scratch[NT / 32]; \
	threadgroup float row[512]; \
	rmsnorm_rope_t<T>(x, out, w, cs, sn, p, f, g, tid, sg, lane, nthr2.x, scratch, row); }
RR(rmsnorm_rope, float)
RR(rmsnorm_rope_bf16, bfloat)

// y[r] = x[r] · rsqrt(mean(x[r]²) + eps) · w over cols; p = (cols, ldx, ldy, 0).
kernel void rmsnorm_rows(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                         device const float* w [[buffer(2)]], constant uint4& p [[buffer(3)]],
                         constant float4& f [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
	threadgroup float scratch[NT / 32];
	const uint n = p.x;
	device const float* xr = x + row * p.y;
	device float* yr = y + row * p.z;
	float acc = 0;
	for (uint i = tid; i < n; i += NT) acc += xr[i] * xr[i];
	const float inv = rsqrt(tg_sum(acc, scratch, tid, sg, lane) / n + f.x);
	for (uint i = tid; i < n; i += NT) yr[i] = xr[i] * inv * w[i];
}

// In place softmax(scale · row) over cols of each row (stride ld).
// p = (cols, ld, 0, 0); f = (scale, ...).
kernel void softmax_rows(device float* x [[buffer(0)]], constant uint4& p [[buffer(1)]],
                         constant float4& f [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
	threadgroup float scratch[NT / 32];
	const uint n = p.x;
	device float* xr = x + row * p.y;
	float m = -INFINITY;
	for (uint i = tid; i < n; i += NT) m = max(m, xr[i] * f.x);
	m = tg_max(m, scratch, tid, sg, lane);
	float acc = 0;
	for (uint i = tid; i < n; i += NT) { float e = exp(xr[i] * f.x - m); xr[i] = e; acc += e; }
	const float inv = 1.0f / tg_sum(acc, scratch, tid, sg, lane);
	for (uint i = tid; i < n; i += NT) xr[i] *= inv;
}

// softmax(scale · row) of f32 x into bf16 p, three read passes and one
// bf16 write (vs softmax_rows + cast_bf16's five passes): the attention
// probabilities feeding a bf16 P·V GEMM. p = (cols, ldx, ldp, 0).
kernel void softmax_rows_bf16(device const float* x [[buffer(0)]], device bfloat* o [[buffer(1)]],
                              constant uint4& p [[buffer(2)]], constant float4& f [[buffer(3)]],
                              uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
	threadgroup float scratch[NT / 32];
	const uint n = p.x;
	device const float* xr = x + row * p.y;
	device bfloat* orow = o + row * p.z;
	float m = -INFINITY;
	for (uint i = tid; i < n; i += NT) m = max(m, xr[i] * f.x);
	m = tg_max(m, scratch, tid, sg, lane);
	float acc = 0;
	for (uint i = tid; i < n; i += NT) acc += exp(xr[i] * f.x - m);
	const float inv = 1.0f / tg_sum(acc, scratch, tid, sg, lane);
	for (uint i = tid; i < n; i += NT) orow[i] = bfloat(exp(xr[i] * f.x - m) * inv);
}

// Masked variant: softmax(scale · row + mask[row]) with mask rows of stride
// p.z (additive, -inf masks).
kernel void softmax_rows_masked(device float* x [[buffer(0)]], device const float* mask [[buffer(1)]],
                                constant uint4& p [[buffer(2)]], constant float4& f [[buffer(3)]],
                                uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
	threadgroup float scratch[NT / 32];
	const uint n = p.x;
	device float* xr = x + row * p.y;
	device const float* mr = mask + row * p.z;
	float m = -INFINITY;
	for (uint i = tid; i < n; i += NT) m = max(m, xr[i] * f.x + mr[i]);
	m = tg_max(m, scratch, tid, sg, lane);
	float acc = 0;
	for (uint i = tid; i < n; i += NT) { float e = exp(xr[i] * f.x + mr[i] - m); xr[i] = e; acc += e; }
	const float inv = 1.0f / tg_sum(acc, scratch, tid, sg, lane);
	for (uint i = tid; i < n; i += NT) xr[i] *= inv;
}

// dst[r, :] = src[idx[r], :] over cols; p = (cols, lds, ldd, 0).
kernel void gather_rows(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                        device const uint* idx [[buffer(2)]], constant uint4& p [[buffer(3)]],
                        uint2 i [[thread_position_in_grid]]) {
	if (i.x >= p.x) return;
	dst[i.y * p.z + i.x] = src[idx[i.y] * p.y + i.x];
}

// dst[idx[r], :] += src[r, :] over cols; p = (cols, ldd, lds, 0).
kernel void scatter_add_rows(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
                             device const uint* idx [[buffer(2)]], constant uint4& p [[buffer(3)]],
                             uint2 i [[thread_position_in_grid]]) {
	if (i.x >= p.x) return;
	dst[idx[i.y] * p.y + i.x] += src[i.y * p.z + i.x];
}

// out[r, c] = silu(a[r, c]) · b[r, c] over [rows, cols] with row strides
// p = (cols, lda, ldb, ldo).
template <typename T>
void silu_mul_t(device const float* a, device const float* b, device T* o, constant uint4& p, uint2 i) {
	if (i.x >= p.x) return;
	const float v = a[i.y * p.y + i.x];
	o[i.y * p.w + i.x] = T(v / (1.0f + exp(-v)) * b[i.y * p.z + i.x]);
}
kernel void silu_mul(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* o [[buffer(2)]],
                     constant uint4& p [[buffer(3)]], uint2 i [[thread_position_in_grid]]) { silu_mul_t<float>(a, b, o, p, i); }
kernel void silu_mul_bf16(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device bfloat* o [[buffer(2)]],
                          constant uint4& p [[buffer(3)]], uint2 i [[thread_position_in_grid]]) { silu_mul_t<bfloat>(a, b, o, p, i); }

// x[r, c] += g[c] · y[r, c]; p = (cols, ldx, ldy, 0).
kernel void gated_add(device float* x [[buffer(0)]], device const float* g [[buffer(1)]],
                      device const float* y [[buffer(2)]], constant uint4& p [[buffer(3)]],
                      uint2 i [[thread_position_in_grid]]) {
	if (i.x >= p.x) return;
	x[i.y * p.y + i.x] += g[i.x] * y[i.y * p.z + i.x];
}
`

var nnPSO struct {
	once                                                               sync.Once
	layerNorm, rmsRope, softmax, softmaxMask, siluMul, gateAdd, gather *Pipeline
	rmsRows, softmaxBF16, layerNormBF16, rmsRopeBF16, siluMulBF16      *Pipeline
	scatterAdd                                                         *Pipeline
	err                                                                error
}

func (d *Device) nnPipelines() error {
	nnPSO.once.Do(func() {
		for _, k := range []struct {
			name string
			dst  **Pipeline
		}{{"layernorm_mod", &nnPSO.layerNorm}, {"rmsnorm_rope", &nnPSO.rmsRope}, {"softmax_rows", &nnPSO.softmax},
			{"silu_mul", &nnPSO.siluMul}, {"gated_add", &nnPSO.gateAdd},
			{"softmax_rows_masked", &nnPSO.softmaxMask}, {"gather_rows", &nnPSO.gather},
			{"rmsnorm_rows", &nnPSO.rmsRows}, {"softmax_rows_bf16", &nnPSO.softmaxBF16},
			{"layernorm_mod_bf16", &nnPSO.layerNormBF16}, {"rmsnorm_rope_bf16", &nnPSO.rmsRopeBF16},
			{"silu_mul_bf16", &nnPSO.siluMulBF16}, {"scatter_add_rows", &nnPSO.scatterAdd}} {
			if *k.dst, nnPSO.err = d.Compile(nnSrc, k.name); nnPSO.err != nil {
				return
			}
		}
	})
	return nnPSO.err
}

func f32c(v ...float32) []byte {
	b := make([]byte, 0, 16)
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(x))
	}
	for len(b) < 16 {
		b = append(b, 0)
	}
	return b
}

func (e *Encoder) ready(p *Pipeline) bool {
	if e.err == nil && p == nil {
		e.err = fmt.Errorf("metal: kernel used before Device.Prepare")
	}
	return e.err == nil
}

// LayerNormMod: y[r] = LayerNorm(x[r]) · s over rows of cols (no affine;
// s is a per-column scale, e.g. 1+modulation). Row strides in elements.
func (e *Encoder) LayerNormMod(x, y, s Region, rows, cols, ldx, ldy int, eps float32) {
	if e.ready(nnPSO.layerNorm) {
		e.Dispatch(nnPSO.layerNorm, [3]int{rows * 256, 1, 1}, [3]int{256, 1, 1}, x, y, s, u32s(cols, ldx, ldy, 0), f32c(eps))
	}
}

// RoPE layouts for RMSNormRoPE.
const (
	RopePairs = 0 // rotate channel pairs (2j, 2j+1) — complex view
	RopeHalf  = 1 // rotate (j, j+dh/2) — rotate_half
)

// RMSNormRoPE normalises each dh-wide head of x [T, ld] (heads starting at
// x's offset) with weight w [dh], then rotates channel pairs (2j, 2j+1) by
// cos/sin [T, dh/2] — in place. dh ≤ 512, even.
func (e *Encoder) RMSNormRoPE(x, w, cos, sin Region, t, heads, dh, ld int, eps float32) {
	e.RMSNormRoPEMode(x, w, cos, sin, t, heads, dh, ld, eps, RopePairs)
}

// RMSNormRoPEMode is RMSNormRoPE with a choice of rotary layout.
func (e *Encoder) RMSNormRoPEMode(x, w, cos, sin Region, t, heads, dh, ld int, eps float32, mode int) {
	if dh > 512 || dh%2 != 0 {
		e.err = fmt.Errorf("metal: RMSNormRoPE head dim %d", dh)
		return
	}
	if e.ready(nnPSO.rmsRope) {
		e.Dispatch(nnPSO.rmsRope, [3]int{heads * dh, t, 1}, [3]int{dh, 1, 1}, x, w, cos, sin, u32s(heads, dh, ld, mode), f32c(eps), x)
	}
}

// LayerNormModBF16 is LayerNormMod writing bf16 (a GEMM's activation
// operand) instead of f32.
func (e *Encoder) LayerNormModBF16(x, y, s Region, rows, cols, ldx, ldy int, eps float32) {
	if e.ready(nnPSO.layerNormBF16) {
		e.Dispatch(nnPSO.layerNormBF16, [3]int{rows * 256, 1, 1}, [3]int{256, 1, 1}, x, y, s, u32s(cols, ldx, ldy, 0), f32c(eps))
	}
}

// RMSNormRoPEBF16 is RMSNormRoPEMode reading f32 x and writing the result
// as bf16 into out (same layout), leaving x untouched.
func (e *Encoder) RMSNormRoPEBF16(x, out, w, cos, sin Region, t, heads, dh, ld int, eps float32, mode int) {
	if dh > 512 || dh%2 != 0 {
		e.err = fmt.Errorf("metal: RMSNormRoPE head dim %d", dh)
		return
	}
	if e.ready(nnPSO.rmsRopeBF16) {
		e.Dispatch(nnPSO.rmsRopeBF16, [3]int{heads * dh, t, 1}, [3]int{dh, 1, 1}, x, w, cos, sin, u32s(heads, dh, ld, mode), f32c(eps), out)
	}
}

// SiLUMulBF16 is SiLUMul writing bf16.
func (e *Encoder) SiLUMulBF16(a, b, o Region, rows, cols, lda, ldb, ldo int) {
	if e.ready(nnPSO.siluMulBF16) {
		e.Dispatch(nnPSO.siluMulBF16, [3]int{cols, rows, 1}, [3]int{256, 1, 1}, a, b, o, u32s(cols, lda, ldb, ldo))
	}
}

// RMSNormRows: y[r] = x[r] · rsqrt(mean(x[r]²) + eps) · w over rows of cols.
func (e *Encoder) RMSNormRows(x, y, w Region, rows, cols, ldx, ldy int, eps float32) {
	if e.ready(nnPSO.rmsRows) {
		e.Dispatch(nnPSO.rmsRows, [3]int{rows * 256, 1, 1}, [3]int{256, 1, 1}, x, y, w, u32s(cols, ldx, ldy, 0), f32c(eps))
	}
}

// SoftmaxRows applies softmax(scale · x) to each of rows rows in place.
func (e *Encoder) SoftmaxRows(x Region, rows, cols, ld int, scale float32) {
	if e.ready(nnPSO.softmax) {
		e.Dispatch(nnPSO.softmax, [3]int{rows * 256, 1, 1}, [3]int{256, 1, 1}, x, u32s(cols, ld, 0, 0), f32c(scale))
	}
}

// SoftmaxRowsBF16 writes softmax(scale · x) of f32 rows as bf16 into out
// (ld/ldo row strides in elements).
func (e *Encoder) SoftmaxRowsBF16(x, out Region, rows, cols, ld, ldo int, scale float32) {
	if e.ready(nnPSO.softmaxBF16) {
		e.Dispatch(nnPSO.softmaxBF16, [3]int{rows * 256, 1, 1}, [3]int{256, 1, 1}, x, out, u32s(cols, ld, ldo, 0), f32c(scale))
	}
}

// SoftmaxRowsMasked applies softmax(scale · x + mask) row by row in place;
// mask rows (additive, -inf to mask) have stride ldm.
func (e *Encoder) SoftmaxRowsMasked(x, mask Region, rows, cols, ld, ldm int, scale float32) {
	if e.ready(nnPSO.softmaxMask) {
		e.Dispatch(nnPSO.softmaxMask, [3]int{rows * 256, 1, 1}, [3]int{256, 1, 1}, x, mask, u32s(cols, ld, ldm, 0), f32c(scale))
	}
}

// GatherRows: dst[r] = src[idx[r]] for r < rows (idx: uint32 row indices).
func (e *Encoder) GatherRows(src, dst, idx Region, rows, cols, lds, ldd int) {
	if e.ready(nnPSO.gather) {
		e.Dispatch(nnPSO.gather, [3]int{cols, rows, 1}, [3]int{256, 1, 1}, src, dst, idx, u32s(cols, lds, ldd, 0))
	}
}

// ScatterAddRows: dst[idx[r]] += src[r] for r < rows (idx: uint32 rows of
// dst, distinct).
func (e *Encoder) ScatterAddRows(dst, src, idx Region, rows, cols, ldd, lds int) {
	if e.ready(nnPSO.scatterAdd) {
		e.Dispatch(nnPSO.scatterAdd, [3]int{cols, rows, 1}, [3]int{256, 1, 1}, dst, src, idx, u32s(cols, ldd, lds, 0))
	}
}

// SiLUMul: o = silu(a) · b over [rows, cols] (o may alias a or b).
func (e *Encoder) SiLUMul(a, b, o Region, rows, cols, lda, ldb, ldo int) {
	if e.ready(nnPSO.siluMul) {
		e.Dispatch(nnPSO.siluMul, [3]int{cols, rows, 1}, [3]int{256, 1, 1}, a, b, o, u32s(cols, lda, ldb, ldo))
	}
}

// GatedAdd: x += g ⊙ y over [rows, cols], g a per-column vector.
func (e *Encoder) GatedAdd(x, g, y Region, rows, cols, ldx, ldy int) {
	if e.ready(nnPSO.gateAdd) {
		e.Dispatch(nnPSO.gateAdd, [3]int{cols, rows, 1}, [3]int{256, 1, 1}, x, g, y, u32s(cols, ldx, ldy, 0))
	}
}
