//go:build darwin && arm64

package metal

import (
	"fmt"
	"sync"
)

// Decode kernels: the m = 1..4 shapes of autoregressive generation, where
// every weight is read once per token and the GEMM tile machinery idles.
//
// gemv_*: y[r, n] = Σ_k x[r, k] · W[n, k] (+ bias[n]) (+ y[r, n] when
// accumulating — a residual add for free) for R ≤ 4 activation rows sharing
// one pass over W. One simdgroup per output n; each lane reads 8 weights a
// step (coalesced 512 B per simdgroup for bf16), simd_sum reduces.
// W is bf16 [N, K] (PyTorch Linear layout) or int8 with one f32 scale per
// 64 weights (gemm.I8GWeights' layout) dequantised in registers.
//
// sample_logits: one token from a logits row on the GPU, so a loop of
// dependent decode steps can stay inside one command buffer — argmax, or
// temperature + exact top-k (ties at the k-th value kept) and a draw with
// a host-supplied uniform in index order.
//
// embed_row: dst = widen(bf16 table[code]) for a code a previous kernel
// wrote.
//
// attn_decode: causal attention of T new queries over a KV cache kept one
// row per position ([pos, KV·dh]), grouped-query heads, online softmax
// split across simdgroups and merged through threadgroup memory — no score
// matrix, any cache length.
const decodeSrc = `
#include <metal_stdlib>
using namespace metal;
constant uint SG = 8; // simdgroups per threadgroup

// p = (N, K, ldw, R); q = (ldx, ldy, flags, 0): flags bit 0 accumulate
// into y, bit 1 add bias. Each lane takes 16 consecutive weights a step
// (512 B of int8 / 1 KiB of bf16 per simdgroup), K a multiple of 16.
template <typename LOAD>
void gemv(device const float* x, device float* y, device const float* bias, constant uint4& p,
          constant uint4& q, uint tg, uint sg, uint lane, LOAD load) {
	const uint n = tg * SG + sg;
	if (n >= p.x) return;
	const uint K = p.y, R = p.w;
	float acc[4] = {0, 0, 0, 0};
	for (uint k = lane * 16; k < K; k += 512) {
		float4 w[4];
		load(n, k, w);
		for (uint r = 0; r < R; r++) {
			device const float4* xr = (device const float4*)(x + r * q.x + k);
			acc[r] += dot(w[0], xr[0]) + dot(w[1], xr[1]) + dot(w[2], xr[2]) + dot(w[3], xr[3]);
		}
	}
	for (uint r = 0; r < R; r++) {
		float s = simd_sum(acc[r]);
		if (lane == 0) {
			if (q.z & 2u) s += bias[n];
			device float* yr = y + r * q.y + n;
			*yr = (q.z & 1u) ? *yr + s : s;
		}
	}
}

kernel void gemv_bf16(device const float* x [[buffer(0)]], device const bfloat* W [[buffer(1)]],
                      device float* y [[buffer(2)]], device const float* bias [[buffer(3)]],
                      constant uint4& p [[buffer(4)]], constant uint4& q [[buffer(5)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
	gemv(x, y, bias, p, q, tg, sg, lane, [&](uint n, uint k, thread float4* w) {
		device const bfloat4* src = (device const bfloat4*)(W + n * p.z + k);
		for (uint i = 0; i < 4; i++) w[i] = float4(src[i]);
	});
}

// Int8 weights W [N, K] (row stride ldw = p.z) with scales S [N, K/64].
kernel void gemv_q8(device const float* x [[buffer(0)]], device const char* W [[buffer(1)]],
                    device float* y [[buffer(2)]], device const float* bias [[buffer(3)]],
                    constant uint4& p [[buffer(4)]], constant uint4& q [[buffer(5)]],
                    device const float* S [[buffer(6)]],
                    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
	gemv(x, y, bias, p, q, tg, sg, lane, [&](uint n, uint k, thread float4* w) {
		device const char4* src = (device const char4*)(W + n * p.z + k);
		const float s = S[n * (p.y / 64) + k / 64];
		for (uint i = 0; i < 4; i++) w[i] = float4(src[i]) * s;
	});
}

// Staged one-row GEMVs: x [K ≤ 8192] is loaded once into threadgroup
// memory and shared by 32 simdgroups (outputs), instead of every lane
// re-reading it from device memory (4× the weight bytes for int8).
template <typename LOAD>
void gemv_staged(device const float* x, device float* y, device const float* bias, constant uint4& p,
                 constant uint4& q, uint tg, uint sg, uint lane, uint tid, threadgroup float4* xs, LOAD load) {
	const uint K = p.y;
	for (uint i = tid; i < K / 4; i += 1024) xs[i] = ((device const float4*)x)[i];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	const uint n = tg * 32 + sg;
	if (n >= p.x) return;
	float acc = 0;
	for (uint k = lane * 16; k < K; k += 512) acc += load(n, k, xs + k / 4);
	float t = simd_sum(acc);
	if (lane == 0) {
		if (q.z & 2u) t += bias[n];
		device float* yr = y + n;
		*yr = (q.z & 1u) ? *yr + t : t;
	}
}

kernel void gemv_q8s(device const float* x [[buffer(0)]], device const char* W [[buffer(1)]],
                     device float* y [[buffer(2)]], device const float* bias [[buffer(3)]],
                     constant uint4& p [[buffer(4)]], constant uint4& q [[buffer(5)]],
                     device const float* S [[buffer(6)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
	threadgroup float4 xs[2048];
	gemv_staged(x, y, bias, p, q, tg, sg, lane, tid, xs, [&](uint n, uint k, threadgroup float4* xv) {
		device const char4* w = (device const char4*)(W + n * p.z + k);
		float a = 0;
		for (uint i = 0; i < 4; i++) a += dot(float4(w[i]), xv[i]);
		return a * S[n * (p.y / 64) + k / 64];
	});
}

kernel void gemv_bf16s(device const float* x [[buffer(0)]], device const bfloat* W [[buffer(1)]],
                       device float* y [[buffer(2)]], device const float* bias [[buffer(3)]],
                       constant uint4& p [[buffer(4)]], constant uint4& q [[buffer(5)]],
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
	threadgroup float4 xs[2048];
	gemv_staged(x, y, bias, p, q, tg, sg, lane, tid, xs, [&](uint n, uint k, threadgroup float4* xv) {
		device const bfloat4* w = (device const bfloat4*)(W + n * p.z + k);
		float a = 0;
		for (uint i = 0; i < 4; i++) a += dot(float4(w[i]), xv[i]);
		return a;
	});
}

// Fused one-row decode GEMVs (gemv_f_*, glu_f_*): x staged in threadgroup
// memory, optionally RMS-normalised on the way in (x·rsqrt(mean x²+eps)·nw
// — the pre-projection norm costs no dispatch); gemv_f splits its outputs
// over three destinations (n < n1 → y0, n < n2 → y1[n-n1], else
// y2[n-n2]: q, the K-cache row, the V-cache row from one QKV weight); glu_f
// computes y[n] = silu(Wg[n]·x)·(Wu[n]·x) from two weights in one pass.
// p = (N, K, n1, n2); q = (flags, ldw, 0, 0): flags bit 0 accumulate into
// the output, bit 2 normalise x; f = (eps).
static float stage_x(device const float* x, device const float* nw, constant uint4& q, constant float4& f,
                     uint K, uint tid, uint sg, uint lane, threadgroup float4* xs, threadgroup float* red) {
	float ss = 0;
	for (uint i = tid; i < K / 4; i += 1024) {
		const float4 v = ((device const float4*)x)[i];
		xs[i] = v;
		ss += dot(v, v);
	}
	if ((q.x & 4u) == 0) {
		threadgroup_barrier(mem_flags::mem_threadgroup);
		return 1;
	}
	ss = simd_sum(ss);
	if (lane == 0) red[sg] = ss;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float t = red[lane];
	t = simd_sum(t);
	const float inv = rsqrt(t / K + f.x);
	for (uint i = tid; i < K / 4; i += 1024) xs[i] = xs[i] * inv * ((device const float4*)nw)[i];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return inv;
}

template <typename DOT>
void gemv_f(device const float* x, device const float* nw, device float* y0, device float* y1, device float* y2,
            constant uint4& p, constant uint4& q, constant float4& f, uint tg, uint sg, uint lane, uint tid,
            threadgroup float4* xs, threadgroup float* red, DOT dotw) {
	const uint K = p.y;
	stage_x(x, nw, q, f, K, tid, sg, lane, xs, red);
	const uint n = tg * 32 + sg;
	if (n >= p.x) return;
	float acc = 0;
	for (uint k = lane * 16; k < K; k += 512) acc += dotw(n, k, xs + k / 4);
	const float t = simd_sum(acc);
	if (lane == 0) {
		device float* yr = n < p.z ? y0 + n : (n < p.w ? y1 + (n - p.z) : y2 + (n - p.w));
		*yr = (q.x & 1u) ? *yr + t : t;
	}
}

template <typename DOT>
void glu_f(device const float* x, device const float* nw, device float* y, constant uint4& p, constant uint4& q,
           constant float4& f, uint tg, uint sg, uint lane, uint tid, threadgroup float4* xs, threadgroup float* red,
           DOT dotw) {
	const uint K = p.y;
	stage_x(x, nw, q, f, K, tid, sg, lane, xs, red);
	const uint n = tg * 32 + sg;
	if (n >= p.x) return;
	float2 acc = 0;
	for (uint k = lane * 16; k < K; k += 512) acc += dotw(n, k, xs + k / 4);
	const float g = simd_sum(acc.x), u = simd_sum(acc.y);
	if (lane == 0) y[n] = g / (1.0f + exp(-g)) * u;
}

#define FUSED_ARGS \
	uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
	uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]

kernel void gemv_f_bf16(device const float* x [[buffer(0)]], device const bfloat* W [[buffer(1)]],
                        device float* y0 [[buffer(2)]], device float* y1 [[buffer(3)]], device float* y2 [[buffer(4)]],
                        device const float* nw [[buffer(5)]], constant uint4& p [[buffer(6)]],
                        constant uint4& q [[buffer(7)]], constant float4& f [[buffer(8)]], FUSED_ARGS) {
	threadgroup float4 xs[2040]; // K ≤ 8160: with red, the 32 KiB threadgroup limit
	threadgroup float red[32];
	gemv_f(x, nw, y0, y1, y2, p, q, f, tg, sg, lane, tid, xs, red, [&](uint n, uint k, threadgroup float4* xv) {
		device const bfloat4* w = (device const bfloat4*)(W + n * q.y + k);
		return dot(float4(w[0]), xv[0]) + dot(float4(w[1]), xv[1]) + dot(float4(w[2]), xv[2]) + dot(float4(w[3]), xv[3]);
	});
}

kernel void gemv_f_q8(device const float* x [[buffer(0)]], device const char* W [[buffer(1)]],
                      device float* y0 [[buffer(2)]], device float* y1 [[buffer(3)]], device float* y2 [[buffer(4)]],
                      device const float* nw [[buffer(5)]], constant uint4& p [[buffer(6)]],
                      constant uint4& q [[buffer(7)]], constant float4& f [[buffer(8)]],
                      device const float* S [[buffer(9)]], FUSED_ARGS) {
	threadgroup float4 xs[2040]; // K ≤ 8160: with red, the 32 KiB threadgroup limit
	threadgroup float red[32];
	gemv_f(x, nw, y0, y1, y2, p, q, f, tg, sg, lane, tid, xs, red, [&](uint n, uint k, threadgroup float4* xv) {
		device const char4* w = (device const char4*)(W + n * q.y + k);
		const float a = dot(float4(w[0]), xv[0]) + dot(float4(w[1]), xv[1]) + dot(float4(w[2]), xv[2]) + dot(float4(w[3]), xv[3]);
		return a * S[n * (p.y / 64) + k / 64];
	});
}

kernel void glu_f_bf16(device const float* x [[buffer(0)]], device const bfloat* Wg [[buffer(1)]],
                       device const bfloat* Wu [[buffer(2)]], device float* y [[buffer(3)]],
                       device const float* nw [[buffer(4)]], constant uint4& p [[buffer(5)]],
                       constant uint4& q [[buffer(6)]], constant float4& f [[buffer(7)]], FUSED_ARGS) {
	threadgroup float4 xs[2040]; // K ≤ 8160: with red, the 32 KiB threadgroup limit
	threadgroup float red[32];
	glu_f(x, nw, y, p, q, f, tg, sg, lane, tid, xs, red, [&](uint n, uint k, threadgroup float4* xv) {
		device const bfloat4* g = (device const bfloat4*)(Wg + n * q.y + k);
		device const bfloat4* u = (device const bfloat4*)(Wu + n * q.y + k);
		float2 a = 0;
		for (uint i = 0; i < 4; i++) a += float2(dot(float4(g[i]), xv[i]), dot(float4(u[i]), xv[i]));
		return a;
	});
}

kernel void glu_f_q8(device const float* x [[buffer(0)]], device const char* Wg [[buffer(1)]],
                     device const char* Wu [[buffer(2)]], device float* y [[buffer(3)]],
                     device const float* nw [[buffer(4)]], constant uint4& p [[buffer(5)]],
                     constant uint4& q [[buffer(6)]], constant float4& f [[buffer(7)]],
                     device const float* Sg [[buffer(8)]], device const float* Su [[buffer(9)]], FUSED_ARGS) {
	threadgroup float4 xs[2040]; // K ≤ 8160: with red, the 32 KiB threadgroup limit
	threadgroup float red[32];
	glu_f(x, nw, y, p, q, f, tg, sg, lane, tid, xs, red, [&](uint n, uint k, threadgroup float4* xv) {
		device const char4* g = (device const char4*)(Wg + n * q.y + k);
		device const char4* u = (device const char4*)(Wu + n * q.y + k);
		float2 a = 0;
		for (uint i = 0; i < 4; i++) a += float2(dot(float4(g[i]), xv[i]), dot(float4(u[i]), xv[i]));
		const uint s = n * (p.y / 64) + k / 64;
		return a * float2(Sg[s], Su[s]);
	});
}

// One-token RMSNorm + RoPE of q (heads [0, H)) and k (heads [H, H+KV)) in
// one dispatch, in place; p = (H, KV, dh, mode), f = (eps). One
// threadgroup of dh threads per head; cs/sn are this position's dh/2 row.
kernel void rmsnorm_rope_qk(device float* qx [[buffer(0)]], device float* kx [[buffer(1)]],
                            device const float* qw [[buffer(2)]], device const float* kw [[buffer(3)]],
                            device const float* cs [[buffer(4)]], device const float* sn [[buffer(5)]],
                            constant uint4& p [[buffer(6)]], constant float4& f [[buffer(7)]],
                            uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint nthr [[threads_per_threadgroup]]) {
	threadgroup float scratch[16];
	threadgroup float row[512];
	const uint dh = p.z, hd = dh / 2;
	const bool isq = h < p.x;
	device float* xr = isq ? qx + h * dh : kx + (h - p.x) * dh;
	device const float* w = isq ? qw : kw;
	const float v = tid < dh ? xr[tid] : 0.0f;
	const float ss = simd_sum(v * v);
	if (lane == 0) scratch[sg] = ss;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float tot = 0;
	for (uint i = 0; i < (nthr + 31) / 32; i++) tot += scratch[i];
	const float inv = rsqrt(tot / dh + f.x);
	if (tid < dh) row[tid] = v * inv * w[tid];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (tid >= dh) return;
	if (p.w == 1) { // rotate_half
		const uint j = tid % hd;
		const float c = cs[j], s = sn[j];
		xr[tid] = tid < hd ? row[j] * c - row[j + hd] * s : row[j + hd] * c + row[j] * s;
	} else { // interleaved pairs
		const uint j = tid / 2;
		const float c = cs[j], s = sn[j], re = row[2 * j], im = row[2 * j + 1];
		xr[tid] = (tid % 2 == 0) ? re * c - im * s : re * s + im * c;
	}
}

// q [T, H·dh] (row stride ldq), K/V caches [*, KV·dh], out [T, H·dh] (row
// stride ldo). Query t sits at position pos0 + t and attends keys
// [0, pos0 + t]. p = (H, KV, dh, pos0); p2 = (ldq, ldo, 0, 0); f = (scale).
// dh ≤ 128 (each lane holds dh/32 output channels). One threadgroup of SG
// simdgroups per (head, t); simdgroup s takes keys s, s+SG, ….
kernel void attn_decode(device const float* Q [[buffer(0)]], device const float* Kc [[buffer(1)]],
                        device const float* Vc [[buffer(2)]], device float* O [[buffer(3)]],
                        constant uint4& p [[buffer(4)]], constant uint4& p2 [[buffer(5)]],
                        constant float4& f [[buffer(6)]], uint2 g [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
	const uint H = p.x, KV = p.y, dh = p.z, h = g.x, t = g.y;
	const uint kvh = h / (H / KV), ldk = KV * dh, per = dh / 32;
	const uint n = p.w + t + 1;
	device const float* qr = Q + t * p2.x + h * dh;
	float qv[4], acc[4] = {0, 0, 0, 0};
	for (uint i = 0; i < per; i++) qv[i] = qr[lane + 32 * i];
	float m = -INFINITY, l = 0;
	for (uint j = sg; j < n; j += SG) {
		device const float* kr = Kc + j * ldk + kvh * dh;
		float s = 0;
		for (uint i = 0; i < per; i++) s += qv[i] * kr[lane + 32 * i];
		s = simd_sum(s) * f.x;
		const float mn = max(m, s), c = exp(m - mn), e = exp(s - mn);
		l = l * c + e;
		device const float* vr = Vc + j * ldk + kvh * dh;
		for (uint i = 0; i < per; i++) acc[i] = acc[i] * c + e * vr[lane + 32 * i];
		m = mn;
	}
	threadgroup float tm[SG], tl[SG], ta[SG * 128];
	if (lane == 0) { tm[sg] = m; tl[sg] = l; }
	for (uint i = 0; i < per; i++) ta[sg * 128 + lane + 32 * i] = acc[i];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (sg != 0) return;
	float M = -INFINITY;
	for (uint s = 0; s < SG; s++) M = max(M, tm[s]);
	float L = 0, w[SG];
	for (uint s = 0; s < SG; s++) { w[s] = exp(tm[s] - M); L += tl[s] * w[s]; }
	device float* orow = O + t * p2.y + h * dh;
	for (uint i = 0; i < per; i++) {
		float v = 0;
		for (uint s = 0; s < SG; s++) v += ta[s * 128 + lane + 32 * i] * w[s];
		orow[lane + 32 * i] = v / L;
	}
}

// attn_decode_rope: attn_decode for up to 4 new tokens whose q and k rows
// arrive raw (straight from the QKV GEMV): q and the new k rows are
// RMS-normalised (qw / kw) and rotated (rotate_half or pairs; cos/sin
// rows per position) inside the kernel — no separate norm/RoPE dispatch.
// Keys before pos0 come from the cache (already normalised); the new
// ones are recomputed by every threadgroup from Kn [n, KV·dh], and the
// threadgroups with h % (H/KV) == 0, t == 0 write them, finished, to the
// cache rows pos0.. for later steps (no one reads those rows in this
// dispatch). p = (H, KV, dh, pos0); p2 = (n, mode, 0, 0); f = (scale, eps).
static void norm_rope_lane(thread float* v, uint per, device const float* w, device const float* cs,
                           device const float* sn, uint mode, float eps, uint lane, uint dh) {
	float ss = 0;
	for (uint i = 0; i < per; i++) ss += v[i] * v[i];
	const float inv = rsqrt(simd_sum(ss) / dh + eps);
	for (uint i = 0; i < per; i++) v[i] *= inv * w[lane + 32 * i];
	const uint hp = per / 2;
	if (mode == 1) { // rotate_half: element i pairs with i + per/2 in the same lane
		for (uint i = 0; i < hp; i++) {
			const uint j = lane + 32 * i;
			const float c = cs[j], s = sn[j], a = v[i], b = v[i + hp];
			v[i] = a * c - b * s;
			v[i + hp] = b * c + a * s;
		}
	} else { // pairs (2j, 2j+1): neighbours across lanes
		for (uint i = 0; i < per; i++) {
			const uint e = lane + 32 * i, j = e / 2;
			const float o = simd_shuffle_xor(v[i], 1u);
			const float c = cs[j], s = sn[j];
			v[i] = (e % 2 == 0) ? v[i] * c - o * s : o * s + v[i] * c;
		}
	}
}

kernel void attn_decode_rope(device const float* Q [[buffer(0)]], device float* Kc [[buffer(1)]],
                             device const float* Vc [[buffer(2)]], device float* O [[buffer(3)]],
                             device const float* Kn [[buffer(4)]], device const float* qw [[buffer(5)]],
                             device const float* kw [[buffer(6)]], device const float* COS [[buffer(7)]],
                             device const float* SIN [[buffer(8)]], constant uint4& p [[buffer(9)]],
                             constant uint4& p2 [[buffer(10)]], constant float4& f [[buffer(11)]],
                             uint2 g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
	const uint H = p.x, KV = p.y, dh = p.z, pos0 = p.w, h = g.x, t = g.y, nn = p2.x, mode = p2.y;
	const uint group = H / KV, kvh = h / group, ldk = KV * dh, per = dh / 32, hd = dh / 2;
	const uint n = pos0 + t + 1;
	float qv[4], acc[4] = {0, 0, 0, 0};
	for (uint i = 0; i < per; i++) qv[i] = Q[t * H * dh + h * dh + lane + 32 * i];
	norm_rope_lane(qv, per, qw, COS + (pos0 + t) * hd, SIN + (pos0 + t) * hd, mode, f.y, lane, dh);
	float m = -INFINITY, l = 0;
	for (uint j = sg; j < n; j += SG) {
		float kv[4];
		if (j < pos0) {
			for (uint i = 0; i < per; i++) kv[i] = Kc[j * ldk + kvh * dh + lane + 32 * i];
		} else {
			const uint u = j - pos0;
			for (uint i = 0; i < per; i++) kv[i] = Kn[u * ldk + kvh * dh + lane + 32 * i];
			norm_rope_lane(kv, per, kw, COS + j * hd, SIN + j * hd, mode, f.y, lane, dh);
		}
		float s = 0;
		for (uint i = 0; i < per; i++) s += qv[i] * kv[i];
		s = simd_sum(s) * f.x;
		const float mn = max(m, s), c = exp(m - mn), e = exp(s - mn);
		l = l * c + e;
		device const float* vr = Vc + j * ldk + kvh * dh;
		for (uint i = 0; i < per; i++) acc[i] = acc[i] * c + e * vr[lane + 32 * i];
		m = mn;
	}
	threadgroup float tm[SG], tl[SG], ta[SG * 128];
	if (lane == 0) { tm[sg] = m; tl[sg] = l; }
	for (uint i = 0; i < per; i++) ta[sg * 128 + lane + 32 * i] = acc[i];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (sg == 0) {
		float M = -INFINITY;
		for (uint s = 0; s < SG; s++) M = max(M, tm[s]);
		float L = 0, w[SG];
		for (uint s = 0; s < SG; s++) { w[s] = exp(tm[s] - M); L += tl[s] * w[s]; }
		device float* orow = O + t * H * dh + h * dh;
		for (uint i = 0; i < per; i++) {
			float v = 0;
			for (uint s = 0; s < SG; s++) v += ta[s * 128 + lane + 32 * i] * w[s];
			orow[lane + 32 * i] = v / L;
		}
	}
	// One threadgroup per kv head stores the finished new keys.
	if (h % group == 0 && t == 0 && sg == 1) {
		for (uint u = 0; u < nn; u++) {
			float kv[4];
			for (uint i = 0; i < per; i++) kv[i] = Kn[u * ldk + kvh * dh + lane + 32 * i];
			norm_rope_lane(kv, per, kw, COS + (pos0 + u) * hd, SIN + (pos0 + u) * hd, mode, f.y, lane, dh);
			for (uint i = 0; i < per; i++) Kc[(pos0 + u) * ldk + kvh * dh + lane + 32 * i] = kv[i];
		}
	}
}

constant uint ST = 1024; // sample_logits threads (one threadgroup)

static float tsum(float v, threadgroup float* sc, uint sg, uint lane) {
	v = simd_sum(v);
	if (lane == 0) sc[sg] = v;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float t = lane < ST / 32 ? sc[lane] : 0.0f;
	t = simd_sum(t);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return t;
}

// Order-preserving float → uint key.
static uint fkey(float f) { uint b = as_type<uint>(f); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }

// logits [V]; p = (V, topK, greedy, outIndex); f = (temperature, u, 0, 0);
// writes the token to codes[outIndex]. Each thread owns the contiguous
// indices [tid·per, (tid+1)·per) so the draw runs in index order.
// With p2.y != 0 the chosen code's bf16 embedding row (table [*, p2.x])
// is also written to dst — the next decode step's input, no extra
// dispatch.
kernel void sample_logits(device const float* logits [[buffer(0)]], device uint* codes [[buffer(1)]],
                          constant uint4& p [[buffer(2)]], constant float4& f [[buffer(3)]],
                          device const bfloat* table [[buffer(4)]], device float* dst [[buffer(5)]],
                          constant uint4& p2 [[buffer(6)]],
                          uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
	threadgroup float sc[ST / 32];
	threadgroup uint si[ST / 32];
	threadgroup uint chosen;
	const uint V = p.x, per = (V + ST - 1) / ST, lo = tid * per, hi = min(lo + per, V);
	if (p.z != 0) { // argmax, first index of the maximum
		float bv = -INFINITY; uint bi = 0xffffffffu;
		for (uint i = lo; i < hi; i++) if (logits[i] > bv) { bv = logits[i]; bi = i; }
		const float mv = simd_max(bv);
		uint mi = simd_min(bv == mv ? bi : 0xffffffffu);
		if (lane == 0) { sc[sg] = mv; si[sg] = mi; }
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if (tid == 0) {
			float best = -INFINITY; uint idx = 0xffffffffu;
			for (uint s = 0; s < ST / 32; s++)
				if (sc[s] > best || (sc[s] == best && si[s] < idx)) { best = sc[s]; idx = si[s]; }
			codes[p.w] = idx;
			chosen = idx;
		}
	} else {
	const float temp = f.x > 0 ? f.x : 1.0f;
	// k-th largest key by bitwise radix select: the largest K with
	// count(key >= K) >= k.
	uint K = 0;
	if (p.y > 0 && p.y < V) {
		for (int bit = 31; bit >= 0; bit--) {
			const uint cand = K | (1u << bit);
			float cnt = 0;
			for (uint i = lo; i < hi; i++) cnt += fkey(logits[i] / temp) >= cand ? 1.0f : 0.0f;
			if (tsum(cnt, sc, sg, lane) >= float(p.y)) K = cand;
		}
	}
	float m = -INFINITY;
	for (uint i = lo; i < hi; i++) { const float x = logits[i] / temp; if (fkey(x) >= K) m = max(m, x); }
	m = simd_max(m);
	if (lane == 0) sc[sg] = m;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float M = -INFINITY;
	for (uint s = 0; s < ST / 32; s++) M = max(M, sc[s]);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float part = 0;
	for (uint i = lo; i < hi; i++) { const float x = logits[i] / temp; if (fkey(x) >= K) part += exp(x - M); }
	// Exclusive prefix of the per-thread sums, in thread (= index) order.
	threadgroup float pre[ST];
	pre[tid] = part;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (uint off = 1; off < ST; off <<= 1) {
		const float v = tid >= off ? pre[tid - off] : 0.0f;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		pre[tid] += v;
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	const float total = pre[ST - 1], target = f.y * total;
	const float before = pre[tid] - part;
	if (before <= target && target < pre[tid] && part > 0) {
		float c = before;
		uint pick = hi - 1;
		for (uint i = lo; i < hi; i++) {
			const float x = logits[i] / temp;
			if (fkey(x) < K) continue;
			c += exp(x - M);
			if (target < c) { pick = i; break; }
			pick = i;
		}
		codes[p.w] = pick;
		chosen = pick;
	}
	// target >= total (rounding): the last kept index.
	if (tid == ST - 1 && !(target < total)) {
		uint last = 0;
		for (uint i = 0; i < V; i++) if (fkey(logits[i] / temp) >= K) last = i;
		codes[p.w] = last;
		chosen = last;
	}
	}
	if (p2.y == 0) return;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	const uint code = chosen;
	for (uint c = tid; c < p2.x; c += ST) dst[c] = float(table[code * p2.x + c]);
}

// dst[c] = float(table[codes[p.y] · p.x + c]) for c < p.x (bf16 table).
kernel void embed_row(device const bfloat* table [[buffer(0)]], device const uint* codes [[buffer(1)]],
                      device float* dst [[buffer(2)]], constant uint4& p [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
	if (i >= p.x) return;
	dst[i] = float(table[codes[p.y] * p.x + i]);
}
`

var decodePSO struct {
	once                   sync.Once
	gemvBF16, gemvQ8, attn *Pipeline
	gemvQ8S, gemvBF16S     *Pipeline
	gemvFBF16, gemvFQ8     *Pipeline
	gluFBF16, gluFQ8       *Pipeline
	ropeQK, attnRope       *Pipeline
	sample, embed          *Pipeline
	err                    error
}

// PrepareDecode compiles the decode kernels.
func (d *Device) PrepareDecode() error {
	decodePSO.once.Do(func() {
		for _, k := range []struct {
			name string
			dst  **Pipeline
		}{{"gemv_bf16", &decodePSO.gemvBF16}, {"gemv_q8", &decodePSO.gemvQ8}, {"attn_decode", &decodePSO.attn},
			{"gemv_q8s", &decodePSO.gemvQ8S}, {"gemv_bf16s", &decodePSO.gemvBF16S},
			{"gemv_f_bf16", &decodePSO.gemvFBF16}, {"gemv_f_q8", &decodePSO.gemvFQ8},
			{"glu_f_bf16", &decodePSO.gluFBF16}, {"glu_f_q8", &decodePSO.gluFQ8},
			{"rmsnorm_rope_qk", &decodePSO.ropeQK}, {"attn_decode_rope", &decodePSO.attnRope},
			{"sample_logits", &decodePSO.sample}, {"embed_row", &decodePSO.embed}} {
			if *k.dst, decodePSO.err = d.Compile(decodeSrc, k.name); decodePSO.err != nil {
				return
			}
		}
	})
	return decodePSO.err
}

// Gemv describes y[r, n] = Σ_k X[r, k] · W[n, k] (+ Bias[n]) for r < Rows
// ≤ 4, W [N, K] bf16 (BF16) or int8 with per-64 scales S. Accumulate adds
// into Y. Row strides in elements (0 = packed). K a multiple of 16.
type Gemv struct {
	N, K, Rows    int
	X, W, Y, Bias Region
	S             Region // int8 W: scales [N, K/64]
	LDX, LDY, LDW int
	BF16          bool
	Accumulate    bool
	HasBias       bool
}

// gemvStaged selects the threadgroup-staged kernels for one-row GEMVs
// (K ≤ 8192); false keeps the per-lane-read kernels (benchmarks A/B them).
var gemvStaged = true

// Gemv encodes g. Call Device.PrepareDecode first.
func (e *Encoder) Gemv(g Gemv) {
	if e.err != nil {
		return
	}
	if decodePSO.gemvBF16 == nil {
		e.err = fmt.Errorf("metal: Gemv before Device.PrepareDecode")
		return
	}
	if g.Rows < 1 || g.Rows > 4 || g.K%16 != 0 || g.N <= 0 || (!g.BF16 && g.K%64 != 0) {
		e.err = fmt.Errorf("metal: gemv %d rows × N %d × K %d", g.Rows, g.N, g.K)
		return
	}
	ldx, ldy, ldw := or(g.LDX, g.K), or(g.LDY, g.N), or(g.LDW, g.K)
	flags := 0
	if g.Accumulate {
		flags |= 1
	}
	bias := g.Bias
	if g.HasBias {
		flags |= 2
	} else {
		bias = g.Y // unread
	}
	grid, group := [3]int{(g.N + 7) / 8 * 256, 1, 1}, [3]int{256, 1, 1}
	if g.Rows == 1 && g.K <= 8192 && gemvStaged {
		sgrid, sgroup := [3]int{(g.N + 31) / 32 * 1024, 1, 1}, [3]int{1024, 1, 1}
		if g.BF16 {
			e.Dispatch(decodePSO.gemvBF16S, sgrid, sgroup, g.X, g.W, g.Y, bias, u32s(g.N, g.K, ldw, 1), u32s(ldx, ldy, flags, 0))
		} else {
			e.Dispatch(decodePSO.gemvQ8S, sgrid, sgroup, g.X, g.W, g.Y, bias, u32s(g.N, g.K, ldw, 1), u32s(ldx, ldy, flags, 0), g.S)
		}
		return
	}
	if g.BF16 {
		e.Dispatch(decodePSO.gemvBF16, grid, group, g.X, g.W, g.Y, bias, u32s(g.N, g.K, ldw, g.Rows), u32s(ldx, ldy, flags, 0))
		return
	}
	e.Dispatch(decodePSO.gemvQ8, grid, group, g.X, g.W, g.Y, bias, u32s(g.N, g.K, ldw, g.Rows), u32s(ldx, ldy, flags, 0), g.S)
}

// AttnDecode encodes causal attention of t queries Q [t, heads·dh] (query
// i at position pos0+i) over the caches K, V [pos0+t, kvHeads·dh] into
// out [t, heads·dh]. dh ≤ 128, a multiple of 32; heads a multiple of
// kvHeads.
func (e *Encoder) AttnDecode(q, k, v, out Region, t, heads, kvHeads, dh, pos0, ldq, ldo int, scale float32) {
	if e.err != nil {
		return
	}
	if dh > 128 || dh%32 != 0 || kvHeads <= 0 || heads%kvHeads != 0 {
		e.err = fmt.Errorf("metal: attn_decode heads %d/%d dh %d", heads, kvHeads, dh)
		return
	}
	if e.ready(decodePSO.attn) {
		e.Dispatch(decodePSO.attn, [3]int{heads * 256, t, 1}, [3]int{256, 1, 1}, q, k, v, out,
			u32s(heads, kvHeads, dh, pos0), u32s(or(ldq, heads*dh), or(ldo, heads*dh), 0, 0), f32c(scale))
	}
}

// AttnDecodeRoPE is AttnDecode for t ≤ 4 new tokens with q [t, heads·dh]
// and the new keys kNew [t, kvHeads·dh] raw: both are RMS-normalised (qw,
// kw; eps) and rotated (mode; cos/sin tables [pos, dh/2] from position 0)
// in the kernel, and the finished new keys are written to k's cache rows
// pos0... The V rows must already be in the cache.
func (e *Encoder) AttnDecodeRoPE(q, k, v, out, kNew, qw, kw, cos, sin Region, t, heads, kvHeads, dh, pos0, mode int, scale, eps float32) {
	if e.err != nil {
		return
	}
	if dh > 128 || dh%64 != 0 || kvHeads <= 0 || heads%kvHeads != 0 || t < 1 || t > 4 || heads/kvHeads < 1 {
		e.err = fmt.Errorf("metal: attn_decode_rope heads %d/%d dh %d t %d", heads, kvHeads, dh, t)
		return
	}
	if e.ready(decodePSO.attnRope) {
		e.Dispatch(decodePSO.attnRope, [3]int{heads * 256, t, 1}, [3]int{256, 1, 1}, q, k, v, out, kNew, qw, kw, cos, sin,
			u32s(heads, kvHeads, dh, pos0), u32s(t, mode, 0, 0), f32c(scale, eps))
	}
}

// SampleLogits encodes drawing one token from logits [v] into
// codes[outIndex] (uint32): argmax when greedy (first maximum), else
// softmax over temperature-scaled logits restricted to the top k (k <= 0:
// all; ties at the k-th value kept) sampled with the uniform u in [0, 1)
// in index order. v ≤ 1024·64.
func (e *Encoder) SampleLogits(logits, codes Region, v, topK int, greedy bool, temperature, u float32, outIndex int) {
	if e.ready(decodePSO.sample) {
		e.Dispatch(decodePSO.sample, [3]int{1024, 1, 1}, [3]int{1024, 1, 1}, logits, codes,
			u32s(v, topK, b2i(greedy), outIndex), f32c(temperature, u), codes, codes, u32s(0, 0, 0, 0))
	}
}

// SampleEmbed is SampleLogits that also writes the chosen code's bf16
// embedding row (table [*, d]) to dst, as EmbedRow would.
func (e *Encoder) SampleEmbed(logits, codes Region, v, topK int, greedy bool, temperature, u float32, outIndex int, table, dst Region, d int) {
	if e.ready(decodePSO.sample) {
		e.Dispatch(decodePSO.sample, [3]int{1024, 1, 1}, [3]int{1024, 1, 1}, logits, codes,
			u32s(v, topK, b2i(greedy), outIndex), f32c(temperature, u), table, dst, u32s(d, 1, 0, 0))
	}
}

// EmbedRow encodes dst[0:d] = widen(table[codes[index]]) for a bf16 table
// with rows of d.
func (e *Encoder) EmbedRow(table, codes, dst Region, d, index int) {
	if e.ready(decodePSO.embed) {
		e.Dispatch(decodePSO.embed, [3]int{d, 1, 1}, [3]int{256, 1, 1}, table, codes, dst, u32s(d, index, 0, 0))
	}
}

// DecodeWeight is one [N, K] weight for the fused decode GEMVs: bf16, or
// int8 with per-64 scales.
type DecodeWeight struct {
	W, S Region
	BF16 bool
	LDW  int // row stride in elements (0 = K)
}

// FusedGemv encodes one-row y = W·x' over [n, k], x' = x, or with Norm
// RMSNorm(x)·NormW (eps). Outputs split: n < N1 → Y0, n < N2 →
// Y1[n-N1], else Y2[n-N2] (N1 = N2 = n: all Y0). Accumulate adds.
// k ≤ 8160, a multiple of 16 (64 for int8).
type FusedGemv struct {
	N, K       int
	X          Region
	W          DecodeWeight
	Y0, Y1, Y2 Region
	N1, N2     int
	Norm       bool
	NormW      Region
	Eps        float32
	Accumulate bool
}

func (e *Encoder) fusedOK(k int, w DecodeWeight) bool {
	if e.err != nil {
		return false
	}
	if decodePSO.gemvFBF16 == nil {
		e.err = fmt.Errorf("metal: fused GEMV before Device.PrepareDecode")
		return false
	}
	if k > 8160 || k%16 != 0 || (!w.BF16 && k%64 != 0) {
		e.err = fmt.Errorf("metal: fused GEMV K %d", k)
		return false
	}
	return true
}

func fusedFlags(acc, norm bool) int {
	f := 0
	if acc {
		f |= 1
	}
	if norm {
		f |= 4
	}
	return f
}

// Gemv1 encodes g.
func (e *Encoder) Gemv1(g FusedGemv) {
	if !e.fusedOK(g.K, g.W) {
		return
	}
	n1, n2 := g.N1, g.N2
	if n1 == 0 && n2 == 0 {
		n1, n2 = g.N, g.N
	}
	y1, y2, nw := g.Y1, g.Y2, g.NormW
	if y1.B == nil {
		y1 = g.Y0
	}
	if y2.B == nil {
		y2 = g.Y0
	}
	if nw.B == nil {
		nw = g.X
	}
	grid, group := [3]int{(g.N + 31) / 32 * 1024, 1, 1}, [3]int{1024, 1, 1}
	p, q, f := u32s(g.N, g.K, n1, n2), u32s(fusedFlags(g.Accumulate, g.Norm), or(g.W.LDW, g.K), 0, 0), f32c(g.Eps)
	if g.W.BF16 {
		e.Dispatch(decodePSO.gemvFBF16, grid, group, g.X, g.W.W, g.Y0, y1, y2, nw, p, q, f)
		return
	}
	e.Dispatch(decodePSO.gemvFQ8, grid, group, g.X, g.W.W, g.Y0, y1, y2, nw, p, q, f, g.W.S)
}

// GLU1 encodes one-row y[n] = silu(Wg[n]·x')·(Wu[n]·x') over [n, k]
// (x' as FusedGemv); Wg and Wu share a format.
func (e *Encoder) GLU1(n, k int, x Region, wg, wu DecodeWeight, y Region, norm bool, normW Region, eps float32) {
	if !e.fusedOK(k, wg) {
		return
	}
	if normW.B == nil {
		normW = x
	}
	grid, group := [3]int{(n + 31) / 32 * 1024, 1, 1}, [3]int{1024, 1, 1}
	p, q, f := u32s(n, k, 0, 0), u32s(fusedFlags(false, norm), or(wg.LDW, k), 0, 0), f32c(eps)
	if wg.BF16 {
		e.Dispatch(decodePSO.gluFBF16, grid, group, x, wg.W, wu.W, y, normW, p, q, f)
		return
	}
	e.Dispatch(decodePSO.gluFQ8, grid, group, x, wg.W, wu.W, y, normW, p, q, f, wg.S, wu.S)
}

// RMSNormRoPEQK encodes, for one token, RMSNorm (weights qw / kw) then
// RoPE (mode RopePairs or RopeHalf, cos/sin the position's dh/2 row) of h
// query heads at q and kv key heads at k, in place.
func (e *Encoder) RMSNormRoPEQK(q, k, qw, kw, cos, sin Region, h, kv, dh, mode int, eps float32) {
	if dh > 512 || dh%2 != 0 {
		e.err = fmt.Errorf("metal: RMSNormRoPEQK head dim %d", dh)
		return
	}
	if e.ready(decodePSO.ropeQK) {
		e.Dispatch(decodePSO.ropeQK, [3]int{(h + kv) * dh, 1, 1}, [3]int{dh, 1, 1}, q, k, qw, kw, cos, sin,
			u32s(h, kv, dh, mode), f32c(eps))
	}
}
