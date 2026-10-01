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
// into y, bit 1 add bias.
template <typename LOAD>
void gemv(device const float* x, device float* y, device const float* bias, constant uint4& p,
          constant uint4& q, uint tg, uint sg, uint lane, LOAD load) {
	const uint n = tg * SG + sg;
	if (n >= p.x) return;
	const uint K = p.y, R = p.w;
	float acc[4] = {0, 0, 0, 0};
	for (uint k = lane * 8; k < K; k += 256) {
		float4 w0, w1;
		load(n, k, w0, w1);
		for (uint r = 0; r < R; r++) {
			device const float* xr = x + r * q.x + k;
			acc[r] += dot(w0, *(device const float4*)xr) + dot(w1, *(device const float4*)(xr + 4));
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
	gemv(x, y, bias, p, q, tg, sg, lane, [&](uint n, uint k, thread float4& w0, thread float4& w1) {
		device const bfloat4* w = (device const bfloat4*)(W + n * p.z + k);
		w0 = float4(w[0]);
		w1 = float4(w[1]);
	});
}

// Int8 weights W [N, K] (row stride ldw = p.z) with scales S [N, K/64].
kernel void gemv_q8(device const float* x [[buffer(0)]], device const char* W [[buffer(1)]],
                    device float* y [[buffer(2)]], device const float* bias [[buffer(3)]],
                    constant uint4& p [[buffer(4)]], constant uint4& q [[buffer(5)]],
                    device const float* S [[buffer(6)]],
                    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
	gemv(x, y, bias, p, q, tg, sg, lane, [&](uint n, uint k, thread float4& w0, thread float4& w1) {
		device const char4* w = (device const char4*)(W + n * p.z + k);
		const float s = S[n * (p.y / 64) + k / 64];
		w0 = float4(w[0]) * s;
		w1 = float4(w[1]) * s;
	});
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
kernel void sample_logits(device const float* logits [[buffer(0)]], device uint* codes [[buffer(1)]],
                          constant uint4& p [[buffer(2)]], constant float4& f [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
	threadgroup float sc[ST / 32];
	threadgroup uint si[ST / 32];
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
		}
		return;
	}
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
	}
	// target >= total (rounding): the last kept index.
	if (tid == ST - 1 && !(target < total)) {
		uint last = 0;
		for (uint i = 0; i < V; i++) if (fkey(logits[i] / temp) >= K) last = i;
		codes[p.w] = last;
	}
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
// into Y. Row strides in elements (0 = packed). K a multiple of 8.
type Gemv struct {
	N, K, Rows    int
	X, W, Y, Bias Region
	S             Region // int8 W: scales [N, K/64]
	LDX, LDY, LDW int
	BF16          bool
	Accumulate    bool
	HasBias       bool
}

// Gemv encodes g. Call Device.PrepareDecode first.
func (e *Encoder) Gemv(g Gemv) {
	if e.err != nil {
		return
	}
	if decodePSO.gemvBF16 == nil {
		e.err = fmt.Errorf("metal: Gemv before Device.PrepareDecode")
		return
	}
	if g.Rows < 1 || g.Rows > 4 || g.K%8 != 0 || g.N <= 0 || (!g.BF16 && g.K%64 != 0) {
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

// SampleLogits encodes drawing one token from logits [v] into
// codes[outIndex] (uint32): argmax when greedy (first maximum), else
// softmax over temperature-scaled logits restricted to the top k (k <= 0:
// all; ties at the k-th value kept) sampled with the uniform u in [0, 1)
// in index order. v ≤ 1024·64.
func (e *Encoder) SampleLogits(logits, codes Region, v, topK int, greedy bool, temperature, u float32, outIndex int) {
	if e.ready(decodePSO.sample) {
		e.Dispatch(decodePSO.sample, [3]int{1024, 1, 1}, [3]int{1024, 1, 1}, logits, codes,
			u32s(v, topK, b2i(greedy), outIndex), f32c(temperature, u))
	}
}

// EmbedRow encodes dst[0:d] = widen(table[codes[index]]) for a bf16 table
// with rows of d.
func (e *Encoder) EmbedRow(table, codes, dst Region, d, index int) {
	if e.ready(decodePSO.embed) {
		e.Dispatch(decodePSO.embed, [3]int{d, 1, 1}, [3]int{256, 1, 1}, table, codes, dst, u32s(d, index, 0, 0))
	}
}
