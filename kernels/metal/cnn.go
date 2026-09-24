//go:build darwin && arm64

package metal

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
)

// actFnSrc is the activation shared by the elementwise and convolution
// kernels: act(v) for the Act* codes (alpha/beta parameterise hardsigmoid,
// clip, leakyrelu).
const actFnSrc = `
static float erf_act(float x) {
	const float z = fabs(x), t = 1.0f / (1.0f + 0.5f * z);
	const float r = t * exp(-z * z - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f +
		t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f +
		t * (-0.82215223f + t * 0.17087277f)))))))));
	return x >= 0 ? 1.0f - r : r - 1.0f;
}
static float act_fn(float v, uint act, float alpha, float beta) {
	switch (act) {
	case 1: return max(v, 0.0f);                                              // relu
	case 2: return v * clamp(v / 6.0f + 0.5f, 0.0f, 1.0f);                    // hardswish
	case 3: return clamp(alpha * v + beta, 0.0f, 1.0f);                       // hardsigmoid
	case 4: return 1.0f / (1.0f + exp(-v));                                   // sigmoid
	case 5: return v / (1.0f + exp(-v));                                      // silu
	case 6: return min(max(v, alpha), beta);                                  // clip
	case 7: return v >= 0 ? v : alpha * v;                                    // leakyrelu
	case 8: return 0.5f * v * (1.0f + erf_act(v * 0.70710678f));              // gelu (erf)
	case 9: return 0.5f * v * (1.0f + precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v))); // gelu (tanh)
	default: return v;
	}
}
`

// NCHW convolution, pooling and activation kernels for executing ONNX
// graphs on the GPU (f32).
const cnnSrc = `
#include <metal_stdlib>
using namespace metal;
` + actFnSrc + `

// Conv geometry: a = (C, H, W, OW), b = (KH, KW, SH, SW), c = (DH, DW, PT, PL).

// cols[k, j] = x[c, oy·SH - PT + ky·DH, ox·SW - PL + kx·DW] (0 outside) for
// k = (c·KH + ky)·KW + kx and output pixel p = p0 + j (j < pc); one image's
// channels x [C, H, W]. d = (p0, pc, K, 0).
template <typename T>
void im2col(device const float* x, device T* cols, uint4 a, uint4 b, uint4 c, uint4 d, uint2 i) {
	const uint j = i.x, k = i.y;
	if (j >= d.y || k >= d.z) return;
	const uint p = d.x + j, oy = p / a.w, ox = p % a.w;
	const uint kx = k % b.y, ky = (k / b.y) % b.x, ch = k / (b.x * b.y);
	const int iy = int(oy * b.z + ky * c.x) - int(c.z), ix = int(ox * b.w + kx * c.y) - int(c.w);
	float v = 0;
	if (iy >= 0 && iy < int(a.y) && ix >= 0 && ix < int(a.z)) v = x[(ch * a.y + uint(iy)) * a.z + uint(ix)];
	cols[k * d.y + j] = T(v);
}
kernel void im2col_nchw(device const float* x [[buffer(0)]], device float* cols [[buffer(1)]],
                        constant uint4& a [[buffer(2)]], constant uint4& b [[buffer(3)]],
                        constant uint4& c [[buffer(4)]], constant uint4& d [[buffer(5)]],
                        uint2 i [[thread_position_in_grid]]) { im2col(x, cols, a, b, c, d, i); }
// The same, writing bf16 columns (the bf16 GEMM's B operand).
kernel void im2col_nchw_bf16(device const float* x [[buffer(0)]], device bfloat* cols [[buffer(1)]],
                             constant uint4& a [[buffer(2)]], constant uint4& b [[buffer(3)]],
                             constant uint4& c [[buffer(4)]], constant uint4& d [[buffer(5)]],
                             uint2 i [[thread_position_in_grid]]) { im2col(x, cols, a, b, c, d, i); }

// Direct convolution (grouped / depthwise): one thread per output element of
// out [N, M, OH, OW]; w [M, Cg, KH, KW]. e = (Cg, Mg, M, OH), f = (hasBias,
// N, 0, 0).
kernel void conv_direct(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                        device const float* bias [[buffer(2)]], device float* o [[buffer(3)]],
                        constant uint4& a [[buffer(4)]], constant uint4& b [[buffer(5)]],
                        constant uint4& c [[buffer(6)]], constant uint4& e [[buffer(7)]],
                        constant uint4& f [[buffer(8)]], uint3 i [[thread_position_in_grid]]) {
	const uint OW = a.w, OH = e.w, p = i.x, m = i.y, n = i.z;
	if (p >= OH * OW || m >= e.z || n >= f.y) return;
	const uint oy = p / OW, ox = p % OW, g = m / e.y, Cg = e.x;
	const int y0 = int(oy * b.z) - int(c.z), x0 = int(ox * b.w) - int(c.w);
	device const float* wm = w + m * Cg * b.x * b.y;
	device const float* xn = x + (n * a.x + g * Cg) * a.y * a.z;
	float acc = f.x != 0 ? bias[m] : 0.0f;
	for (uint ch = 0; ch < Cg; ch++) {
		device const float* xc = xn + ch * a.y * a.z;
		for (uint ky = 0; ky < b.x; ky++) {
			const int iy = y0 + int(ky * c.x);
			if (iy < 0 || iy >= int(a.y)) continue;
			for (uint kx = 0; kx < b.y; kx++) {
				const int ix = x0 + int(kx * c.y);
				if (ix < 0 || ix >= int(a.z)) continue;
				acc += xc[uint(iy) * a.z + uint(ix)] * wm[(ch * b.x + ky) * b.y + kx];
			}
		}
	}
	o[((n * e.z + m) * OH + oy) * OW + ox] = acc;
}

// Depthwise convolution (group == C == M), kernel width ≤ 7: each thread
// computes DWX horizontally adjacent outputs of one row, loading each input
// row segment once into registers and sharing it across them and across
// the kernel's taps. Grid (ceil(OW/DWX), OH, N·C); e = (hasBias, act, 0, OH),
// q = (alpha, beta, scale, shift): the fused epilogue.
#define DWX 4
kernel void conv_dw(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                    device const float* bias [[buffer(2)]], device float* o [[buffer(3)]],
                    constant uint4& a [[buffer(4)]], constant uint4& b [[buffer(5)]],
                    constant uint4& c [[buffer(6)]], constant uint4& e [[buffer(7)]],
                    constant float4& q [[buffer(8)]], uint3 i [[thread_position_in_grid]]) {
	const uint OW = a.w, OH = e.w, ox0 = i.x * DWX, oy = i.y, nc = i.z, ch = nc % a.x;
	if (ox0 >= OW || oy >= OH) return;
	const int H = int(a.y), W = int(a.z), KH = int(b.x), KW = int(b.y), sw = int(b.w), dw = int(c.y);
	const int y0 = int(oy * b.z) - int(c.z), x0 = int(ox0) * sw - int(c.w);
	device const float* xp = x + nc * a.y * a.z;
	device const float* wp = w + ch * uint(KH * KW);
	const float bv = e.x != 0 ? bias[ch] : 0.0f;
	float acc[DWX];
	for (int j = 0; j < DWX; j++) acc[j] = bv;
	const int span = (DWX - 1) * sw + (KW - 1) * dw + 1; // ≤ 3·sw + 6·dw + 1
	for (int ky = 0; ky < KH; ky++) {
		const int iy = y0 + ky * int(c.x);
		if (iy < 0 || iy >= H) continue;
		device const float* row = xp + iy * W;
		float seg[32];
		for (int t = 0; t < span && t < 32; t++) {
			const int ix = x0 + t;
			seg[t] = (ix >= 0 && ix < W) ? row[ix] : 0.0f;
		}
		for (int kx = 0; kx < KW; kx++) {
			const float wv = wp[ky * KW + kx];
			for (int j = 0; j < DWX; j++) acc[j] = fma(seg[j * sw + kx * dw], wv, acc[j]);
		}
	}
	device float* op = o + (nc * OH + oy) * OW + ox0;
	const uint n = min(uint(DWX), OW - ox0);
	for (uint j = 0; j < n; j++) op[j] = act_fn(acc[j], e.y, q.x, q.y) * q.z + q.w;
}

// Transposed convolution (gather form: each output sums the inputs whose
// taps land on it; no atomics). x [N, Cin, H, W], w [Cin, CoutG, KH, KW],
// out [N, Cout, OH, OW]; a = (Cin, H, W, OW), e = (CinG, CoutG, Cout, OH).
kernel void convt_direct(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                         device const float* bias [[buffer(2)]], device float* o [[buffer(3)]],
                         constant uint4& a [[buffer(4)]], constant uint4& b [[buffer(5)]],
                         constant uint4& c [[buffer(6)]], constant uint4& e [[buffer(7)]],
                         constant uint4& f [[buffer(8)]], uint3 i [[thread_position_in_grid]]) {
	const uint OW = a.w, OH = e.w, p = i.x, oc = i.y, n = i.z;
	if (p >= OH * OW || oc >= e.z || n >= f.y) return;
	const int oy = int(p / OW), ox = int(p % OW);
	const uint g = oc / e.y, ocg = oc % e.y, CinG = e.x;
	const int H = int(a.y), W = int(a.z), sh = int(b.z), sw = int(b.w);
	float acc = f.x != 0 ? bias[oc] : 0.0f;
	for (uint ky = 0; ky < b.x; ky++) {
		const int ty = oy + int(c.z) - int(ky * c.x);
		if (ty < 0 || ty % sh != 0 || ty / sh >= H) continue;
		const int iy = ty / sh;
		for (uint kx = 0; kx < b.y; kx++) {
			const int tx = ox + int(c.w) - int(kx * c.y);
			if (tx < 0 || tx % sw != 0 || tx / sw >= W) continue;
			const int ix = tx / sw;
			for (uint ic = 0; ic < CinG; ic++) {
				const uint ci = g * CinG + ic;
				acc += x[((n * a.x + ci) * uint(H) + uint(iy)) * uint(W) + uint(ix)] *
				       w[((ci * e.y + ocg) * b.x + ky) * b.y + kx];
			}
		}
	}
	o[((n * e.z + oc) * OH + uint(oy)) * OW + uint(ox)] = acc;
}

// Separable resize of planes x [NC, H, W] → o [NC, OH, OW] from tap tables:
// ti = [y0 OH | y1 OH | x0 OW | x1 OW] (int), tw = [wy OH | wx OW] (upper
// tap weights; 0 for nearest). a = (H, W, OH, OW).
kernel void resize_taps(device const float* x [[buffer(0)]], device float* o [[buffer(1)]],
                        device const int* ti [[buffer(2)]], device const float* tw [[buffer(3)]],
                        constant uint4& a [[buffer(4)]], uint2 i [[thread_position_in_grid]]) {
	const uint OH = a.z, OW = a.w;
	if (i.x >= OH * OW) return;
	const uint r = i.x / OW, q = i.x % OW;
	device const float* xp = x + i.y * a.x * a.y;
	const int y0 = ti[r], y1 = ti[OH + r], x0 = ti[2 * OH + q], x1 = ti[2 * OH + OW + q];
	const float wy = tw[r], wx = tw[OH + q];
	const float v00 = xp[y0 * a.y + x0];
	float top = v00, bot;
	if (wx != 0) top = v00 * (1 - wx) + xp[y0 * a.y + x1] * wx;
	float v = top;
	if (wy != 0) {
		bot = xp[y1 * a.y + x0];
		if (wx != 0) bot = bot * (1 - wx) + xp[y1 * a.y + x1] * wx;
		v = top * (1 - wy) + bot * wy;
	}
	o[i.y * OH * OW + i.x] = v;
}

// 2-D max / average pooling over planes x [NC, H, W] → o [NC, OH, OW].
// a = (H, W, OH, OW), b = (KH, KW, SH, SW), c = (PT, PL, PB, PR),
// d = (max, countIncludePad, 0, 0).
kernel void pool2d(device const float* x [[buffer(0)]], device float* o [[buffer(1)]],
                   constant uint4& a [[buffer(2)]], constant uint4& b [[buffer(3)]],
                   constant uint4& c [[buffer(4)]], constant uint4& d [[buffer(5)]],
                   uint2 i [[thread_position_in_grid]]) {
	const uint p = i.x, nc = i.y;
	if (p >= a.z * a.w) return;
	const int H = int(a.x), W = int(a.y);
	const int y0 = int((p / a.w) * b.z) - int(c.x), x0 = int((p % a.w) * b.w) - int(c.y);
	device const float* xp = x + nc * a.x * a.y;
	float m = -INFINITY, s = 0;
	int cnt = 0;
	for (int ky = max(y0, 0); ky < min(y0 + int(b.x), H); ky++) {
		for (int kx = max(x0, 0); kx < min(x0 + int(b.y), W); kx++) {
			const float v = xp[ky * W + kx];
			m = max(m, v);
			s += v;
			cnt++;
		}
	}
	float r;
	if (d.x != 0) {
		r = m;
	} else {
		if (d.y != 0) {
			const int hA = max(y0, -int(c.x)), hB = min(y0 + int(b.x), H + int(c.z));
			const int wA = max(x0, -int(c.y)), wB = min(x0 + int(b.y), W + int(c.w));
			cnt = (hB - hA) * (wB - wA);
		}
		r = cnt > 0 ? s / float(cnt) : 0.0f;
	}
	o[nc * a.z * a.w + p] = r;
}

// o[i] = act(x[i])·scale + shift; p = (n, act), q = (alpha, beta, scale, shift).
kernel void act_ew(device const float* x [[buffer(0)]], device float* o [[buffer(1)]],
                   constant uint4& p [[buffer(2)]], constant float4& q [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
	if (i >= p.x) return;
	o[i] = act_fn(x[i], p.y, q.x, q.y) * q.z + q.w;
}

// In place: x[i] = act(x[i] + bias[(i / inner) % M])·scale + shift — a
// GEMM's or conv's bias and epilogue in one pass. p = (n, inner, M, act),
// r = (hasBias, 0, 0, 0), q = (alpha, beta, scale, shift).
kernel void bias_act(device float* x [[buffer(0)]], device const float* bias [[buffer(1)]],
                     constant uint4& p [[buffer(2)]], constant uint4& r [[buffer(3)]],
                     constant float4& q [[buffer(4)]], uint i [[thread_position_in_grid]]) {
	if (i >= p.x) return;
	float v = x[i];
	if (r.x != 0) v += bias[(i / p.y) % p.z];
	x[i] = act_fn(v, p.w, q.x, q.y) * q.z + q.w;
}
`

// cnnBlockSrc holds the register-blocked kernels, compiled per block width
// CB (output channels per thread): each thread reads an input tap once and
// accumulates CB outputs.
const cnnBlockSrc = `
#include <metal_stdlib>
using namespace metal;
` + actFnSrc + `

// Direct convolution, group 1: one thread per (pixel, block of CB output
// channels) — no im2col scratch, which dominates thin convs at high
// resolution. Grid (P, ceil(M/CB), N); e = (C, M, OH, hasBias).
kernel void conv_direct_cb(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                           device const float* bias [[buffer(2)]], device float* o [[buffer(3)]],
                           constant uint4& a [[buffer(4)]], constant uint4& b [[buffer(5)]],
                           constant uint4& c [[buffer(6)]], constant uint4& e [[buffer(7)]],
                           constant uint4& ea [[buffer(8)]], constant float4& eq [[buffer(9)]],
                           uint3 i [[thread_position_in_grid]]) {
	const uint OW = a.w, OH = e.z, P = OH * OW, p = i.x, m0 = i.y * CB, n = i.z, M = e.y, C = e.x;
	if (p >= P || m0 >= M) return;
	const uint KK = b.x * b.y, K = C * KK, plane = a.y * a.z;
	const int y0 = int((p / OW) * b.z) - int(c.z), x0 = int((p % OW) * b.w) - int(c.w);
	const uint mc = min(uint(CB), M - m0);
	float acc[CB];
	for (uint j = 0; j < CB; j++) acc[j] = (e.w != 0 && j < mc) ? bias[m0 + j] : 0.0f;
	device const float* xn = x + n * C * plane;
	for (uint ky = 0; ky < b.x; ky++) {
		const int iy = y0 + int(ky * c.x);
		if (iy < 0 || iy >= int(a.y)) continue;
		for (uint kx = 0; kx < b.y; kx++) {
			const int ix = x0 + int(kx * c.y);
			if (ix < 0 || ix >= int(a.z)) continue;
			device const float* xp = xn + uint(iy) * a.z + uint(ix);
			device const float* wp = w + m0 * K + ky * b.y + kx;
			for (uint ch = 0; ch < C; ch++) {
				const float v = xp[ch * plane];
				if (mc == CB) {
					for (uint j = 0; j < CB; j++) acc[j] = fma(v, wp[j * K + ch * KK], acc[j]);
				} else {
					for (uint j = 0; j < mc; j++) acc[j] = fma(v, wp[j * K + ch * KK], acc[j]);
				}
			}
		}
	}
	device float* op = o + (n * M + m0) * P + p;
	for (uint j = 0; j < mc; j++) op[j * P] = act_fn(acc[j], ea.x, eq.x, eq.y) * eq.z + eq.w;
}

// Transposed convolution, gather form, blocked over output channels:
// thread (pixel, block of CB channels of one group). w [Cin, CoutG, KH, KW];
// a = (Cin, H, W, OW), e = (CinG, CoutG, Cout, OH), f = (hasBias, N,
// blocks per group, 0).
kernel void convt_cb(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                     device const float* bias [[buffer(2)]], device float* o [[buffer(3)]],
                     constant uint4& a [[buffer(4)]], constant uint4& b [[buffer(5)]],
                     constant uint4& c [[buffer(6)]], constant uint4& e [[buffer(7)]],
                     constant uint4& f [[buffer(8)]], constant float4& eq [[buffer(9)]],
                     uint3 i [[thread_position_in_grid]]) {
	const uint OW = a.w, OH = e.w, P = OH * OW, p = i.x, n = i.z, CoutG = e.y, CinG = e.x;
	const uint g = i.y / f.z, ocg0 = (i.y % f.z) * CB;
	if (p >= P || n >= f.y || ocg0 >= CoutG) return;
	const uint mc = min(uint(CB), CoutG - ocg0), oc0 = g * CoutG + ocg0, KK = b.x * b.y;
	const int oy = int(p / OW), ox = int(p % OW), H = int(a.y), W = int(a.z), sh = int(b.z), sw = int(b.w);
	float acc[CB];
	for (uint j = 0; j < CB; j++) acc[j] = (f.x != 0 && j < mc) ? bias[oc0 + j] : 0.0f;
	for (uint ky = 0; ky < b.x; ky++) {
		const int ty = oy + int(c.z) - int(ky * c.x);
		if (ty < 0 || ty % sh != 0 || ty / sh >= H) continue;
		const int iy = ty / sh;
		for (uint kx = 0; kx < b.y; kx++) {
			const int tx = ox + int(c.w) - int(kx * c.y);
			if (tx < 0 || tx % sw != 0 || tx / sw >= W) continue;
			const int ix = tx / sw;
			device const float* xp = x + ((n * a.x + g * CinG) * uint(H) + uint(iy)) * uint(W) + uint(ix);
			device const float* wp = w + ((g * CinG) * CoutG + ocg0) * KK + ky * b.y + kx;
			for (uint ic = 0; ic < CinG; ic++) {
				const float v = xp[ic * uint(H) * uint(W)];
				device const float* wr = wp + ic * CoutG * KK;
				if (mc == CB) {
					for (uint j = 0; j < CB; j++) acc[j] = fma(v, wr[j * KK], acc[j]);
				} else {
					for (uint j = 0; j < mc; j++) acc[j] = fma(v, wr[j * KK], acc[j]);
				}
			}
		}
	}
	device float* op = o + (n * e.z + oc0) * P + p;
	for (uint j = 0; j < mc; j++) op[j * P] = act_fn(acc[j], f.w, eq.x, eq.y) * eq.z + eq.w;
}
`

// blockWidths are the compiled CB variants of cnnBlockSrc. Only 8: on M5
// Pro, 16 and 32 measured 2-25x slower than 8 (accumulators spill), even
// with 24 output channels (BenchmarkConvThin, BenchmarkConvTranspose).
var blockWidths = [...]int{8}

// Activation codes for Act.
const (
	ActNone = iota
	ActRelu
	ActHardSwish
	ActHardSigmoid
	ActSigmoid
	ActSiLU
	ActClip
	ActLeakyRelu
	ActGeluErf
	ActGeluTanh
)

var cnnPSO struct {
	once                        sync.Once
	im2col, direct, pool2d, act *Pipeline
	convT, resize, dw, im2colBF *Pipeline
	biasAct                     *Pipeline
	directCB, convTCB           [len(blockWidths)]*Pipeline
	err                         error
}

// PrepareCNN compiles the convolution, pooling and activation kernels.
func (d *Device) PrepareCNN() error {
	cnnPSO.once.Do(func() {
		for _, k := range []struct {
			name string
			dst  **Pipeline
		}{{"im2col_nchw", &cnnPSO.im2col}, {"conv_direct", &cnnPSO.direct}, {"pool2d", &cnnPSO.pool2d}, {"act_ew", &cnnPSO.act},
			{"convt_direct", &cnnPSO.convT}, {"resize_taps", &cnnPSO.resize}, {"conv_dw", &cnnPSO.dw}, {"im2col_nchw_bf16", &cnnPSO.im2colBF}, {"bias_act", &cnnPSO.biasAct}} {
			if *k.dst, cnnPSO.err = d.Compile(cnnSrc, k.name); cnnPSO.err != nil {
				return
			}
		}
		for i, cb := range blockWidths {
			src := fmt.Sprintf("#define CB %d\n", cb) + cnnBlockSrc
			if cnnPSO.directCB[i], cnnPSO.err = d.Compile(src, "conv_direct_cb"); cnnPSO.err != nil {
				return
			}
			if cnnPSO.convTCB[i], cnnPSO.err = d.Compile(src, "convt_cb"); cnnPSO.err != nil {
				return
			}
		}
	})
	return cnnPSO.err
}

// ConvGeom is a 2-D convolution's shape: input [N, C, H, W], weights
// [M, C/Group, KH, KW], output [N, M, OH, OW].
type ConvGeom struct {
	N, C, H, W, M, OH, OW int
	KH, KW, SH, SW        int
	DH, DW, PT, PL        int
	Group                 int
}

func (g ConvGeom) args() (Arg, Arg, Arg) {
	return u32s(g.C, g.H, g.W, g.OW), u32s(g.KH, g.KW, g.SH, g.SW), u32s(g.DH, g.DW, g.PT, g.PL)
}

// Im2ColNCHW writes cols [C·KH·KW, pc] for output pixels [p0, p0+pc) of
// one image x [C, H, W] (g.N and g.Group are ignored; g.C is the image's
// channel count).
func (e *Encoder) Im2ColNCHW(x, cols Region, g ConvGeom, p0, pc int) {
	e.im2col(cnnPSO.im2col, x, cols, g, p0, pc)
}

// Im2ColNCHWBF16 is Im2ColNCHW writing bf16 columns.
func (e *Encoder) Im2ColNCHWBF16(x, cols Region, g ConvGeom, p0, pc int) {
	e.im2col(cnnPSO.im2colBF, x, cols, g, p0, pc)
}

func (e *Encoder) im2col(p *Pipeline, x, cols Region, g ConvGeom, p0, pc int) {
	K := g.C * g.KH * g.KW
	if p0 < 0 || pc <= 0 || p0+pc > g.OH*g.OW {
		e.err = fmt.Errorf("metal: im2col pixels [%d, %d) of %d", p0, p0+pc, g.OH*g.OW)
		return
	}
	if e.ready(p) {
		a, b, c := g.args()
		e.Dispatch(p, [3]int{pc, K, 1}, [3]int{64, 4, 1}, x, cols, a, b, c, u32s(p0, pc, K, 0))
	}
}

// ConvDirect computes the whole convolution out = conv(x, w) (+ bias) one
// thread per output element — for grouped and depthwise convolutions,
// whose per-group GEMMs are too small. bias may be the zero Region.
func (e *Encoder) ConvDirect(x, w, bias, out Region, g ConvGeom) {
	if g.Group <= 0 || g.C%g.Group != 0 || g.M%g.Group != 0 {
		e.err = fmt.Errorf("metal: ConvDirect C=%d M=%d group=%d", g.C, g.M, g.Group)
		return
	}
	hasBias := 1
	if bias.B == nil {
		bias, hasBias = w, 0
	}
	if e.ready(cnnPSO.direct) {
		a, b, c := g.args()
		e.Dispatch(cnnPSO.direct, [3]int{g.OH * g.OW, g.M, g.N}, [3]int{64, 1, 1}, x, w, bias, out, a, b, c,
			u32s(g.C/g.Group, g.M/g.Group, g.M, g.OH), u32s(hasBias, g.N, 0, 0))
	}
}

// ConvDirectBlocked computes a group-1 convolution (+ bias) with each
// thread producing a block of output channels of one pixel: for thin convs
// (few output channels, large planes) where im2col's scratch traffic
// dominates. bias may be the zero Region.
func (e *Encoder) ConvDirectBlocked(x, w, bias, out Region, g ConvGeom, ep ConvEpilogue) {
	e.convDirectCB(blockFor(g.M), x, w, bias, out, g, ep)
}

// blockFor picks the block-width variant for n output channels.
func blockFor(n int) int {
	for i, cb := range blockWidths {
		if n <= cb {
			return i
		}
	}
	return len(blockWidths) - 1
}

func (e *Encoder) convDirectCB(v int, x, w, bias, out Region, g ConvGeom, ep ConvEpilogue) {
	if g.Group != 1 {
		e.err = fmt.Errorf("metal: ConvDirectBlocked group=%d", g.Group)
		return
	}
	hasBias := 1
	if bias.B == nil {
		bias, hasBias = w, 0
	}
	if e.ready(cnnPSO.directCB[v]) {
		a, b, c := g.args()
		cb := blockWidths[v]
		e.Dispatch(cnnPSO.directCB[v], [3]int{g.OH * g.OW, (g.M + cb - 1) / cb, g.N}, [3]int{64, 1, 1}, x, w, bias, out, a, b, c,
			u32s(g.C, g.M, g.OH, hasBias), u32s(ep.Act, 0, 0, 0), ep.params())
	}
}

// ConvTransposeBlocked is ConvTransposeDirect with each thread producing a
// block of a group's output channels (inputs read once per block).
func (e *Encoder) ConvTransposeBlocked(x, w, bias, out Region, g ConvGeom, ep ConvEpilogue) {
	e.convTCB(blockFor(g.M/max(g.Group, 1)), x, w, bias, out, g, ep)
}

func (e *Encoder) convTCB(v int, x, w, bias, out Region, g ConvGeom, ep ConvEpilogue) {
	if g.Group <= 0 || g.C%g.Group != 0 || g.M%g.Group != 0 {
		e.err = fmt.Errorf("metal: ConvTransposeBlocked C=%d M=%d group=%d", g.C, g.M, g.Group)
		return
	}
	hasBias := 1
	if bias.B == nil {
		bias, hasBias = w, 0
	}
	if e.ready(cnnPSO.convTCB[v]) {
		a, b, c := g.args()
		cb, coutG := blockWidths[v], g.M/g.Group
		blocks := (coutG + cb - 1) / cb
		e.Dispatch(cnnPSO.convTCB[v], [3]int{g.OH * g.OW, g.Group * blocks, g.N}, [3]int{64, 1, 1}, x, w, bias, out, a, b, c,
			u32s(g.C/g.Group, coutG, g.M, g.OH), u32s(hasBias, g.N, blocks, ep.Act), ep.params())
	}
}

// DepthwiseOK reports whether ConvDepthwise handles g: group == C == M,
// and the register window (3·SW + (KW−1)·DW + 1) fits 32.
func DepthwiseOK(g ConvGeom) bool {
	return g.Group == g.C && g.C == g.M && 3*g.SW+(g.KW-1)*g.DW+1 <= 32
}

// ConvDepthwise computes a depthwise convolution (+ bias, then the
// epilogue), four adjacent outputs per thread. bias may be the zero Region.
func (e *Encoder) ConvDepthwise(x, w, bias, out Region, g ConvGeom, ep ConvEpilogue) {
	if !DepthwiseOK(g) {
		e.err = fmt.Errorf("metal: ConvDepthwise unsupported geometry %+v", g)
		return
	}
	hasBias := 1
	if bias.B == nil {
		bias, hasBias = w, 0
	}
	if e.ready(cnnPSO.dw) {
		a, b, c := g.args()
		e.Dispatch(cnnPSO.dw, [3]int{(g.OW + 3) / 4, g.OH, g.N * g.C}, [3]int{32, 2, 1}, x, w, bias, out, a, b, c,
			u32s(hasBias, ep.Act, 0, g.OH), ep.params())
	}
}

// ConvTransposeDirect computes out = convtranspose(x, w) (+ bias) one
// thread per output. Geometry: g.C = Cin, g.M = Cout, g.H/g.W the input,
// g.OH/g.OW the output, g.PT/g.PL the leading pads; w is [Cin, Cout/Group,
// KH, KW]. bias may be the zero Region.
func (e *Encoder) ConvTransposeDirect(x, w, bias, out Region, g ConvGeom) {
	if g.Group <= 0 || g.C%g.Group != 0 || g.M%g.Group != 0 {
		e.err = fmt.Errorf("metal: ConvTransposeDirect C=%d M=%d group=%d", g.C, g.M, g.Group)
		return
	}
	hasBias := 1
	if bias.B == nil {
		bias, hasBias = w, 0
	}
	if e.ready(cnnPSO.convT) {
		a, b, c := g.args()
		e.Dispatch(cnnPSO.convT, [3]int{g.OH * g.OW, g.M, g.N}, [3]int{64, 1, 1}, x, w, bias, out, a, b, c,
			u32s(g.C/g.Group, g.M/g.Group, g.M, g.OH), u32s(hasBias, g.N, 0, 0))
	}
}

// ResizeTaps resamples planes x [planes, h, w] to out [planes, oh, ow]
// from tap tables (see the kernel): idx holds 2·oh + 2·ow int32 indices,
// wts oh + ow float weights.
func (e *Encoder) ResizeTaps(x, out, idx, wts Region, planes, h, w, oh, ow int) {
	if e.ready(cnnPSO.resize) {
		e.Dispatch(cnnPSO.resize, [3]int{oh * ow, planes, 1}, [3]int{64, 1, 1}, x, out, idx, wts, u32s(h, w, oh, ow))
	}
}

// Pool2D is max (or average) pooling of planes x [planes, H, W] into out
// [planes, OH, OW]; pads = (top, left, bottom, right). includePad selects
// ONNX count_include_pad (window clipped to the padded extent).
func (e *Encoder) Pool2D(x, out Region, planes, h, w, oh, ow, kh, kw, sh, sw int, pads [4]int, isMax, includePad bool) {
	if e.ready(cnnPSO.pool2d) {
		e.Dispatch(cnnPSO.pool2d, [3]int{oh * ow, planes, 1}, [3]int{64, 1, 1}, x, out,
			u32s(h, w, oh, ow), u32s(kh, kw, sh, sw), u32s(pads[0], pads[1], pads[2], pads[3]), u32s(b2i(isMax), b2i(includePad), 0, 0))
	}
}

// Act writes out[i] = act(x[i])·scale + shift for i < n (out may alias x);
// alpha and beta parameterise hardsigmoid, clip and leakyrelu.
func (e *Encoder) Act(act int, x, out Region, n int, alpha, beta, scale, shift float32) {
	if e.ready(cnnPSO.act) {
		q := make([]byte, 0, 16)
		for _, v := range [4]float32{alpha, beta, scale, shift} {
			q = binary.LittleEndian.AppendUint32(q, math.Float32bits(v))
		}
		e.Dispatch(cnnPSO.act, [3]int{n, 1, 1}, [3]int{256, 1, 1}, x, out, u32s(n, act), q)
	}
}

// BiasAct applies x = act(x + bias[(i / inner) % m])·scale + shift in
// place over n elements (bias may be the zero Region: epilogue only). For
// NCHW channels inner is the plane size; for a GEMM's per-column bias it
// is 1 with m = columns.
func (e *Encoder) BiasAct(x, bias Region, n, inner, m int, ep ConvEpilogue) {
	hasBias := 1
	if bias.B == nil {
		bias, hasBias = x, 0
	}
	if e.ready(cnnPSO.biasAct) {
		e.Dispatch(cnnPSO.biasAct, [3]int{n, 1, 1}, [3]int{256, 1, 1}, x, bias, u32s(n, max(inner, 1), max(m, 1), ep.Act),
			u32s(hasBias, 0, 0, 0), ep.params())
	}
}

// params is the epilogue's float constants (alpha, beta, scale, shift).
func (ep ConvEpilogue) params() []byte {
	scale := ep.Scale
	if scale == 0 {
		scale = 1
	}
	q := make([]byte, 0, 16)
	for _, v := range [4]float32{ep.Alpha, ep.Beta, scale, ep.Shift} {
		q = binary.LittleEndian.AppendUint32(q, math.Float32bits(v))
	}
	return q
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
