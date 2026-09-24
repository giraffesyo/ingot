//go:build darwin && arm64

package metal

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
)

// NCHW convolution, pooling and activation kernels for executing ONNX
// graphs on the GPU (f32).
const cnnSrc = `
#include <metal_stdlib>
using namespace metal;

// Conv geometry: a = (C, H, W, OW), b = (KH, KW, SH, SW), c = (DH, DW, PT, PL).

// cols[k, j] = x[c, oy·SH - PT + ky·DH, ox·SW - PL + kx·DW] (0 outside) for
// k = (c·KH + ky)·KW + kx and output pixel p = p0 + j (j < pc); one image's
// channels x [C, H, W]. d = (p0, pc, K, 0).
kernel void im2col_nchw(device const float* x [[buffer(0)]], device float* cols [[buffer(1)]],
                        constant uint4& a [[buffer(2)]], constant uint4& b [[buffer(3)]],
                        constant uint4& c [[buffer(4)]], constant uint4& d [[buffer(5)]],
                        uint2 i [[thread_position_in_grid]]) {
	const uint j = i.x, k = i.y;
	if (j >= d.y || k >= d.z) return;
	const uint p = d.x + j, oy = p / a.w, ox = p % a.w;
	const uint kx = k % b.y, ky = (k / b.y) % b.x, ch = k / (b.x * b.y);
	const int iy = int(oy * b.z + ky * c.x) - int(c.z), ix = int(ox * b.w + kx * c.y) - int(c.w);
	float v = 0;
	if (iy >= 0 && iy < int(a.y) && ix >= 0 && ix < int(a.z)) v = x[(ch * a.y + uint(iy)) * a.z + uint(ix)];
	cols[k * d.y + j] = v;
}

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
	float v = x[i];
	switch (p.y) {
	case 1: v = max(v, 0.0f); break;                                          // relu
	case 2: v = v * clamp(v / 6.0f + 0.5f, 0.0f, 1.0f); break;                // hardswish
	case 3: v = clamp(q.x * v + q.y, 0.0f, 1.0f); break;                      // hardsigmoid
	case 4: v = 1.0f / (1.0f + exp(-v)); break;                               // sigmoid
	case 5: v = v / (1.0f + exp(-v)); break;                                  // silu
	case 6: v = min(max(v, q.x), q.y); break;                                 // clip
	case 7: v = v >= 0 ? v : q.x * v; break;                                  // leakyrelu
	default: break;
	}
	o[i] = v * q.z + q.w;
}
`

// cnnBlockSrc holds the register-blocked kernels, compiled per block width
// CB (output channels per thread): each thread reads an input tap once and
// accumulates CB outputs.
const cnnBlockSrc = `
#include <metal_stdlib>
using namespace metal;

// Direct convolution, group 1: one thread per (pixel, block of CB output
// channels) — no im2col scratch, which dominates thin convs at high
// resolution. Grid (P, ceil(M/CB), N); e = (C, M, OH, hasBias).
kernel void conv_direct_cb(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                           device const float* bias [[buffer(2)]], device float* o [[buffer(3)]],
                           constant uint4& a [[buffer(4)]], constant uint4& b [[buffer(5)]],
                           constant uint4& c [[buffer(6)]], constant uint4& e [[buffer(7)]],
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
	for (uint j = 0; j < mc; j++) op[j * P] = acc[j];
}

// Transposed convolution, gather form, blocked over output channels:
// thread (pixel, block of CB channels of one group). w [Cin, CoutG, KH, KW];
// a = (Cin, H, W, OW), e = (CinG, CoutG, Cout, OH), f = (hasBias, N,
// blocks per group, 0).
kernel void convt_cb(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                     device const float* bias [[buffer(2)]], device float* o [[buffer(3)]],
                     constant uint4& a [[buffer(4)]], constant uint4& b [[buffer(5)]],
                     constant uint4& c [[buffer(6)]], constant uint4& e [[buffer(7)]],
                     constant uint4& f [[buffer(8)]], uint3 i [[thread_position_in_grid]]) {
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
	for (uint j = 0; j < mc; j++) op[j * P] = acc[j];
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
)

var cnnPSO struct {
	once                        sync.Once
	im2col, direct, pool2d, act *Pipeline
	convT, resize               *Pipeline
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
			{"convt_direct", &cnnPSO.convT}, {"resize_taps", &cnnPSO.resize}} {
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
	K := g.C * g.KH * g.KW
	if p0 < 0 || pc <= 0 || p0+pc > g.OH*g.OW {
		e.err = fmt.Errorf("metal: Im2ColNCHW pixels [%d, %d) of %d", p0, p0+pc, g.OH*g.OW)
		return
	}
	if e.ready(cnnPSO.im2col) {
		a, b, c := g.args()
		e.Dispatch(cnnPSO.im2col, [3]int{pc, K, 1}, [3]int{64, 4, 1}, x, cols, a, b, c, u32s(p0, pc, K, 0))
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
func (e *Encoder) ConvDirectBlocked(x, w, bias, out Region, g ConvGeom) {
	e.convDirectCB(blockFor(g.M), x, w, bias, out, g)
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

func (e *Encoder) convDirectCB(v int, x, w, bias, out Region, g ConvGeom) {
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
			u32s(g.C, g.M, g.OH, hasBias))
	}
}

// ConvTransposeBlocked is ConvTransposeDirect with each thread producing a
// block of a group's output channels (inputs read once per block).
func (e *Encoder) ConvTransposeBlocked(x, w, bias, out Region, g ConvGeom) {
	e.convTCB(blockFor(g.M/max(g.Group, 1)), x, w, bias, out, g)
}

func (e *Encoder) convTCB(v int, x, w, bias, out Region, g ConvGeom) {
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
			u32s(g.C/g.Group, coutG, g.M, g.OH), u32s(hasBias, g.N, blocks, 0))
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

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
