//go:build darwin && arm64

package metal

import (
	"fmt"
	"sync"
)

// Image kernels over NHWC activations ([H, W, C] row-major, one image):
// convolutions become GEMMs over im2col rows, per-pixel channel ops are row
// ops.
const convSrc = `
#include <metal_stdlib>
using namespace metal;

// cols[p, (ci*3 + kh)*3 + kw] = x[h+kh-1, w+kw-1, ci] (zero outside), for
// pixels p in [p0, p0+rows) of an H×W image — the K order of PyTorch's
// [co, ci, 3, 3] weights. p = (H, W, C, p0).
kernel void im2col3x3(device const float* x [[buffer(0)]], device float* cols [[buffer(1)]],
                      constant uint4& p [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
	const uint H = p.x, W = p.y, C = p.z, K = C * 9;
	if (i.x >= K) return;
	const uint pix = p.w + i.y, h = pix / W, w = pix % W;
	const uint ci = i.x / 9, kh = (i.x / 3) % 3, kw = i.x % 3;
	const int hh = int(h) + int(kh) - 1, ww = int(w) + int(kw) - 1;
	cols[i.y * K + i.x] = (hh < 0 || ww < 0 || hh >= int(H) || ww >= int(W)) ? 0.0f : x[(uint(hh) * W + uint(ww)) * C + ci];
}

// Strided 3×3 im2col: output pixel (oh, ow) of an OH×OW grid reads input
// (oh·s + kh − pt, ow·s + kw − pl), zero outside. p = (H, W, C, p0),
// q = (OW, stride, padTop, padLeft).
kernel void im2col3x3s(device const float* x [[buffer(0)]], device float* cols [[buffer(1)]],
                       constant uint4& p [[buffer(2)]], constant uint4& q [[buffer(3)]],
                       uint2 i [[thread_position_in_grid]]) {
	const uint H = p.x, W = p.y, C = p.z, K = C * 9;
	if (i.x >= K) return;
	const uint pix = p.w + i.y, oh = pix / q.x, ow = pix % q.x;
	const uint ci = i.x / 9, kh = (i.x / 3) % 3, kw = i.x % 3;
	const int hh = int(oh * q.y + kh) - int(q.z), ww = int(ow * q.y + kw) - int(q.w);
	cols[i.y * K + i.x] = (hh < 0 || ww < 0 || hh >= int(H) || ww >= int(W)) ? 0.0f : x[(uint(hh) * W + uint(ww)) * C + ci];
}

// AvgDown3D for one NHWC frame: space-to-depth by 2 with a zero frame
// prepended in time when ft = 2, then each output channel the mean of its
// group of Cin·ft·4/Cout channels. x [2·OH, 2·OW, Cin] → o [OH, OW, Cout];
// p = (OH, OW, Cin, Cout), q = (ft, 0, 0, 0).
kernel void avgdown_s2d(device const float* x [[buffer(0)]], device float* o [[buffer(1)]],
                        constant uint4& p [[buffer(2)]], constant uint4& q [[buffer(3)]],
                        uint3 i [[thread_position_in_grid]]) {
	const uint OW = p.y, Cin = p.z, Cout = p.w, ft = q.x;
	if (i.x >= Cout) return;
	const uint G = Cin * ft * 4 / Cout, oh = i.z, ow = i.y;
	float acc = 0;
	for (uint g = 0; g < G; g++) {
		const uint k = i.x * G + g, c = k / (ft * 4), t = (k / 4) % ft, hs = (k / 2) % 2, ws = k % 2;
		if (ft > 1 && t == 0) continue; // the padded (first) frame
		acc += x[((2 * oh + hs) * 2 * OW + 2 * ow + ws) * Cin + c];
	}
	o[(oh * OW + ow) * Cout + i.x] = acc / G;
}

// dst = src over n floats.
kernel void copy_f32(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                     constant uint4& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
	if (i < p.x) dst[i] = src[i];
}

// erf with |error| < 1.2e-7 (Numerical Recipes erfc, Chebyshev fit).
static float erf_acc(float x) {
	const float z = fabs(x), t = 1.0f / (1.0f + 0.5f * z);
	const float r = t * exp(-z * z - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f +
		t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f +
		t * (-0.82215223f + t * 0.17087277f)))))))));
	return x >= 0 ? 1.0f - r : r - 1.0f;
}

// x = gelu(x) over n elements: p.y = 0 exact (erf), 1 tanh approximation.
kernel void gelu_inplace(device float* x [[buffer(0)]], constant uint4& p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
	if (i >= p.x) return;
	const float v = x[i];
	x[i] = p.y == 0 ? 0.5f * v * (1.0f + erf_acc(v * 0.70710678f))
	                : 0.5f * v * (1.0f + precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
}

// x = silu(x) over n elements.
kernel void silu_inplace(device float* x [[buffer(0)]], constant uint4& p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
	if (i >= p.x) return;
	const float v = x[i];
	x[i] = v / (1.0f + exp(-v));
}

// x[r, c] += b[c]; p = (cols, ld, 0, 0).
kernel void add_bias(device float* x [[buffer(0)]], device const float* b [[buffer(1)]],
                     constant uint4& p [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
	if (i.x >= p.x) return;
	x[i.y * p.y + i.x] += b[i.x];
}

// out[2H, 2W, C] = nearest-2x upsample of x[H, W, C]; p = (H, W, C, 0).
kernel void upsample2x(device const float* x [[buffer(0)]], device float* o [[buffer(1)]],
                       constant uint4& p [[buffer(2)]], uint3 i [[thread_position_in_grid]]) {
	const uint W = p.y, C = p.z;
	if (i.x >= C) return;
	o[(i.z * 2 * W + i.y) * C + i.x] = x[((i.z / 2) * W + i.y / 2) * C + i.x];
}

// Depth-to-space by 2 with a channel map: out[2h+hs, 2w+ws, oc] =
// x[h, w, idx[(oc*2 + hs)*2 + ws]] for x [H, W, Cin], out [2H, 2W, Cout].
// p = (H, W, Cin, Cout).
kernel void d2s2_map(device const float* x [[buffer(0)]], device float* o [[buffer(1)]],
                     device const uint* idx [[buffer(2)]], constant uint4& p [[buffer(3)]],
                     uint3 i [[thread_position_in_grid]]) {
	const uint W = p.y, Cin = p.z, Cout = p.w;
	if (i.x >= Cout) return;
	const uint oy = i.z, ox = i.y, hs = oy % 2, ws = ox % 2;
	o[(oy * 2 * W + ox) * Cout + i.x] = x[((oy / 2) * W + ox / 2) * Cin + idx[(i.x * 2 + hs) * 2 + ws]];
}
`

var convPSO struct {
	once                                  sync.Once
	im2col, silu, bias, upsample, d2s, cp *Pipeline
	gelu, im2colS, avgDown                *Pipeline
	err                                   error
}

// PrepareConv compiles the image kernels (call before Device.Run).
func (d *Device) PrepareConv() error {
	if err := d.Prepare(); err != nil {
		return err
	}
	convPSO.once.Do(func() {
		for _, k := range []struct {
			name string
			dst  **Pipeline
		}{{"im2col3x3", &convPSO.im2col}, {"silu_inplace", &convPSO.silu}, {"add_bias", &convPSO.bias},
			{"upsample2x", &convPSO.upsample}, {"d2s2_map", &convPSO.d2s}, {"copy_f32", &convPSO.cp},
			{"gelu_inplace", &convPSO.gelu}, {"im2col3x3s", &convPSO.im2colS}, {"avgdown_s2d", &convPSO.avgDown}} {
			if *k.dst, convPSO.err = d.Compile(convSrc, k.name); convPSO.err != nil {
				return
			}
		}
	})
	return convPSO.err
}

// Im2Col3x3 writes im2col rows for pixels [p0, p0+rows) of x [H, W, C]
// (pad 1, stride 1) into cols [rows, 9C].
func (e *Encoder) Im2Col3x3(x, cols Region, h, w, c, p0, rows int) {
	if e.ready(convPSO.im2col) {
		e.Dispatch(convPSO.im2col, [3]int{9 * c, rows, 1}, [3]int{256, 1, 1}, x, cols, u32s(h, w, c, p0))
	}
}

// Im2Col3x3Strided writes im2col rows for output pixels [p0, p0+rows) of
// an oh×ow grid over x [H, W, C]: input (oh·stride + kh − padTop,
// ow·stride + kw − padLeft), zero outside.
func (e *Encoder) Im2Col3x3Strided(x, cols Region, h, w, c, ow, stride, padTop, padLeft, p0, rows int) {
	if e.ready(convPSO.im2colS) {
		e.Dispatch(convPSO.im2colS, [3]int{9 * c, rows, 1}, [3]int{256, 1, 1}, x, cols, u32s(h, w, c, p0), u32s(ow, stride, padTop, padLeft))
	}
}

// AvgDownS2D is QwenImage21AvgDown3D over one NHWC frame x [2·oh, 2·ow,
// cin] into out [oh, ow, cout] (ft 2: a zero frame averaged in).
func (e *Encoder) AvgDownS2D(x, out Region, oh, ow, cin, cout, ft int) {
	if e.ready(convPSO.avgDown) {
		e.Dispatch(convPSO.avgDown, [3]int{cout, ow, oh}, [3]int{min(cout, 256), 1, 1}, x, out, u32s(oh, ow, cin, cout), u32s(ft, 0, 0, 0))
	}
}

// CopyF32 copies n floats.
func (e *Encoder) CopyF32(src, dst Region, n int) {
	if e.ready(convPSO.cp) {
		e.Dispatch(convPSO.cp, [3]int{n, 1, 1}, [3]int{256, 1, 1}, src, dst, u32s(n, 0, 0, 0))
	}
}

// SiLU applies x = silu(x) in place over n elements.
func (e *Encoder) SiLU(x Region, n int) {
	if e.ready(convPSO.silu) {
		e.Dispatch(convPSO.silu, [3]int{n, 1, 1}, [3]int{256, 1, 1}, x, u32s(n, 0, 0, 0))
	}
}

// GELU applies gelu in place over n elements (tanh: the tanh
// approximation; otherwise exact erf).
func (e *Encoder) GELU(x Region, n int, tanh bool) {
	mode := 0
	if tanh {
		mode = 1
	}
	if e.ready(convPSO.gelu) {
		e.Dispatch(convPSO.gelu, [3]int{n, 1, 1}, [3]int{256, 1, 1}, x, u32s(n, mode, 0, 0))
	}
}

// AddBias adds b[c] to every row of x [rows, cols].
func (e *Encoder) AddBias(x, b Region, rows, cols, ld int) {
	if e.ready(convPSO.bias) {
		e.Dispatch(convPSO.bias, [3]int{cols, rows, 1}, [3]int{256, 1, 1}, x, b, u32s(cols, ld, 0, 0))
	}
}

// Upsample2x writes the nearest-neighbour 2x upsample of x [H, W, C].
func (e *Encoder) Upsample2x(x, out Region, h, w, c int) {
	if e.ready(convPSO.upsample) {
		e.Dispatch(convPSO.upsample, [3]int{c, 2 * w, 2 * h}, [3]int{min(c, 256), 1, 1}, x, out, u32s(h, w, c, 0))
	}
}

// DepthToSpace2Map: out[2h+hs, 2w+ws, oc] = x[h, w, idx[(oc·2+hs)·2+ws]]
// for x [H, W, cin] and out [2H, 2W, cout]; idx holds cout·4 uint32s.
func (e *Encoder) DepthToSpace2Map(x, out, idx Region, h, w, cin, cout int) {
	if cout > 1<<16 {
		e.err = fmt.Errorf("metal: DepthToSpace2Map %d channels", cout)
		return
	}
	if e.ready(convPSO.d2s) {
		e.Dispatch(convPSO.d2s, [3]int{cout, 2 * w, 2 * h}, [3]int{min(cout, 256), 1, 1}, x, out, idx, u32s(h, w, cin, cout))
	}
}
