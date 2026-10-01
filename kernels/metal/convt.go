//go:build darwin && arm64

package metal

import (
	"fmt"
	"sync"
)

// 1-D transposed convolution as GEMM + col2im: col[(oc·K + k), i] =
// Σ_c W[c, oc, k]·x[c, i] is one matrix-unit GEMM against the transposed
// weight, and col2im gathers each output out[oc, t] = bias[oc] +
// Σ col[oc·K + k, i] over t = i·s + k — every output by one thread, so
// chunks of inputs accumulate without atomics. The vocoder shape
// (kernel 2s, hundreds of channels) where the direct kernel re-reads
// inputs per output channel.
const convtSrc = `
#include <metal_stdlib>
using namespace metal;

// p = (M, K, s, OW); q = (t0, Tc, mode, hasBias): mode 0 writes
// bias (or 0) over out [M, OW]; mode 1 adds the contributions of inputs
// [t0, t0+Tc) from col [M·K, Tc].
kernel void col2im_1d(device const float* col [[buffer(0)]], device const float* bias [[buffer(1)]],
                      device float* out [[buffer(2)]], constant uint4& p [[buffer(3)]],
                      constant uint4& q [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
	const uint M = p.x, K = p.y, s = p.z, OW = p.w, t0 = q.x, Tc = q.y;
	const uint oc = g.y;
	if (oc >= M) return;
	if (q.z == 0) {
		if (g.x < OW) out[oc * OW + g.x] = q.w ? bias[oc] : 0.0f;
		return;
	}
	const uint t = t0 * s + g.x; // outputs this chunk can touch start at t0·s
	if (t >= OW || g.x >= (Tc - 1) * s + K) return;
	float acc = 0;
	// i = (t - k)/s for k ≡ t (mod s), k < K, i in [t0, t0+Tc).
	for (uint k = t % s; k < K && k <= t; k += s) {
		const uint i = (t - k) / s;
		if (i >= t0 && i < t0 + Tc) acc += col[(oc * K + k) * Tc + (i - t0)];
	}
	out[oc * OW + t] += acc;
}
`

var convtPSO struct {
	once   sync.Once
	col2im *Pipeline
	err    error
}

// PrepareConvT compiles the GEMM conv-transpose kernels.
func (d *Device) PrepareConvT() error {
	convtPSO.once.Do(func() { convtPSO.col2im, convtPSO.err = d.Compile(convtSrc, "col2im_1d") })
	return convtPSO.err
}

// ConvTranspose1D encodes out [M, OW] = convtranspose(x [C, T]) (stride s,
// kernel K, no padding or dilation, group 1; OW = (T-1)·s + K) for one
// image: wT is the weight transposed to [M·K, C] (row-major), col scratch
// of at least M·K·chunk floats; inputs go in chunks of chunk.
func (e *Encoder) ConvTranspose1D(x, wT, bias, out, col Region, C, T, M, K, s, chunk int, hasBias bool) {
	if e.err != nil {
		return
	}
	if convtPSO.col2im == nil {
		e.err = fmt.Errorf("metal: ConvTranspose1D before PrepareConvT")
		return
	}
	OW := (T-1)*s + K
	hb := b2i(hasBias)
	if !hasBias {
		bias = out
	}
	e.Dispatch(convtPSO.col2im, [3]int{OW, M, 1}, [3]int{256, 1, 1}, col, bias, out, u32s(M, K, s, OW), u32s(0, 1, 0, hb))
	for t0 := 0; t0 < T; t0 += chunk {
		tc := min(chunk, T-t0)
		e.Gemm(Gemm{M: M * K, N: tc, K: C, A: wT, B: Region{B: x.B, Off: x.Off + 4*t0}, LDB: T, C: col, LDC: tc})
		e.Dispatch(convtPSO.col2im, [3]int{(tc-1)*s + K, M, 1}, [3]int{256, 1, 1}, col, bias, out,
			u32s(M, K, s, OW), u32s(t0, tc, 1, hb))
	}
}
