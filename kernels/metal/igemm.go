//go:build darwin && arm64

package metal

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
)

// igemmSrc: implicit-GEMM convolution (group 1, NCHW) on Metal 4's
// matmul2d. Each threadgroup (4 simdgroups) owns a 64-channel × 64-pixel
// output tile of one image; per 32-deep slice of K = C·KH·KW it stages the
// weight tile and the im2col tile — gathered from the input on the fly,
// zero-padded at every edge — in threadgroup memory and accumulates with
// the matrix units into a cooperative register tile. The im2col matrix
// never reaches device memory. Bias, activation and post-affine are
// applied in registers before the single store. T is the operand type:
// float, or bfloat (weights pre-converted, activations rounded on staging).
const igemmSrc = `
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
using ext = dextents<int32_t, 2>;

constant constexpr int BM = 64, BN = 64, BK = 32;

struct IGArgs {
	uint C, H, W, OW;      // input channels and plane, output width
	uint KH, KW, SH, SW;
	uint DH, DW, PT, PL;
	uint M, OH, K, hasBias;
	uint act, pad0, pad1, pad2;
	float alpha, beta, scale, shift;
};

static float ig_act(float v, uint act, float alpha, float beta) {
	switch (act) {
	case 1: return max(v, 0.0f);
	case 2: return v * clamp(v / 6.0f + 0.5f, 0.0f, 1.0f);
	case 3: return clamp(alpha * v + beta, 0.0f, 1.0f);
	case 4: return 1.0f / (1.0f + exp(-v));
	case 5: return v / (1.0f + exp(-v));
	case 6: return min(max(v, alpha), beta);
	case 7: return v >= 0 ? v : alpha * v;
	default: return v;
	}
}

template <typename T, typename TW>
void conv_igemm(device const float* x, device const TW* w, device const float* bias, device float* o,
                constant IGArgs& a, uint3 tg, uint tid, threadgroup T* Wt, threadgroup T* Ct) {
	// Wt [m][k] and Ct [k][p]: the staged tiles (declared by the kernels).
	const uint P = a.OH * a.OW, m0 = tg.y * BM, p0 = tg.x * BN, n = tg.z;
	device const float* xn = x + n * a.C * a.H * a.W;
	tensor<threadgroup T, ext, tensor_inline> tA(Wt, ext(BK, BM));
	tensor<threadgroup T, ext, tensor_inline> tB(Ct, ext(BN, BK));
	constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
	matmul2d<desc, execution_simdgroups<4>> op;
	auto cD = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
	#pragma unroll
	for (uint16_t i = 0; i < cD.get_capacity(); ++i) if (cD.is_valid_element(i)) cD[i] = 0;
	const uint KK = a.KH * a.KW;
	// Each thread stages one pixel column (tid % BN) of the column tile for
	// every other k (tid / BN + 2j): its input window origin is fixed.
	const uint pc = tid % BN, kr = tid / BN, p = p0 + pc;
	const bool pin = p < P;
	const int oy0 = pin ? int((p / a.OW) * a.SH) - int(a.PT) : 0;
	const int ox0 = pin ? int((p % a.OW) * a.SW) - int(a.PL) : 0;
	// Weights: thread stages row m = tid / 2, half the k slice (coalesced
	// along k).
	const uint wm = tid / 2, wk = (tid % 2) * (BK / 2);
	device const TW* wrow = w + (m0 + wm) * a.K;
	const bool win = m0 + wm < a.M;
	for (uint k0 = 0; k0 < a.K; k0 += BK) {
		#pragma unroll
		for (uint j = 0; j < BK / 2; j++) {
			const uint k = k0 + wk + j;
			Wt[wm * BK + wk + j] = (win && k < a.K) ? T(wrow[k]) : T(0);
		}
		#pragma unroll 4
		for (uint kk = kr; kk < BK; kk += 2) {
			const uint k = k0 + kk;
			float v = 0;
			if (pin && k < a.K) {
				const uint ch = k / KK, r = k - ch * KK, ky = r / a.KW, kx = r - ky * a.KW;
				const int iy = oy0 + int(ky * a.DH), ix = ox0 + int(kx * a.DW);
				if (iy >= 0 && iy < int(a.H) && ix >= 0 && ix < int(a.W)) v = xn[(ch * a.H + uint(iy)) * a.W + uint(ix)];
			}
			Ct[kk * BN + pc] = T(v);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		op.run(tA, tB, cD);
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	device float* on = o + n * a.M * P;
	#pragma unroll
	for (uint16_t i = 0; i < cD.get_capacity(); ++i) {
		if (!cD.is_valid_element(i)) continue;
		auto idx = cD.get_multidimensional_index(i); // (pixel, channel)
		const uint p = p0 + idx[0], m = m0 + idx[1];
		if (p >= P || m >= a.M) continue;
		float v = cD[i] + (a.hasBias != 0 ? bias[m] : 0.0f);
		on[m * P + p] = ig_act(v, a.act, a.alpha, a.beta) * a.scale + a.shift;
	}
}

kernel void conv_igemm_f32(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                           device const float* bias [[buffer(2)]], device float* o [[buffer(3)]],
                           constant IGArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
	threadgroup float Wt[BM * BK], Ct[BK * BN];
	conv_igemm<float, float>(x, w, bias, o, a, tg, tid, Wt, Ct);
}
kernel void conv_igemm_bf16(device const float* x [[buffer(0)]], device const bfloat* w [[buffer(1)]],
                            device const float* bias [[buffer(2)]], device float* o [[buffer(3)]],
                            constant IGArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
	threadgroup bfloat Wt[BM * BK], Ct[BK * BN];
	conv_igemm<bfloat, bfloat>(x, w, bias, o, a, tg, tid, Wt, Ct);
}
`

var igemmPSO struct {
	once      sync.Once
	f32, bf16 *Pipeline
	err       error
}

// PrepareIGEMM compiles the implicit-GEMM convolution kernels.
func (d *Device) PrepareIGEMM() error {
	igemmPSO.once.Do(func() {
		if igemmPSO.f32, igemmPSO.err = d.Compile(igemmSrc, "conv_igemm_f32"); igemmPSO.err != nil {
			return
		}
		igemmPSO.bf16, igemmPSO.err = d.Compile(igemmSrc, "conv_igemm_bf16")
	})
	return igemmPSO.err
}

// ConvEpilogue is a convolution's fused output transform:
// act(v + bias)·Scale + Shift (Act codes as for Encoder.Act; Scale 0 is
// treated as 1).
type ConvEpilogue struct {
	Act                        int
	Alpha, Beta, Scale, Shift float32
}

// ConvIGEMM computes a group-1 convolution as an implicit GEMM on the
// matrix units, with bias and epilogue fused. w is f32 [M, C·KH·KW], or
// bf16 of that layout when bf16 is set (activations are then rounded to
// bf16 too). bias may be the zero Region.
func (e *Encoder) ConvIGEMM(x, w, bias, out Region, g ConvGeom, bf16 bool, ep ConvEpilogue) {
	if g.Group != 1 {
		e.err = fmt.Errorf("metal: ConvIGEMM group=%d", g.Group)
		return
	}
	p := igemmPSO.f32
	if bf16 {
		p = igemmPSO.bf16
	}
	if !e.ready(p) {
		return
	}
	hasBias := 1
	if bias.B == nil {
		bias, hasBias = w, 0
	}
	if ep.Scale == 0 {
		ep.Scale = 1
	}
	args := u32s(g.C, g.H, g.W, g.OW, g.KH, g.KW, g.SH, g.SW, g.DH, g.DW, g.PT, g.PL,
		g.M, g.OH, g.C*g.KH*g.KW, hasBias, ep.Act, 0, 0, 0)
	for _, f := range [4]float32{ep.Alpha, ep.Beta, ep.Scale, ep.Shift} {
		args = binary.LittleEndian.AppendUint32(args, math.Float32bits(f))
	}
	P := g.OH * g.OW
	e.Dispatch(p, [3]int{(P + 63) / 64 * 128, (g.M + 63) / 64, g.N}, [3]int{128, 1, 1}, x, w, bias, out, args)
}
