//go:build darwin && arm64

package metal

import (
	"fmt"
	"sync"
)

// flashSrc: fused attention for one head per threadgroup-column, bf16 Q/K/V
// with f32 accumulation, over keys in two segments (a cached prefix and the
// current tokens). Per 64-row query block and 64-key block: S = Q·Kᵀ into
// registers (matmul2d cooperative tensor), online softmax with per-row
// state in threadgroup memory, P staged as a bf16 tile in threadgroup
// memory, O += P·V into a register accumulator. S never reaches device
// memory.
const flashSrc = `
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef BQ
#define BQ 16
#define BK 64
#define NSG 4
#endif
constant constexpr int DH = 128;
using ext = dextents<int32_t, 2>;

struct FlashArgs {
	uint tq, n1, n2;        // queries, keys in segment 1, keys in segment 2
	uint ldq, ldk1, ldk2;   // row strides (elements) of q, k1/v1, k2/v2
	uint ldo, pad0;
	float scale, pad1, pad2, pad3;
};

// BQ, BK and NSG are set by the Go side (compileFlash).
// Each simdgroup owns BQ query rows (a threadgroup covers NSG·BQ) and runs
// its own single-simdgroup tensor ops (reduce_rows needs that scope).
kernel void flash_attn(device bfloat* Q [[buffer(0)]], device bfloat* K1 [[buffer(1)]], device bfloat* V1 [[buffer(2)]],
                       device bfloat* K2 [[buffer(3)]], device bfloat* V2 [[buffer(4)]], device float* O [[buffer(5)]],
                       constant FlashArgs& a [[buffer(6)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
	threadgroup float m_all[NSG * BQ], l_all[NSG * BQ], c_all[NSG * BQ], b_all[NSG * BQ];
	threadgroup bfloat P_all[NSG * BQ * BK];
	threadgroup float* m_run = m_all + sg * BQ;
	threadgroup float* l_run = l_all + sg * BQ;
	threadgroup float* corr = c_all + sg * BQ;
	threadgroup float* blk = b_all + sg * BQ;
	threadgroup bfloat* Pt = P_all + sg * BQ * BK;
	const uint h = tg.y, q0 = (tg.x * NSG + sg) * BQ;
	if (q0 >= a.tq) return;
	for (uint r = lane; r < BQ; r += 32) { m_run[r] = -INFINITY; l_run[r] = 0; }

	tensor<device bfloat, ext, tensor_inline> tQ(Q + h * DH, ext(DH, a.tq), array<int32_t, 2>{1, int(a.ldq)});
	tensor<device float, ext, tensor_inline> tO(O + h * DH, ext(DH, a.tq), array<int32_t, 2>{1, int(a.ldo)});
	tensor<threadgroup bfloat, ext, tensor_inline> tP(Pt, ext(BK, BQ));
	constexpr auto dS = matmul2d_descriptor(BQ, BK, DH, false, true, false);
	constexpr auto dO = matmul2d_descriptor(BQ, DH, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
	matmul2d<dS, execution_simdgroup> opS;
	matmul2d<dO, execution_simdgroup> opO;
	auto mQ = tQ.slice(0, q0);
	auto mO = tO.slice(0, q0);

	using KT = tensor<device bfloat, ext, tensor_inline>;
	KT seg0k(K1 + h * DH, ext(DH, a.n1), array<int32_t, 2>{1, int(a.ldk1)});
	KT seg0v(V1 + h * DH, ext(DH, a.n1), array<int32_t, 2>{1, int(a.ldk1)});
	KT seg1k(K2 + h * DH, ext(DH, a.n2), array<int32_t, 2>{1, int(a.ldk2)});
	KT seg1v(V2 + h * DH, ext(DH, a.n2), array<int32_t, 2>{1, int(a.ldk2)});
	auto sK0 = seg0k.slice(0, 0);
	auto sV0 = seg0v.slice(0, 0);
	auto tPs = tP.slice(0, 0);
	auto cS = opS.get_destination_cooperative_tensor<decltype(mQ), decltype(sK0), float>();
	auto rRed = opS.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(sK0), float>();
	auto cO = opO.get_destination_cooperative_tensor<decltype(tPs), decltype(sV0), float>();
	#pragma unroll
	for (uint16_t i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0;
	simdgroup_barrier(mem_flags::mem_threadgroup);

	for (int seg = 0; seg < 2; seg++) {
		const uint n = seg == 0 ? a.n1 : a.n2;
		for (uint k0 = 0; k0 < n; k0 += BK) {
			auto mK = (seg == 0 ? seg0k : seg1k).slice(0, k0);
			auto mV = (seg == 0 ? seg0v : seg1v).slice(0, k0);
			#pragma unroll
			for (uint16_t i = 0; i < cS.get_capacity(); ++i) if (cS.is_valid_element(i)) cS[i] = 0;
			opS.run(mQ, mK, cS);
			// Scale, mask padded keys; block row max -> blk.
			#pragma unroll
			for (uint16_t i = 0; i < cS.get_capacity(); ++i) {
				if (!cS.is_valid_element(i)) continue;
				auto idx = cS.get_multidimensional_index(i); // (col, row)
				cS[i] = (k0 + idx[0] < n) ? cS[i] * a.scale : -INFINITY;
			}
			reduce_rows(cS, rRed, reduction_operation::max, -INFINITY);
			#pragma unroll
			for (uint16_t i = 0; i < rRed.get_capacity(); ++i) {
				if (rRed.is_valid_element(i)) blk[rRed.get_multidimensional_index(i)[0]] = rRed[i];
			}
			simdgroup_barrier(mem_flags::mem_threadgroup);
			for (uint r = lane; r < BQ; r += 32) {
				const float mn = max(m_run[r], blk[r]);
				corr[r] = exp(m_run[r] - mn);
				m_run[r] = mn;
			}
			simdgroup_barrier(mem_flags::mem_threadgroup);
			#pragma unroll
			for (uint16_t i = 0; i < cS.get_capacity(); ++i) {
				if (!cS.is_valid_element(i)) continue;
				auto idx = cS.get_multidimensional_index(i);
				const float p = exp(cS[i] - m_run[idx[1]]);
				Pt[idx[1] * BK + idx[0]] = bfloat(p);
				cS[i] = p;
			}
			reduce_rows(cS, rRed, reduction_operation::sum, 0.0f);
			#pragma unroll
			for (uint16_t i = 0; i < rRed.get_capacity(); ++i) {
				if (rRed.is_valid_element(i)) blk[rRed.get_multidimensional_index(i)[0]] = rRed[i];
			}
			#pragma unroll
			for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
				if (cO.is_valid_element(i)) cO[i] *= corr[cO.get_multidimensional_index(i)[1]];
			}
			simdgroup_barrier(mem_flags::mem_threadgroup);
			for (uint r = lane; r < BQ; r += 32) l_run[r] = l_run[r] * corr[r] + blk[r];
			opO.run(tPs, mV, cO);
			simdgroup_barrier(mem_flags::mem_threadgroup);
		}
	}
	simdgroup_barrier(mem_flags::mem_threadgroup);
	#pragma unroll
	for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
		if (cO.is_valid_element(i)) cO[i] /= l_run[cO.get_multidimensional_index(i)[1]];
	}
	cO.store(mO);
}
`

var flashPSO struct {
	once    sync.Once
	p       *Pipeline
	bq, nsg int
	err     error
}

// Flash tile: query rows per simdgroup, key block, simdgroups per group.
// Swept on the M5 Pro at a 1024² DiT step (4096 queries, 32 heads, 4127
// keys): BQ=16 is the sweet spot (17.6-17.9 ms for BK 32/64 × NSG 4/8;
// BQ=32 44 ms, BQ=8 64 ms, BK=128 41 ms or over the 32 KB threadgroup
// memory limit) vs 35.5 ms for the per-head GEMM/softmax/GEMM chain.
var flashBQ, flashBK, flashNSG = 16, 64, 4

// PrepareFlash compiles the fused attention kernel.
func (d *Device) PrepareFlash() error {
	if err := d.Prepare(); err != nil {
		return err
	}
	flashPSO.once.Do(func() {
		flashPSO.p, flashPSO.err = d.compileFlash(flashBQ, flashBK, flashNSG)
		flashPSO.bq, flashPSO.nsg = flashBQ, flashNSG
	})
	return flashPSO.err
}

func (d *Device) compileFlash(bq, bk, nsg int) (*Pipeline, error) {
	return d.Compile(fmt.Sprintf("#define BQ %d\n#define BK %d\n#define NSG %d\n", bq, bk, nsg)+flashSrc, "flash_attn")
}

// Flash describes fused softmax(scale·Q·Kᵀ)·V for heads of width 128 over
// keys in two segments (K1/V1 then K2/V2; either may be empty), all bf16
// [rows, ld] with heads at column h·128; O is f32. Every head of Q's
// Heads is computed.
type Flash struct {
	Q, K1, V1, K2, V2, O Region
	Tq, N1, N2, Heads    int
	LDQ, LD1, LD2, LDO   int
	Scale                float32
}

// Flash encodes f. Call Device.PrepareFlash first.
func (e *Encoder) Flash(f Flash) {
	if e.err != nil {
		return
	}
	if flashPSO.p == nil {
		e.err = fmt.Errorf("metal: Flash before PrepareFlash")
		return
	}
	if f.N1+f.N2 == 0 || f.Tq == 0 {
		e.err = fmt.Errorf("metal: Flash with no queries or keys")
		return
	}
	args := append(u32s(f.Tq, f.N1, f.N2, f.LDQ, f.LD1, f.LD2, f.LDO, 0), f32c(f.Scale)...)
	rows := flashPSO.bq * flashPSO.nsg
	e.Dispatch(flashPSO.p, [3]int{(f.Tq + rows - 1) / rows * 32 * flashPSO.nsg, f.Heads, 1}, [3]int{32 * flashPSO.nsg, 1, 1},
		f.Q, f.K1, f.V1, f.K2, f.V2, f.O, args)
}
