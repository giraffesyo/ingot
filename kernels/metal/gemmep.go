//go:build darwin && arm64

package metal

import (
	"fmt"
	"sync"
)

// gemmEpSrc: C[M,N] = act(A[M,K] · op(B) + bias) with A and B bf16 and f32
// accumulation, the epilogue applied to the product while it is still in
// registers (matmul2d's cooperative destination): bias, activation and the
// rounding of a bf16 C cost no extra pass over C — at a 4096×8192 MLP
// output the separate bias, GELU and bf16-cast passes take as long as the
// product itself. Tiles as in gemmSrc (64×64 over 4 simdgroups).
const gemmEpSrc = `
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
` + actFnSrc + `
// d = (M, N, K, act), ld = (lda, ldb, ldc, hasBias).
template <bool NT, typename TC>
void gemm_ep(device bfloat* A, device bfloat* B, device TC* C, device const float* bias, uint4 d, uint4 ld, uint2 tg) {
	using ext = dextents<int32_t, 2>;
	constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, NT, false,
	                                          matmul2d_descriptor::mode::multiply_accumulate);
	matmul2d<desc, execution_simdgroups<4>> op;
	const uint row0 = tg.y * 64, col0 = tg.x * 64;
	tensor<device bfloat, ext, tensor_inline> tA(A, ext(d.z, d.x), array<int32_t, 2>{1, int(ld.x)});
	auto mA = tA.slice(0, row0);
	const uint2 eb = NT ? uint2(d.z, d.y) : uint2(d.y, d.z); // B's extents, innermost first
	tensor<device bfloat, ext, tensor_inline> tB(B, ext(eb.x, eb.y), array<int32_t, 2>{1, int(ld.y)});
	auto mB = NT ? tB.slice(0, col0) : tB.slice(col0, 0);
	auto cC = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
	#pragma unroll
	for (uint16_t i = 0; i < cC.get_capacity(); ++i) if (cC.is_valid_element(i)) cC[i] = 0;
	op.run(mA, mB, cC);
	#pragma unroll
	for (uint16_t i = 0; i < cC.get_capacity(); ++i) {
		if (!cC.is_valid_element(i)) continue;
		auto idx = cC.get_multidimensional_index(i); // (col, row) within the tile
		const uint col = col0 + idx[0], row = row0 + idx[1];
		if (col >= d.y || row >= d.x) continue;
		float v = cC[i];
		if (ld.w != 0) v += bias[col];
		C[row * ld.z + col] = TC(act_fn(v, d.w, 0.0f, 0.0f));
	}
}
#define GEMM_EP(name, NT, TC) \
kernel void name(device bfloat* A [[buffer(0)]], device bfloat* B [[buffer(1)]], device TC* C [[buffer(2)]], \
                 device const float* bias [[buffer(3)]], constant uint4& d [[buffer(4)]], constant uint4& ld [[buffer(5)]], \
                 uint2 tg [[threadgroup_position_in_grid]]) { gemm_ep<NT, TC>(A, B, C, bias, d, ld, tg); }
GEMM_EP(gemm_ep_nt, true, float)
GEMM_EP(gemm_ep_nt_bf16, true, bfloat)
GEMM_EP(gemm_ep_nn, false, float)
GEMM_EP(gemm_ep_nn_bf16, false, bfloat)
`

// GemmEp describes C[M,N] = act(A[M,K] · op(B) + Bias): bf16 A and B
// (row-major, row strides in elements; TransB as in Gemm), f32
// accumulation, and the epilogue fused into the product. Bias is f32 [N]
// (the zero Region: none); Act is a parameterless Act* code (ActNone,
// ActRelu, ActSiLU, ActGeluErf, ActGeluTanh, ...); CBF16 writes C as bf16
// (a following bf16 product's operand) instead of f32. C must not overlap
// A or B.
type GemmEp struct {
	M, N, K       int
	A, B, C       Region
	LDA, LDB, LDC int // 0 = packed
	TransB, CBF16 bool
	Bias          Region
	Act           int
}

var gemmEpPSO struct {
	once                   sync.Once
	nt, ntBF16, nn, nnBF16 *Pipeline
	err                    error
}

func (d *Device) gemmEpPipelines() error {
	gemmEpPSO.once.Do(func() {
		for _, k := range []struct {
			name string
			dst  **Pipeline
		}{{"gemm_ep_nt", &gemmEpPSO.nt}, {"gemm_ep_nt_bf16", &gemmEpPSO.ntBF16}, {"gemm_ep_nn", &gemmEpPSO.nn},
			{"gemm_ep_nn_bf16", &gemmEpPSO.nnBF16}} {
			if *k.dst, gemmEpPSO.err = d.Compile(gemmEpSrc, k.name); gemmEpPSO.err != nil {
				return
			}
		}
	})
	return gemmEpPSO.err
}

// GemmEp encodes g. Call Device.Prepare first.
func (e *Encoder) GemmEp(g GemmEp) {
	if e.err != nil {
		return
	}
	if gemmEpPSO.nt == nil {
		e.err = fmt.Errorf("metal: GemmEp before Device.Prepare")
		return
	}
	if g.M <= 0 || g.N <= 0 || g.K <= 0 {
		e.err = fmt.Errorf("metal: gemm dims %d×%d×%d", g.M, g.N, g.K)
		return
	}
	lda, ldc := or(g.LDA, g.K), or(g.LDC, g.N)
	bRows, bCols := g.K, g.N
	p, p16 := gemmEpPSO.nn, gemmEpPSO.nnBF16
	if g.TransB {
		bRows, bCols = g.N, g.K
		p, p16 = gemmEpPSO.nt, gemmEpPSO.ntBF16
	}
	ldb := or(g.LDB, bCols)
	cesz := 4
	if g.CBF16 {
		p, cesz = p16, 2
	}
	bias, hasBias := g.Bias, 1
	if bias.B == nil {
		bias, hasBias = g.A, 0 // any bound buffer when unused
	}
	for _, s := range []struct {
		r                   Region
		rows, cols, ld, esz int
	}{{g.A, g.M, g.K, lda, 2}, {g.B, bRows, bCols, ldb, 2}, {g.C, g.M, g.N, ldc, cesz}, {bias, 1, g.N * hasBias, 0, 4}} {
		if need := s.r.Off + ((s.rows-1)*s.ld+s.cols)*s.esz; need > s.r.B.n {
			e.err = fmt.Errorf("metal: gemm %d×%d×%d: operand needs %d bytes, buffer has %d", g.M, g.N, g.K, need, s.r.B.n)
			return
		}
	}
	e.Dispatch(p, [3]int{(g.N + 63) / 64 * 128, (g.M + 63) / 64, 1}, [3]int{128, 1, 1},
		g.A, g.B, g.C, bias, u32s(g.M, g.N, g.K, g.Act), u32s(lda, ldb, ldc, hasBias))
}
