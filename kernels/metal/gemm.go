//go:build darwin && arm64

package metal

import (
	"encoding/binary"
	"fmt"
	"sync"
)

// gemmSrc: C[M,N] = A[M,K] · W[N,K]ᵀ with A and C f32 and W (PyTorch's
// Linear layout) f32 or bf16, on Metal 4's matmul2d tensor op — the GPU's
// matrix units, f32 accumulation. Tensors are inline views of the buffers
// (extents innermost-first: A is (K, M), W is (K, N) read transposed, C is
// (N, M)). The op bounds-checks edge tiles itself. 64×64 tiles over 4
// simdgroups won a sweep on the M5 Pro (64×32 7.0, 64×64 7.8, 128×64 7.7,
// 128×128/8sg 7.8, 256×128 5.7 TFLOPS at 4096³ f32×bf16).
const gemmSrc = `
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// d = (M, N, K), ld = (lda, ldb, ldc) in elements.
template <bool NT, typename TB>
void gemm(device float* A, device TB* B, device float* C, uint4 d, uint4 ld, uint2 tg) {
	using ext = dextents<int32_t, 2>;
	tensor<device float, ext, tensor_inline> tA(A, ext(d.z, d.x), array<int32_t, 2>{1, int(ld.x)});
	tensor<device float, ext, tensor_inline> tC(C, ext(d.y, d.x), array<int32_t, 2>{1, int(ld.z)});
	constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, NT, false,
	                                          matmul2d_descriptor::mode::multiply);
	matmul2d<desc, execution_simdgroups<4>> op;
	auto mA = tA.slice(0, tg.y * 64);
	auto mC = tC.slice(tg.x * 64, tg.y * 64);
	if (NT) { // B stored [N, K]: extents (K, N)
		tensor<device TB, ext, tensor_inline> tB(B, ext(d.z, d.y), array<int32_t, 2>{1, int(ld.y)});
		auto mB = tB.slice(0, tg.x * 64);
		op.run(mA, mB, mC);
	} else { // B stored [K, N]: extents (N, K)
		tensor<device TB, ext, tensor_inline> tB(B, ext(d.y, d.z), array<int32_t, 2>{1, int(ld.y)});
		auto mB = tB.slice(tg.x * 64, 0);
		op.run(mA, mB, mC);
	}
}
#define GEMM(name, NT, TB) \
kernel void name(device float* A [[buffer(0)]], device TB* B [[buffer(1)]], device float* C [[buffer(2)]], \
                 constant uint4& d [[buffer(3)]], constant uint4& ld [[buffer(4)]], \
                 uint2 tg [[threadgroup_position_in_grid]]) { gemm<NT, TB>(A, B, C, d, ld, tg); }
GEMM(gemm_nt_f32, true, float)
GEMM(gemm_nt_bf16, true, bfloat)
GEMM(gemm_nn_f32, false, float)
`

// Gemm describes C[M,N] = A[M,K] · op(B) with row-major operands at byte
// offsets (Regions) and row strides in elements. TransB: B is stored [N,K]
// (PyTorch Linear weights, or K for Q·Kᵀ); otherwise [K,N]. BF16 selects a
// bf16 B (TransB only). A and C are f32.
type Gemm struct {
	M, N, K       int
	A, B, C       Region
	LDA, LDB, LDC int // 0 = packed (K, K or N, N)
	TransB, BF16  bool
}

var gemmPSO struct {
	once              sync.Once
	ntF32, ntBF16, nn *Pipeline
	err               error
}

func (d *Device) gemmPipelines() error {
	gemmPSO.once.Do(func() {
		for _, k := range []struct {
			name string
			dst  **Pipeline
		}{{"gemm_nt_f32", &gemmPSO.ntF32}, {"gemm_nt_bf16", &gemmPSO.ntBF16}, {"gemm_nn_f32", &gemmPSO.nn}} {
			if *k.dst, gemmPSO.err = d.Compile(gemmSrc, k.name); gemmPSO.err != nil {
				return
			}
		}
	})
	return gemmPSO.err
}

// Prepare compiles the GEMM and nn kernels (Device.Run callbacks cannot).
func (d *Device) Prepare() error {
	if err := d.gemmPipelines(); err != nil {
		return err
	}
	return d.nnPipelines()
}

// Gemm encodes g. Call Device.Prepare first.
func (e *Encoder) Gemm(g Gemm) {
	if e.err != nil {
		return
	}
	if gemmPSO.ntF32 == nil {
		e.err = fmt.Errorf("metal: Gemm before Device.Prepare")
		return
	}
	if g.M <= 0 || g.N <= 0 || g.K <= 0 {
		e.err = fmt.Errorf("metal: gemm dims %d×%d×%d", g.M, g.N, g.K)
		return
	}
	lda, ldb, ldc := or(g.LDA, g.K), g.LDB, or(g.LDC, g.N)
	bRows, bCols := g.K, g.N
	if g.TransB {
		bRows, bCols = g.N, g.K
	}
	ldb = or(ldb, bCols)
	esz := 4
	p := gemmPSO.nn
	switch {
	case g.TransB && g.BF16:
		p, esz = gemmPSO.ntBF16, 2
	case g.TransB:
		p = gemmPSO.ntF32
	case g.BF16:
		e.err = fmt.Errorf("metal: bf16 B needs TransB")
		return
	}
	span := func(r Region, rows, cols, ld, esz int) error {
		if need := r.Off + ((rows-1)*ld+cols)*esz; need > r.B.n {
			return fmt.Errorf("metal: gemm %d×%d×%d: operand needs %d bytes, buffer has %d", g.M, g.N, g.K, need, r.B.n)
		}
		return nil
	}
	for _, err := range []error{span(g.A, g.M, g.K, lda, 4), span(g.B, bRows, bCols, ldb, esz), span(g.C, g.M, g.N, ldc, 4)} {
		if err != nil {
			e.err = err
			return
		}
	}
	e.Dispatch(p, [3]int{(g.N + 63) / 64 * 128, (g.M + 63) / 64, 1}, [3]int{128, 1, 1},
		g.A, g.B, g.C, u32s(g.M, g.N, g.K, 0), u32s(lda, ldb, ldc, 0))
}

// GemmNT computes c[M,N] = a[M,K] · w[N,K]ᵀ (packed; w bf16 when wBF16) in
// its own command buffer and waits.
func (d *Device) GemmNT(m, n, k int, a, w, c *Buffer, wBF16 bool) error {
	if err := d.Prepare(); err != nil {
		return err
	}
	return d.Run(func(e *Encoder) {
		e.Gemm(Gemm{M: m, N: n, K: k, A: a.At(0), B: w.At(0), C: c.At(0), TransB: true, BF16: wBF16})
	})
}

func or(v, def int) int {
	if v == 0 {
		return def
	}
	return v
}

// u32s packs values as a little-endian uint32 constant argument.
func u32s(v ...int) []byte {
	b := make([]byte, 0, 4*len(v))
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, uint32(x))
	}
	return b
}
