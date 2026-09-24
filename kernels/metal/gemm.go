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

template <typename TW>
void gemm(device float* A, device TW* W, device float* C, uint3 d, uint2 tg) {
	tensor<device float, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(d.z, d.x));
	tensor<device TW, dextents<int32_t, 2>, tensor_inline> tW(W, dextents<int32_t, 2>(d.z, d.y));
	tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(d.y, d.x));
	constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, true, false,
	                                          matmul2d_descriptor::mode::multiply);
	matmul2d<desc, execution_simdgroups<4>> op;
	auto mA = tA.slice(0, tg.y * 64);
	auto mW = tW.slice(0, tg.x * 64);
	auto mC = tC.slice(tg.x * 64, tg.y * 64);
	op.run(mA, mW, mC);
}
kernel void gemm_nt_f32(device float* A [[buffer(0)]], device float* W [[buffer(1)]], device float* C [[buffer(2)]],
                        constant uint3& d [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) { gemm(A, W, C, d, tg); }
kernel void gemm_nt_bf16(device float* A [[buffer(0)]], device bfloat* W [[buffer(1)]], device float* C [[buffer(2)]],
                         constant uint3& d [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) { gemm(A, W, C, d, tg); }
`

var gemmPSO struct {
	once     sync.Once
	f32, b16 *Pipeline
	err      error
}

// GemmNT computes c[M,N] = a[M,K] · w[N,K]ᵀ (f32; w f32, or bf16 when
// wBF16) on the GPU and waits.
func (d *Device) GemmNT(m, n, k int, a, w, c *Buffer, wBF16 bool) error {
	gemmPSO.once.Do(func() {
		if gemmPSO.f32, gemmPSO.err = d.Compile(gemmSrc, "gemm_nt_f32"); gemmPSO.err == nil {
			gemmPSO.b16, gemmPSO.err = d.Compile(gemmSrc, "gemm_nt_bf16")
		}
	})
	if gemmPSO.err != nil {
		return gemmPSO.err
	}
	if m <= 0 || n <= 0 || k <= 0 {
		return fmt.Errorf("metal: gemm dims %d×%d×%d", m, n, k)
	}
	esz := 4
	p := gemmPSO.f32
	if wBF16 {
		esz, p = 2, gemmPSO.b16
	}
	if a.n < 4*m*k || w.n < esz*n*k || c.n < 4*m*n {
		return fmt.Errorf("metal: gemm %d×%d×%d: buffers too small", m, n, k)
	}
	dims := binary.LittleEndian.AppendUint32(nil, uint32(m))
	dims = binary.LittleEndian.AppendUint32(dims, uint32(n))
	dims = binary.LittleEndian.AppendUint32(dims, uint32(k))
	dims = binary.LittleEndian.AppendUint32(dims, 0) // uint3 is 16-byte aligned
	return p.Dispatch([3]int{(n + 63) / 64 * 128, (m + 63) / 64, 1}, [3]int{128, 1, 1}, a, w, c, dims)
}
