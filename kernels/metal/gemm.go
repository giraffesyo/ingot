package metal

import (
	"encoding/binary"
	"fmt"
	"sync"
)

// gemmSrc: C[M,N] = A[M,K] · W[N,K]ᵀ (W as PyTorch stores Linear weights),
// f32, simdgroup 8×8 matrices. A threadgroup of 4 simdgroups (2×2) computes
// a 64×64 tile of C, each simdgroup 32×32 (4×4 accumulators), over K tiles
// of 32 staged in threadgroup memory. Ragged edges are zero-padded on load
// and bounds-checked on store.
const gemmSrc = `
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
constant uint BM = 64, BN = 64, BK = 32, NT = 128;

kernel void gemm_nt(device const float* A [[buffer(0)]], device const float* W [[buffer(1)]],
                    device float* C [[buffer(2)]], constant uint3& dims [[buffer(3)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint tid [[thread_index_in_threadgroup]]) {
	threadgroup float As[BM * BK];
	threadgroup float Bs[BK * BN];
	threadgroup float Cs[BM * BN];
	const uint M = dims.x, N = dims.y, K = dims.z;
	const uint m0 = tg.y * BM, n0 = tg.x * BN;
	const uint sm = (sg / 2) * 32, sn = (sg % 2) * 32;
	simdgroup_float8x8 acc[4][4];
	for (uint i = 0; i < 4; i++)
		for (uint j = 0; j < 4; j++)
			acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
	for (uint k0 = 0; k0 < K; k0 += BK) {
		for (uint i = tid; i < BM * BK; i += NT) {
			uint r = i / BK, c = i % BK, gm = m0 + r, gk = k0 + c;
			As[i] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
		}
		for (uint i = tid; i < BN * BK; i += NT) {
			uint n = i / BK, c = i % BK, gn = n0 + n, gk = k0 + c;
			Bs[c * BN + n] = (gn < N && gk < K) ? W[gn * K + gk] : 0.0f;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for (uint kk = 0; kk < BK; kk += 8) {
			simdgroup_float8x8 a[4], b[4];
			for (uint i = 0; i < 4; i++) simdgroup_load(a[i], As + (sm + i * 8) * BK + kk, BK);
			for (uint j = 0; j < 4; j++) simdgroup_load(b[j], Bs + kk * BN + sn + j * 8, BN);
			for (uint i = 0; i < 4; i++)
				for (uint j = 0; j < 4; j++)
					simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	if (m0 + BM <= M && n0 + BN <= N) {
		for (uint i = 0; i < 4; i++)
			for (uint j = 0; j < 4; j++)
				simdgroup_store(acc[i][j], C + (m0 + sm + i * 8) * N + n0 + sn + j * 8, N);
		return;
	}
	for (uint i = 0; i < 4; i++)
		for (uint j = 0; j < 4; j++)
			simdgroup_store(acc[i][j], Cs + (sm + i * 8) * BN + sn + j * 8, BN);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (uint i = tid; i < BM * BN; i += NT) {
		uint r = i / BN, c = i % BN;
		if (m0 + r < M && n0 + c < N) C[(m0 + r) * N + n0 + c] = Cs[i];
	}
}

// Full-tile variant (M % 64 == 0, N % 64 == 0, K % 8 == 0): each simdgroup
// loads its 8×8 operand tiles straight from device memory (W with the
// transpose flag), no threadgroup staging or barriers.
kernel void gemm_nt_full(device const float* A [[buffer(0)]], device const float* W [[buffer(1)]],
                         device float* C [[buffer(2)]], constant uint3& dims [[buffer(3)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
	const uint N = dims.y, K = dims.z;
	const uint m0 = tg.y * BM + (sg / 2) * 32, n0 = tg.x * BN + (sg % 2) * 32;
	simdgroup_float8x8 acc[4][4];
	for (uint i = 0; i < 4; i++)
		for (uint j = 0; j < 4; j++)
			acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
	device const float* a0 = A + m0 * K;
	device const float* w0 = W + n0 * K;
	for (uint k = 0; k < K; k += 8) {
		simdgroup_float8x8 a[4], b[4];
		for (uint i = 0; i < 4; i++) simdgroup_load(a[i], a0 + i * 8 * K + k, K);
		for (uint j = 0; j < 4; j++) simdgroup_load(b[j], w0 + j * 8 * K + k, K, ulong2(0, 0), true);
		for (uint i = 0; i < 4; i++)
			for (uint j = 0; j < 4; j++)
				simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
	}
	for (uint i = 0; i < 4; i++)
		for (uint j = 0; j < 4; j++)
			simdgroup_store(acc[i][j], C + (m0 + i * 8) * N + n0 + j * 8, N);
}`

var gemmPSO struct {
	once    sync.Once
	p, full *Pipeline
	err     error
}

// GemmNT computes c[M,N] = a[M,K] · w[N,K]ᵀ on the GPU and waits.
func (d *Device) GemmNT(m, n, k int, a, w, c *Buffer) error {
	gemmPSO.once.Do(func() {
		if gemmPSO.p, gemmPSO.err = d.Compile(gemmSrc, "gemm_nt"); gemmPSO.err == nil {
			gemmPSO.full, gemmPSO.err = d.Compile(gemmSrc, "gemm_nt_full")
		}
	})
	if gemmPSO.err != nil {
		return gemmPSO.err
	}
	if m <= 0 || n <= 0 || k <= 0 {
		return fmt.Errorf("metal: gemm dims %d×%d×%d", m, n, k)
	}
	dims := binary.LittleEndian.AppendUint32(nil, uint32(m))
	dims = binary.LittleEndian.AppendUint32(dims, uint32(n))
	dims = binary.LittleEndian.AppendUint32(dims, uint32(k))
	dims = binary.LittleEndian.AppendUint32(dims, 0) // uint3 is 16-byte aligned
	gx, gy := (n+63)/64, (m+63)/64
	p := gemmPSO.p
	if m%64 == 0 && n%64 == 0 && k%8 == 0 {
		p = gemmPSO.full
	}
	return p.Dispatch([3]int{gx * 128, gy, 1}, [3]int{128, 1, 1}, a, w, c, dims)
}
