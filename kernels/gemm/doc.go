// Package gemm implements single-precision general matrix multiply:
//
//	C[m×n] = alpha * A[m×k] · B[k×n] + beta * C[m×n]
//
// All matrices are row-major with explicit leading dimensions (row strides).
//
// Two implementations exist:
//   - Ref: naive triple loop, the correctness oracle.
//   - Sgemm: Goto/BLIS-style cache-blocked, packed, register-tiled kernel with
//     a per-architecture micro-kernel (Go fallback; asm where available) and
//     goroutine parallelism over N-blocks.
//
// Blocking parameters (MC, KC, NC, MR, NR) are per-arch constants in params_*.go.
package gemm
