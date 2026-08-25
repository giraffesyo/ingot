//go:build arm64

package gemm

import (
	"testing"

	"golang.org/x/sys/cpu"
)

// TestQgemmSDOTKernels forces the FEAT_DotProd tier (Apple M1, Graviton2)
// with its quad-major/row-major layouts, so it is exercised on I8MM machines
// too. The SDOT instructions themselves are Armv8.2 — present everywhere
// this repo's arm64 asm runs.
func TestQgemmSDOTKernels(t *testing.T) {
	if !cpu.ARM64.HasASIMDDP {
		t.Skip("no dot-product instructions")
	}
	savedU, savedS := qkernel, qkernelS8
	savedQ, savedR := qpackQuad, qctRowMajor
	qkernel, qkernelS8 = qkernelU8S8SDOT, qkernelS8S8SDOT
	qpackQuad, qctRowMajor = true, true
	defer func() {
		qkernel, qkernelS8 = savedU, savedS
		qpackQuad, qctRowMajor = savedQ, savedR
	}()
	t.Run("u8s8", TestQgemmU8S8VsRef)
	t.Run("packedS8", TestQgemmPackedS8VsRef)
}

// BenchmarkQgemmSDOT forces the DotProd tier so its throughput is trackable
// on any arm64 machine (the M1/Graviton2 number, measured wherever).
func BenchmarkQgemmSDOT(b *testing.B) {
	if !cpu.ARM64.HasASIMDDP {
		b.Skip("no dot-product instructions")
	}
	savedU, savedS := qkernel, qkernelS8
	savedQ, savedR := qpackQuad, qctRowMajor
	qkernel, qkernelS8 = qkernelU8S8SDOT, qkernelS8S8SDOT
	qpackQuad, qctRowMajor = true, true
	defer func() {
		qkernel, qkernelS8 = savedU, savedS
		qpackQuad, qctRowMajor = savedQ, savedR
	}()
	BenchmarkQgemmPackedS8(b)
}
