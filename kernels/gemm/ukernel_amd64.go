//go:build amd64

package gemm

import (
	"os"

	"golang.org/x/sys/cpu"
)

// microKernel is the active micro-kernel, chosen once at process start.
//
// Default amd64 fast path is the AVX2 6×16 kernel. An AVX-512 kernel (same tile)
// exists and is correctness-tested, but is NOT auto-selected: on AMD Zen4 — the
// only AVX-512 CPU measured — it is throughput-parity with AVX2 (Zen4 runs
// 512-bit ops on double-pumped 256-bit units, so there is no FLOP gain), and
// AVX-512 carries a downclock risk on some Intel parts. On a CPU with true
// 512-bit datapaths (Intel server) it should win; select it there with
// OCR_GEMM_KERNEL=avx512 and benchmark. Set OCR_GEMM_KERNEL to avx2 / avx512 /
// generic to force a kernel (for A/B measurement); unknown/empty = auto.
var microKernel = pickMicroKernel()

// HasAVX2 / HasAVX512 report which fast paths the CPU supports.
var (
	HasAVX2   = cpu.X86.HasAVX2 && cpu.X86.HasFMA
	HasAVX512 = cpu.X86.HasAVX512F
)

// ActiveKernel names the selected micro-kernel ("avx512"|"avx2"|"generic").
var ActiveKernel string

func pickMicroKernel() func(kc int, ap, bp []float32, c []float32, ldc int, accumulate bool) {
	switch os.Getenv("OCR_GEMM_KERNEL") {
	case "avx512":
		if HasAVX512 {
			ActiveKernel = "avx512"
			return microKernelAVX512
		}
	case "avx2":
		if HasAVX2 {
			ActiveKernel = "avx2"
			return microKernelAVX2
		}
	case "generic":
		ActiveKernel = "generic"
		return microKernelGeneric
	}
	// Auto: AVX2 is the default fast path (see the doc comment above).
	if HasAVX2 {
		ActiveKernel = "avx2"
		return microKernelAVX2
	}
	ActiveKernel = "generic"
	return microKernelGeneric
}

//go:noescape
func microKernelAVX2(kc int, ap, bp []float32, c []float32, ldc int, accumulate bool)

//go:noescape
func microKernelAVX512(kc int, ap, bp []float32, c []float32, ldc int, accumulate bool)
