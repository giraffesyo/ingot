//go:build amd64

package gemm

import (
	"os"
	"time"

	"golang.org/x/sys/cpu"
)

// microKernel is the active micro-kernel, chosen once at process start.
//
// Auto-selection times both kernels on a hot tile at init (~150 µs) and takes
// AVX-512 only when it is decisively faster: on AMD Zen 4 the 512-bit ops run
// double-pumped on 256-bit units (throughput parity — probe keeps AVX2), on
// Zen 5 / Intel server parts with true 512-bit datapaths the probe measures
// the win directly (+12-15% GEMM, gptish -13% end-to-end on Zen 5). The probe
// also inherently prices in any frequency licensing the part applies. Set
// OCR_GEMM_KERNEL to avx2 / avx512 / generic to pin a kernel (for A/B
// measurement); unknown/empty = auto.
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
	// Auto: probe (see the doc comment above).
	if HasAVX2 {
		if HasAVX512 && avx512Faster() {
			ActiveKernel = "avx512"
			return microKernelAVX512
		}
		ActiveKernel = "avx2"
		return microKernelAVX2
	}
	ActiveKernel = "generic"
	return microKernelGeneric
}

// avx512Faster times both micro-kernels on a cache-hot 6×16 tile and reports
// whether the AVX-512 one is >5% faster (best of 3 tight loops per kernel; a
// double-pumped or downclocking part lands at parity and keeps AVX2).
func avx512Faster() bool {
	const kc = 384
	ap := make([]float32, kc*MR)
	bp := make([]float32, kc*NR)
	c := make([]float32, MR*NR)
	for i := range ap {
		ap[i] = float32(i%7) * 0.5
	}
	for i := range bp {
		bp[i] = float32(i%5) * 0.25
	}
	bench := func(k func(kc int, ap, bp []float32, c []float32, ldc int, accumulate bool)) time.Duration {
		k(kc, ap, bp, c, NR, false) // warm (page-in, decode)
		best := time.Duration(1 << 62)
		for r := 0; r < 3; r++ {
			t0 := time.Now()
			for i := 0; i < 24; i++ {
				k(kc, ap, bp, c, NR, false)
			}
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		return best
	}
	a2 := bench(microKernelAVX2)
	a5 := bench(microKernelAVX512)
	return a5*100 < a2*95
}

//go:noescape
func microKernelAVX2(kc int, ap, bp []float32, c []float32, ldc int, accumulate bool)

//go:noescape
func microKernelAVX512(kc int, ap, bp []float32, c []float32, ldc int, accumulate bool)

//go:noescape
func microKernel2AVX512(kc int, ap, bp0, bp1 []float32, c []float32, ldc int, accumulate bool)

// pairKernel reports whether the paired-panel 6x32 kernel should be used
// (adjacent full panels, avx512 active).
func pairKernel() bool { return ActiveKernel == "avx512" }
