//go:build arm64

package gemm

import "testing"

// BenchmarkBkernelMicro times the raw BFMMLA micro-kernel against the int8
// MMLA one at the same byte volume (kg=128: k=512 bf16 vs k=1024 int8).
func BenchmarkBkernelMicro(b *testing.B) {
	if !HasBFMMLA() {
		b.Skip("no BFMMLA")
	}
	const kg = 128
	ap := make([]uint16, kg*qMR*bKG)
	bp := make([]uint16, kg*qNR*bKG)
	var ct [qMR * qNR]float32
	flops := 2 * float64(qMR) * float64(qNR) * float64(kg*bKG)
	b.Run("bf16", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			bkernelBF16(kg, &ap[0], &bp[0], &ct[0])
		}
		b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
	})
	api := make([]int8, kg*qMR*qKG)
	bpi := make([]int8, kg*qNR*qKG)
	var cti [qMR * qNR]int32
	iops := 2 * float64(qMR) * float64(qNR) * float64(kg*qKG)
	b.Run("int8", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			qkernelS8S8(kg, &api[0], &bpi[0], &cti[0])
		}
		b.ReportMetric(iops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GOPS")
	})
}

// BenchmarkBFDot: raw NEON BFDOT throughput (4 independent chains).
func BenchmarkBFDot(b *testing.B) {
	if !HasBFMMLA() {
		b.Skip("no BF16")
	}
	src := make([]uint16, 64)
	b.Run("bfdot", func(b *testing.B) {
		bfdotProbe(int64(b.N), &src[0])
		b.ReportMetric(float64(b.N)*4*16/b.Elapsed().Seconds()/1e9, "GFLOPS")
	})
}
