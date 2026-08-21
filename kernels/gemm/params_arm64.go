//go:build arm64

package gemm

// Blocking parameters for the NEON 8×12 micro-kernel (ukernel_arm64.s).
// Apple M-series: L1d 128KB (P-core), large shared L2. A panel MC×KC×4 bytes,
// B panel KC×NC×4 bytes. Tune with `make bench PKG=./kernels/gemm`.
const (
	MR = 8
	NR = 12
	MC = 256
	KC = 512
	NC = 3072
)
