//go:build amd64

package gemm

// Blocking parameters for the AVX2 6×16 micro-kernel (ukernel_amd64.s).
// MC×KC A panel and KC×NC B panel target L2/L3. Tune on a real x86 box with
// `make bench PKG=./kernels/gemm`; these are conservative starting values.
const (
	MR = 6
	NR = 16
	MC = 144
	KC = 384
	NC = 3072
)
