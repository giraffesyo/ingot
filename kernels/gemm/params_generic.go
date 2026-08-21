//go:build !arm64

package gemm

// Blocking parameters for the pure-Go micro-kernel.
// MR×NR is the register tile; MC×KC is the packed A panel (target L2);
// KC×NC is the packed B panel (target L3 / shared).
const (
	MR = 4
	NR = 4
	MC = 128
	KC = 256
	NC = 1024
)
