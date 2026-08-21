package gemm

// Blocking parameters for the pure-Go micro-kernel.
// MR×NR is the register tile; MC×KC is the packed A panel (target L2);
// KC×NC is the packed B panel (target L3 / shared).
// Tuned loosely for Apple M-series L1d=128KB/L2=16MB; revisit per arch with asm.
const (
	MR = 4
	NR = 4
	MC = 128
	KC = 256
	NC = 1024
)
