//go:build arm64

package sme

import (
	"github.com/giraffesyo/ingot/kernels/par"
)

// Kernel geometry: 4 f32 ZA tiles arranged 2×2 give a 32×32 C block per pass
// (SVL = 512 bits → 16 f32 lanes per tile edge). Available() checks SVL.
const (
	mr = 32
	nr = 32
)

// Sgemm computes C = A·B for row-major A[m×k], B[k×n], C[m×n] (alpha=1,
// beta=0) on the SME unit. K is streamed — no KC blocking: the ZA tile is the
// accumulator for the entire k range, so A and B are read exactly once.
// A is packed into 32-row panels up front; each parallel task packs one
// 32-column B panel and sweeps the A panels.
func Sgemm(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int) {
	if m == 0 || n == 0 {
		return
	}
	if k == 0 {
		for i := 0; i < m; i++ {
			clear(c[i*ldc : i*ldc+n])
		}
		return
	}
	mp := (m + mr - 1) / mr
	np := (n + nr - 1) / nr
	// Pack A: panel ip holds k steps of 32 row-values (zero-padded).
	ap := make([]float32, mp*k*mr)
	par.For(mp, max(1, 4*1024/max(k, 1)), func(ip, _ int) {
		i0 := ip * mr
		rows := min(mr, m-i0)
		dst := ap[ip*k*mr : (ip+1)*k*mr]
		if rows < mr {
			clear(dst)
		}
		// Stream each source row (sequential reads, 128-byte-strided writes).
		for r := 0; r < rows; r++ {
			src := a[(i0+r)*lda : (i0+r)*lda+k]
			for p, v := range src {
				dst[p*mr+r] = v
			}
		}
	})
	// Per-worker: B panel (k×32) + scratch C tile for edges.
	nw := par.Workers()
	slab := make([]float32, nw*(k*nr+mr*nr))
	par.For(np, 1, func(jp, wk int) {
		buf := slab[wk*(k*nr+mr*nr):]
		bp := buf[:k*nr]
		tile := buf[k*nr : k*nr+mr*nr]
		j0 := jp * nr
		cols := min(nr, n-j0)
		if cols < nr {
			clear(bp)
		}
		for p := 0; p < k; p++ {
			copy(bp[p*nr:p*nr+cols], b[p*ldb+j0:p*ldb+j0+cols])
		}
		for ip := 0; ip < mp; ip++ {
			i0 := ip * mr
			rows := min(mr, m-i0)
			if rows == mr && cols == nr {
				zakernel(int64(k), &ap[ip*k*mr], &bp[0], &c[i0*ldc+j0], int64(ldc)*4)
				continue
			}
			zakernel(int64(k), &ap[ip*k*mr], &bp[0], &tile[0], nr*4)
			for r := 0; r < rows; r++ {
				copy(c[(i0+r)*ldc+j0:(i0+r)*ldc+j0+cols], tile[r*nr:r*nr+cols])
			}
		}
	})
}
