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

// PackedA is A[m×k] packed into 32-row panels for the ZA kernel (zero-padded),
// for operands reused across calls (weights): the pack cost is paid once.
type PackedA struct {
	m, k int
	data []float32 // [panel][k][32]
}

// Rows and Cols return the logical dimensions of the packed matrix.
func (p *PackedA) Rows() int { return p.m }
func (p *PackedA) Cols() int { return p.k }

// PackA packs row-major A[m×k] (row stride lda) for the ZA kernel.
func PackA(m, k int, a []float32, lda int) *PackedA {
	mp := (m + mr - 1) / mr
	pa := &PackedA{m: m, k: k, data: make([]float32, mp*k*mr)}
	for ip := 0; ip < mp; ip++ {
		i0 := ip * mr
		rows := min(mr, m-i0)
		dst := pa.data[ip*k*mr : (ip+1)*k*mr]
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
	}
	return pa
}

// Sgemm computes C = A·B for row-major A[m×k], B[k×n], C[m×n] (alpha=1,
// beta=0) on the SME unit. K is streamed — no KC blocking: the ZA tile is the
// accumulator for the entire k range, so A and B are read exactly once.
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
	SgemmPacked(PackA(m, k, a, lda), n, b, ldb, c, ldc, true)
}

// SgemmPacked computes C = op(A)·B with a pre-packed A (alpha=1, beta=0).
// Each task packs one 32-column B panel and sweeps the A panels; parallel
// false keeps everything on the calling goroutine.
func SgemmPacked(pa *PackedA, n int, b []float32, ldb int, c []float32, ldc int, parallel bool) {
	m, k := pa.m, pa.k
	if m == 0 || n == 0 {
		return
	}
	if k == 0 {
		for i := 0; i < m; i++ {
			clear(c[i*ldc : i*ldc+n])
		}
		return
	}
	ap := pa.data
	mp := (m + mr - 1) / mr
	np := (n + nr - 1) / nr
	nw := 1
	if parallel {
		nw = par.Workers()
	}
	slab := make([]float32, nw*(k*nr+mr*nr))
	task := func(jp, wk int) {
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
	}
	if !parallel || np == 1 {
		guard(func() {
			for jp := 0; jp < np; jp++ {
				task(jp, 0)
			}
		})
		return
	}
	par.For(np, 1, func(jp, wk int) {
		guard(func() { task(jp, wk) })
	})
}
