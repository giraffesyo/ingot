package sme

import (
	"github.com/giraffesyo/ingot/kernels/par"
)

// int8 on the matrix unit: SMOPA accumulates a widening 4-way s8 outer
// product into the s32 ZA tiles — 1024 MACs per instruction, 4× the f32
// FMOPA density. Same 32×32 C-block geometry and guard discipline as Sgemm.

const qkg = 4 // k-steps per packed group

// QPackedA is s8 A[m×k] packed for qzakernel: per 32-row panel, per 4-k
// group, two halves of 16 rows × 4 bytes (zero-padded).
type QPackedA struct {
	m, k int
	data []int8
}

// Rows and Cols return the logical dimensions.
func (p *QPackedA) Rows() int { return p.m }
func (p *QPackedA) Cols() int { return p.k }

// QPackA packs row-major s8 A[m×k] (row stride lda).
func QPackA(m, k int, a []int8, lda int) *QPackedA {
	kg := (k + qkg - 1) / qkg
	mp := (m + mr - 1) / mr
	p := &QPackedA{m: m, k: k, data: make([]int8, mp*kg*mr*qkg)}
	for ip := 0; ip < mp; ip++ {
		i0 := ip * mr
		rows := min(mr, m-i0)
		dst := p.data[ip*kg*mr*qkg:]
		for r := 0; r < rows; r++ {
			src := a[(i0+r)*lda : (i0+r)*lda+k]
			half, lane := r/16, r%16
			for q, v := range src {
				g, o := q/qkg, q%qkg
				dst[g*mr*qkg+half*16*qkg+lane*qkg+o] = v
			}
		}
	}
	return p
}

// QgemmPackedS8 computes C[m×n] (s32, row-major) = pa · B[k×n] (s8, stride
// ldb) — raw dot products, zero-point compensation is the caller's. Each task
// packs one 32-column B panel and sweeps the A panels on the SME unit.
func QgemmPackedS8(pa *QPackedA, n int, b []int8, ldb int, c []int32, ldc int, parallel bool) {
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
	kg := (k + qkg - 1) / qkg
	mp := (m + mr - 1) / mr
	np := (n + nr - 1) / nr
	nw := 1
	if parallel {
		nw = par.Workers()
	}
	slabB := make([]int8, nw*kg*nr*qkg)
	slabT := make([]int32, nw*mr*nr)
	task := func(jp, wk int) {
		bp := slabB[wk*kg*nr*qkg : (wk+1)*kg*nr*qkg]
		tile := slabT[wk*mr*nr : (wk+1)*mr*nr]
		j0 := jp * nr
		cols := min(nr, n-j0)
		if cols < nr || k%qkg != 0 {
			clear(bp)
		}
		for g := 0; g < kg; g++ {
			q0 := g * qkg
			ko := min(qkg, k-q0)
			dst := bp[g*nr*qkg:]
			for o := 0; o < ko; o++ {
				src := b[(q0+o)*ldb+j0 : (q0+o)*ldb+j0+cols]
				// halves of 16 columns; contiguous reads, stride-4 writes
				h0 := min(16, cols)
				d := dst[o:]
				for j := 0; j < h0; j++ {
					d[j*qkg] = src[j]
				}
				d = dst[16*qkg+o:]
				for j := 16; j < cols; j++ {
					d[(j-16)*qkg] = src[j]
				}
			}
		}
		guard(func() {
			for ip := 0; ip < mp; ip++ {
				i0 := ip * mr
				rows := min(mr, m-i0)
				if rows == mr && cols == nr {
					qzakernel(int64(kg), &ap[ip*kg*mr*qkg], &bp[0], &c[i0*ldc+j0], int64(ldc)*4)
					continue
				}
				qzakernel(int64(kg), &ap[ip*kg*mr*qkg], &bp[0], &tile[0], nr*4)
				for r := 0; r < rows; r++ {
					copy(c[(i0+r)*ldc+j0:(i0+r)*ldc+j0+cols], tile[r*nr:r*nr+cols])
				}
			}
		})
	}
	if !parallel || np == 1 {
		for jp := 0; jp < np; jp++ {
			task(jp, 0)
		}
		return
	}
	par.For(np, 1, task)
}
