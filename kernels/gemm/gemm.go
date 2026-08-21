package gemm

import (
	"sync"

	"github.com/giraffesyo/ocr/kernels/par"
)

// scratch holds per-worker packing buffers so steady-state calls allocate nothing.
type scratch struct {
	a    []float32 // MC*KC, padded to MR
	b    []float32 // KC*NC, padded to NR
	tile []float32 // MR*NR edge tile
}

var scratchPool = sync.Pool{New: func() any {
	return &scratch{
		a:    make([]float32, (MC+MR)*KC),
		b:    make([]float32, KC*(NC+NR)),
		tile: make([]float32, MR*NR),
	}
}}

// Sgemm computes C = alpha*A·B + beta*C for row-major A[m×k], B[k×n], C[m×n]
// with leading dimensions lda, ldb, ldc.
func Sgemm(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if m == 0 || n == 0 {
		return
	}
	if k == 0 || alpha == 0 {
		scaleC(m, n, beta, c, ldc)
		return
	}
	// Apply beta once up front; the kernel then always accumulates into C
	// for KC-blocks after the first. For the first KC block we overwrite when
	// beta==0 to avoid reading uninitialised C.
	if beta != 0 && beta != 1 {
		scaleC(m, n, beta, c, ldc)
	}
	firstOverwrite := beta == 0

	// Parallelize over NC-wide column blocks; each worker packs its own B panel.
	// For small n, fall through to a single block and parallelize over M instead.
	nBlocks := (n + NC - 1) / NC
	if nBlocks >= par.MaxWorkers || m < MC {
		par.For(nBlocks, 1, func(jb int) {
			j0 := jb * NC
			nc := min(NC, n-j0)
			gemmColBlock(m, nc, k, alpha, a, lda, b[j0:], ldb, c[j0:], ldc, firstOverwrite)
		})
		return
	}
	// Parallelize over MC-tall row blocks inside each column block. B panel is
	// packed once per (jb, pb) and shared read-only by all workers.
	for jb := 0; jb < nBlocks; jb++ {
		j0 := jb * NC
		nc := min(NC, n-j0)
		for p0 := 0; p0 < k; p0 += KC {
			kc := min(KC, k-p0)
			sb := scratchPool.Get().(*scratch)
			packB(kc, nc, b[p0*ldb+j0:], ldb, sb.b)
			mBlocks := (m + MC - 1) / MC
			acc := !(firstOverwrite && p0 == 0)
			par.For(mBlocks, 1, func(ib int) {
				i0 := ib * MC
				mc := min(MC, m-i0)
				sa := scratchPool.Get().(*scratch)
				packA(mc, kc, a[i0*lda+p0:], lda, sa.a)
				macroKernel(mc, nc, kc, alpha, sa.a, sb.b, c[i0*ldc+j0:], ldc, acc, sa.tile)
				scratchPool.Put(sa)
			})
			scratchPool.Put(sb)
		}
	}
}

// gemmColBlock handles one column block of width nc, single-threaded.
func gemmColBlock(m, nc, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, firstOverwrite bool) {
	s := scratchPool.Get().(*scratch)
	defer scratchPool.Put(s)
	for p0 := 0; p0 < k; p0 += KC {
		kc := min(KC, k-p0)
		packB(kc, nc, b[p0*ldb:], ldb, s.b)
		acc := !(firstOverwrite && p0 == 0)
		for i0 := 0; i0 < m; i0 += MC {
			mc := min(MC, m-i0)
			packA(mc, kc, a[i0*lda+p0:], lda, s.a)
			macroKernel(mc, nc, kc, alpha, s.a, s.b, c[i0*ldc:], ldc, acc, s.tile)
		}
	}
}

// macroKernel multiplies a packed mc×kc A panel by a packed kc×nc B panel into C.
func macroKernel(mc, nc, kc int, alpha float32, ap, bp []float32, c []float32, ldc int, accumulate bool, tile []float32) {
	for j := 0; j < nc; j += NR {
		nr := min(NR, nc-j)
		bpan := bp[(j/NR)*kc*NR:]
		for i := 0; i < mc; i += MR {
			mr := min(MR, mc-i)
			apan := ap[(i/MR)*kc*MR:]
			if mr == MR && nr == NR && alpha == 1 {
				microKernel(kc, apan, bpan, c[i*ldc+j:], ldc, accumulate)
				continue
			}
			// Edge tile or alpha != 1: compute into tile, then scatter.
			microKernel(kc, apan, bpan, tile, NR, false)
			for r := 0; r < mr; r++ {
				row := c[(i+r)*ldc+j : (i+r)*ldc+j+nr]
				t := tile[r*NR : r*NR+nr]
				if accumulate {
					for q := range row {
						row[q] += alpha * t[q]
					}
				} else {
					for q := range row {
						row[q] = alpha * t[q]
					}
				}
			}
		}
	}
}

func scaleC(m, n int, beta float32, c []float32, ldc int) {
	for i := 0; i < m; i++ {
		row := c[i*ldc : i*ldc+n]
		if beta == 0 {
			clear(row)
		} else {
			for j := range row {
				row[j] *= beta
			}
		}
	}
}
