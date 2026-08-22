package gemm

import (
	"sync"

	"github.com/giraffesyo/ingot/kernels/par"
)

// gemmCtx is the per-call state shared by all workers. It doubles as the
// par.Task for the three parallel phases (pack B, pack A, macro-kernel) so a
// call allocates nothing in steady state.
type gemmCtx struct {
	a     []float32   // packed A block: MC*KC (MR-padded)
	b     []float32   // packed B panel: KC*NC (NR-padded)
	tiles [][]float32 // per-worker MR*NR edge tiles

	phase int // phasePackB, phasePackA, phaseMacro

	// current block geometry
	kc, nc, mc       int
	nPanels, mPanels int
	mChunks, chunk   int
	alpha            float32
	acc              bool
	transA, transB   bool
	asrc, bsrc, cblk []float32
	lda, ldb, ldc    int
}

const (
	phasePackB = iota
	phasePackA
	phaseMacro
)

// minTaskMACs is the multiply-accumulate count below which an extra worker is
// not worth its hand-off (~1µs at 100 GFLOPS).
const minTaskMACs = 48 * 1024

var ctxPool = sync.Pool{New: func() any {
	g := &gemmCtx{
		a: make([]float32, (MC+MR)*KC),
		b: make([]float32, KC*(NC+NR)),
	}
	g.tiles = make([][]float32, par.Workers())
	for i := range g.tiles {
		g.tiles[i] = make([]float32, MR*NR)
	}
	return g
}}

// Sgemm computes C = alpha*A·B + beta*C for row-major A[m×k], B[k×n], C[m×n]
// with leading dimensions lda, ldb, ldc.
func Sgemm(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	SgemmT(false, false, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// SgemmT is Sgemm with optional transposed operands: if transA, A is stored
// as [k×m] (lda is its row stride) and op(A)=Aᵀ; likewise transB with B
// stored as [n×k]. Transposition is folded into packing, not materialised.
//
// Structure (Goto/BLIS): for each NC-wide column block and KC-deep k block,
// B is packed once into NR-wide panels; for each MC-tall row block, A is packed
// into MR-wide panels and the macro-kernel sweeps the (MR×NR) tiles. Packing
// and the macro-kernel are parallelised across par's worker pool; the k loop is
// sequential so workers always accumulate into disjoint C tiles.
func SgemmT(transA, transB bool, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if m == 0 || n == 0 {
		return
	}
	if k == 0 || alpha == 0 {
		scaleC(m, n, beta, c, ldc)
		return
	}
	if beta != 0 && beta != 1 {
		scaleC(m, n, beta, c, ldc)
	}
	firstOverwrite := beta == 0

	g := ctxPool.Get().(*gemmCtx)
	defer ctxPool.Put(g)
	g.alpha, g.lda, g.ldb, g.ldc = alpha, lda, ldb, ldc
	g.transA, g.transB = transA, transB
	// Cap fan-out by work: ~minTaskMACs per task keeps hand-off cost amortised.
	workers := par.Workers()
	if w := int(int64(m) * int64(n) * int64(k) / minTaskMACs); w < workers {
		workers = max(w, 1)
	}

	for j0 := 0; j0 < n; j0 += NC {
		g.nc = min(NC, n-j0)
		g.nPanels = (g.nc + NR - 1) / NR
		for p0 := 0; p0 < k; p0 += KC {
			g.kc = min(KC, k-p0)
			g.acc = !(firstOverwrite && p0 == 0)
			if transB {
				g.bsrc = b[j0*ldb+p0:] // Bᵀ stored [n×k]: row j0, col p0
			} else {
				g.bsrc = b[p0*ldb+j0:]
			}
			g.phase = phasePackB
			if workers > 1 {
				par.Run(g.nPanels, max(8, g.nPanels/(2*workers)), g)
			} else {
				for t := 0; t < g.nPanels; t++ {
					g.Run(t, 0)
				}
			}
			for i0 := 0; i0 < m; i0 += MC {
				g.mc = min(MC, m-i0)
				g.mPanels = (g.mc + MR - 1) / MR
				if transA {
					g.asrc = a[p0*lda+i0:] // Aᵀ stored [k×m]: row p0, col i0
				} else {
					g.asrc = a[i0*lda+p0:]
				}
				g.phase = phasePackA
				if workers > 1 {
					par.Run(g.mPanels, max(4, g.mPanels/(2*workers)), g)
				} else {
					for t := 0; t < g.mPanels; t++ {
						g.Run(t, 0)
					}
				}
				// 2D task grid: nPanels × mChunks, sized so there are at least
				// ~2 tasks per worker when the problem allows.
				mChunks := 1
				for g.nPanels*mChunks < 2*workers && mChunks < g.mPanels {
					mChunks++
				}
				g.chunk = (g.mPanels + mChunks - 1) / mChunks
				g.mChunks = (g.mPanels + g.chunk - 1) / g.chunk
				g.cblk = c[i0*ldc+j0:]
				g.phase = phaseMacro
				if workers > 1 {
					par.Run(g.nPanels*g.mChunks, 1, g)
				} else {
					for t := 0; t < g.nPanels*g.mChunks; t++ {
						g.Run(t, 0)
					}
				}
			}
		}
	}
}

// Run implements par.Task; dispatches on the current phase.
func (g *gemmCtx) Run(t, w int) {
	switch g.phase {
	case phasePackB:
		if g.transB {
			packBPanelT(g.kc, min(NR, g.nc-t*NR), g.bsrc[t*NR*g.ldb:], g.ldb, g.b[t*g.kc*NR:])
		} else {
			packBPanel(g.kc, min(NR, g.nc-t*NR), g.bsrc[t*NR:], g.ldb, g.b[t*g.kc*NR:])
		}
	case phasePackA:
		if g.transA {
			packAPanelT(g.kc, min(MR, g.mc-t*MR), g.asrc[t*MR:], g.lda, g.a[t*g.kc*MR:])
		} else {
			packAPanel(g.kc, min(MR, g.mc-t*MR), g.asrc[t*MR*g.lda:], g.lda, g.a[t*g.kc*MR:])
		}
	case phaseMacro:
		g.macroTask(t, w)
	}
}

// macroTask computes all MR×NR tiles for one (B panel, A chunk) pair.
func (g *gemmCtx) macroTask(t, w int) {
	jp := t / g.mChunks
	ipStart := (t % g.mChunks) * g.chunk
	ipEnd := min(ipStart+g.chunk, g.mPanels)
	kc, ldc := g.kc, g.ldc
	nr := min(NR, g.nc-jp*NR)
	bpan := g.b[jp*kc*NR:]
	for ip := ipStart; ip < ipEnd; ip++ {
		mr := min(MR, g.mc-ip*MR)
		apan := g.a[ip*kc*MR:]
		cptr := g.cblk[ip*MR*ldc+jp*NR:]
		if mr == MR && nr == NR && g.alpha == 1 {
			microKernel(kc, apan, bpan, cptr, ldc, g.acc)
			continue
		}
		// Edge tile or alpha != 1: compute into a scratch tile, then scatter.
		tile := g.tiles[w]
		microKernel(kc, apan, bpan, tile, NR, false)
		for r := 0; r < mr; r++ {
			row := cptr[r*ldc : r*ldc+nr]
			tr := tile[r*NR : r*NR+nr]
			if g.acc {
				for q := range row {
					row[q] += g.alpha * tr[q]
				}
			} else {
				for q := range row {
					row[q] = g.alpha * tr[q]
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
