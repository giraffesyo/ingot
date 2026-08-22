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
	bpans [][]float32 // per-worker KC*NR B panels (small-M path)
	asm   []float32   // small-M path: all of A packed, [kBlock][mPanel][KC*MR]

	phase int // phasePackB, phasePackA, phaseMacro, phaseSmallPackA, phaseSmallM

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
	phaseSmallPackA
	phaseSmallM
)

// minTaskMACs is the multiply-accumulate count below which an extra worker is
// not worth its hand-off (~1µs at 100 GFLOPS).
const minTaskMACs = 48 * 1024

// macroTaskMACs is the target work per macro-kernel task (~2µs): tasks are
// grouped until they carry at least this much so the atomic hand-off and
// per-task bookkeeping stay amortised even for tiny-K GEMMs.
const macroTaskMACs = 192 * 1024

// smallMMaxA bounds the packed-A footprint (floats) of the small-M path.
const smallMMaxA = 1 << 18

var ctxPool = sync.Pool{New: func() any {
	g := &gemmCtx{
		a: make([]float32, (MC+MR)*KC),
		b: make([]float32, KC*(NC+NR)),
	}
	g.tiles = make([][]float32, par.Workers())
	g.bpans = make([][]float32, par.Workers())
	for i := range g.tiles {
		g.tiles[i] = make([]float32, MR*NR)
		g.bpans[i] = make([]float32, KC*NR)
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
	// Cap fan-out by work: ~minTaskMACs per task keeps hand-off cost amortised.
	workers := par.Workers()
	if w := int(int64(m) * int64(n) * int64(k) / minTaskMACs); w < workers {
		workers = max(w, 1)
	}
	sgemmT(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, workers)
}

// SgemmSerial is Sgemm restricted to the calling goroutine (no worker fan-out).
// Use it from inside an already-parallel region, e.g. a per-tile conv task.
func SgemmSerial(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	sgemmT(false, false, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 1)
}

func sgemmT(transA, transB bool, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int, workers int) {
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

	// Small-M path: A fits comfortably in cache, so pack it once for all of k
	// and sweep the N panels in a single parallel region where each worker
	// packs its own B panel (L1-resident) right before using it. This avoids
	// the NC-block loop and its three hand-offs per block, which dominate
	// small-K / small-M shapes (pointwise convs, GEMV-like heads).
	if mPanels := (m + MR - 1) / MR; m <= MC && mPanels*MR*((k+KC-1)/KC)*KC <= smallMMaxA {
		g.smallM(m, n, k, a, b, c, firstOverwrite, workers)
		return
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
				par.Run(g.nPanels, max(8, g.nPanels/(2*workers), 8192/(g.kc*NR)), g)
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
					grain := max(1, macroTaskMACs/(NR*g.chunk*MR*g.kc))
					par.Run(g.nPanels*g.mChunks, grain, g)
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
	case phaseSmallPackA:
		g.smallPackA(t)
	case phaseSmallM:
		g.smallMTask(t, w)
	}
}

// smallM: see sgemmT. Packed A layout: block (kb, ip) at (kb*mPanels+ip)*KC*MR.
func (g *gemmCtx) smallM(m, n, k int, a, b, c []float32, firstOverwrite bool, workers int) {
	g.mc, g.nc, g.kc = m, n, k
	g.mPanels = (m + MR - 1) / MR
	g.nPanels = (n + NR - 1) / NR
	nkb := (k + KC - 1) / KC
	g.acc = !firstOverwrite
	g.asrc, g.bsrc, g.cblk = a, b, c
	if need := nkb * g.mPanels * KC * MR; cap(g.asm) < need {
		g.asm = make([]float32, need)
	}
	packTasks := nkb * g.mPanels
	g.phase = phaseSmallPackA
	if workers > 1 && packTasks > 1 {
		par.Run(packTasks, max(1, 4096/(KC*MR)), g)
	} else {
		for t := 0; t < packTasks; t++ {
			g.smallPackA(t)
		}
	}
	g.phase = phaseSmallM
	if workers > 1 {
		grain := max(1, macroTaskMACs/(NR*m*k))
		par.Run(g.nPanels, grain, g)
	} else {
		for t := 0; t < g.nPanels; t++ {
			g.smallMTask(t, 0)
		}
	}
}

func (g *gemmCtx) smallPackA(t int) {
	kb, ip := t/g.mPanels, t%g.mPanels
	p0 := kb * KC
	kc := min(KC, g.kc-p0)
	mr := min(MR, g.mc-ip*MR)
	dst := g.asm[(kb*g.mPanels+ip)*KC*MR:]
	if g.transA {
		packAPanelT(kc, mr, g.asrc[p0*g.lda+ip*MR:], g.lda, dst)
	} else {
		packAPanel(kc, mr, g.asrc[ip*MR*g.lda+p0:], g.lda, dst)
	}
}

// smallMTask computes C[:, panel t] over the whole k range: for each k block
// it packs the B panel into this worker's buffer and sweeps the A panels.
func (g *gemmCtx) smallMTask(t, w int) {
	jp := t
	nr := min(NR, g.nc-jp*NR)
	bp := g.bpans[w]
	ldc := g.ldc
	nkb := (g.kc + KC - 1) / KC
	for kb := 0; kb < nkb; kb++ {
		p0 := kb * KC
		kc := min(KC, g.kc-p0)
		if g.transB {
			packBPanelT(kc, nr, g.bsrc[jp*NR*g.ldb+p0:], g.ldb, bp)
		} else {
			packBPanel(kc, nr, g.bsrc[p0*g.ldb+jp*NR:], g.ldb, bp)
		}
		acc := g.acc || kb > 0
		for ip := 0; ip < g.mPanels; ip++ {
			mr := min(MR, g.mc-ip*MR)
			apan := g.asm[(kb*g.mPanels+ip)*KC*MR:]
			cptr := g.cblk[ip*MR*ldc+jp*NR:]
			if mr == MR && nr == NR && g.alpha == 1 {
				microKernel(kc, apan, bp, cptr, ldc, acc)
				continue
			}
			tile := g.tiles[w]
			microKernel(kc, apan, bp, tile, NR, false)
			for r := 0; r < mr; r++ {
				row := cptr[r*ldc : r*ldc+nr]
				tr := tile[r*NR : r*NR+nr]
				if acc {
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
