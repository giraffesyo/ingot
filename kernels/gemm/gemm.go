package gemm

import (
	"sync"

	"github.com/giraffesyo/ocr/kernels/par"
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
	asrc, bsrc, cblk []float32
	lda, ldb, ldc    int
}

const (
	phasePackB = iota
	phasePackA
	phaseMacro
)

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
//
// Structure (Goto/BLIS): for each NC-wide column block and KC-deep k block,
// B is packed once into NR-wide panels; for each MC-tall row block, A is packed
// into MR-wide panels and the macro-kernel sweeps the (MR×NR) tiles. Packing
// and the macro-kernel are parallelised across par's worker pool; the k loop is
// sequential so workers always accumulate into disjoint C tiles.
func Sgemm(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
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
	workers := par.Workers()

	for j0 := 0; j0 < n; j0 += NC {
		g.nc = min(NC, n-j0)
		g.nPanels = (g.nc + NR - 1) / NR
		for p0 := 0; p0 < k; p0 += KC {
			g.kc = min(KC, k-p0)
			g.acc = !(firstOverwrite && p0 == 0)
			g.bsrc = b[p0*ldb+j0:]
			g.phase = phasePackB
			par.Run(g.nPanels, 8, g)
			for i0 := 0; i0 < m; i0 += MC {
				g.mc = min(MC, m-i0)
				g.mPanels = (g.mc + MR - 1) / MR
				g.asrc = a[i0*lda+p0:]
				g.phase = phasePackA
				par.Run(g.mPanels, 4, g)
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
				par.Run(g.nPanels*g.mChunks, 1, g)
			}
		}
	}
}

// Run implements par.Task; dispatches on the current phase.
func (g *gemmCtx) Run(t, w int) {
	switch g.phase {
	case phasePackB:
		packBPanel(g.kc, min(NR, g.nc-t*NR), g.bsrc[t*NR:], g.ldb, g.b[t*g.kc*NR:])
	case phasePackA:
		packAPanel(g.kc, min(MR, g.mc-t*MR), g.asrc[t*MR*g.lda:], g.lda, g.a[t*g.kc*MR:])
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
