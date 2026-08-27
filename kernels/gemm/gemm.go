package gemm

import (
	"sync"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/sme"
	"github.com/giraffesyo/ingot/kernels/vek"
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

// ctxPool is a free list of gemmCtx. It is a plain stack (not sync.Pool) so
// the contexts — several MB each — survive GC and steady-state calls allocate
// nothing; its size is bounded by the peak number of concurrent GEMMs.
var ctxPool = struct {
	mu   sync.Mutex
	free []*gemmCtx
}{}

func getCtx() *gemmCtx {
	ctxPool.mu.Lock()
	n := len(ctxPool.free)
	if n > 0 {
		g := ctxPool.free[n-1]
		ctxPool.free = ctxPool.free[:n-1]
		ctxPool.mu.Unlock()
		return g
	}
	ctxPool.mu.Unlock()
	g := &gemmCtx{}
	g.tiles = make([][]float32, par.Workers())
	g.bpans = make([][]float32, par.Workers())
	for i := range g.tiles {
		g.tiles[i] = make([]float32, MR*NR)
	}
	return g
}

func putCtx(g *gemmCtx) {
	ctxPool.mu.Lock()
	ctxPool.free = append(ctxPool.free, g)
	ctxPool.mu.Unlock()
}

// ensureBlocked allocates the packed A block / B panel buffers used by the
// general (blocked) path on first use of this context.
func (g *gemmCtx) ensureBlocked() {
	if g.a == nil {
		g.a = make([]float32, (MC+MR)*KC)
		g.b = make([]float32, KC*(NC+NR))
	}
}

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

// SgemmTSerial is SgemmT restricted to the calling goroutine.
func SgemmTSerial(transA, transB bool, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	sgemmT(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 1)
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

	if m == 1 {
		gemv(transA, transB, n, k, alpha, a, lda, b, ldb, firstOverwrite, c, workers)
		return
	}

	if !transA && !transB && alpha == 1 && firstOverwrite && smeEligible(m, n, k) {
		smeSgemm(m, n, k, a, lda, b, ldb, c, ldc, workers > 1)
		return
	}

	g := getCtx()
	defer putCtx(g)
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
	g.ensureBlocked()

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
	g.smallMSweep(m, k, workers)
}

// smallMSweep runs the N-panel sweep of the small-M path over g.asm.
func (g *gemmCtx) smallMSweep(m, k, workers int) {
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
	if bp == nil {
		bp = make([]float32, KC*NR)
		g.bpans[w] = bp
	}
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

// PackedA is A[m×k] pre-packed into the small-M path's panel layout, for
// operands that are reused across many calls (weights). Build once with PackA
// and multiply with SgemmPackedA; the pack cost (a full pass over A, run every
// call otherwise) is paid once.
type PackedA struct {
	m, k int
	data []float32    // [kBlock][mPanel][KC*MR]
	sme  *sme.PackedA // additionally packed for the SME kernel (nil unless enabled+eligible)
}

// Rows and Cols return the logical dimensions of the packed matrix.
func (p *PackedA) Rows() int { return p.m }
func (p *PackedA) Cols() int { return p.k }

// PackFits reports whether an m×k operand is small enough for the pre-packed
// small-M sweep path (SgemmPackedA): the packed A must stay cache-resident.
func PackFits(m, k int) bool {
	mPanels := (m + MR - 1) / MR
	return mPanels*MR*((k+KC-1)/KC)*KC <= smallMMaxA
}

// PackA packs row-major A[m×k] (row stride lda); if transA, A is stored as
// [k×m] and op(A)=Aᵀ is packed.
func PackA(transA bool, m, k int, a []float32, lda int) *PackedA {
	mPanels := (m + MR - 1) / MR
	nkb := (k + KC - 1) / KC
	p := &PackedA{m: m, k: k, data: make([]float32, nkb*mPanels*KC*MR)}
	if !transA && smeEligible(m, 1<<30, k) {
		p.sme = smePackA(m, k, a, lda)
	}
	for kb := 0; kb < nkb; kb++ {
		p0 := kb * KC
		kc := min(KC, k-p0)
		for ip := 0; ip < mPanels; ip++ {
			mr := min(MR, m-ip*MR)
			dst := p.data[(kb*mPanels+ip)*KC*MR:]
			if transA {
				packAPanelT(kc, mr, a[p0*lda+ip*MR:], lda, dst)
			} else {
				packAPanel(kc, mr, a[ip*MR*lda+p0:], lda, dst)
			}
		}
	}
	return p
}

// SgemmPackedA computes C = op(A)·B + beta·C with a pre-packed A (see PackA),
// B[k×n] row-major with stride ldb. workers ≤ 1 keeps the work on the calling
// goroutine (like SgemmSerial); otherwise the N panels are swept in parallel.
func SgemmPackedA(pa *PackedA, n int, b []float32, ldb int, beta float32, c []float32, ldc int, parallel bool) {
	m, k := pa.m, pa.k
	if m == 0 || n == 0 {
		return
	}
	if pa.sme != nil && beta == 0 && n >= 16 {
		smeSgemmPacked(pa.sme, n, b, ldb, c, ldc, parallel)
		return
	}
	if k == 0 {
		scaleC(m, n, beta, c, ldc)
		return
	}
	if beta != 0 && beta != 1 {
		scaleC(m, n, beta, c, ldc)
	}
	workers := 1
	if parallel {
		workers = par.Workers()
		if w := int(int64(m) * int64(n) * int64(k) / minTaskMACs); w < workers {
			workers = max(w, 1)
		}
	}
	g := getCtx()
	defer putCtx(g)
	g.alpha, g.ldb, g.ldc = 1, ldb, ldc
	g.transA, g.transB = false, false
	g.mc, g.nc, g.kc = m, n, k
	g.mPanels = (m + MR - 1) / MR
	g.nPanels = (n + NR - 1) / NR
	g.acc = beta != 0
	g.bsrc, g.cblk = b, c
	saved := g.asm
	g.asm = pa.data
	g.smallMSweep(m, k, workers)
	g.asm = saved
}

// gemv handles m == 1: y[1×n] = alpha·x[1×k]·op(B) (+ y). Packing B into
// micro-kernel panels would cost a full pass over B for one row of output, so
// B is streamed directly: row-major B[k×n] as k axpys into column chunks of y
// (parallel over chunks); transposed B[n×k] as one dot product per output
// (parallel over rows). Memory-bound either way, which is the best a GEMV can do.
// gemvTask carries gemv's parallel state as a pointer Task (par.Run allocates
// nothing for pointer tasks; a closure would allocate per call).
type gemvTask struct {
	a, b, c        []float32
	ldb, k, xs     int
	n, chunk       int
	nChunks        int
	alpha          float32
	firstOverwrite bool
	dot            bool // transB: one dot product per output element
}

var gemvTaskPool = sync.Pool{New: func() any { return new(gemvTask) }}

func (g *gemvTask) Run(i, _ int) {
	if g.dot {
		row := g.b[i*g.ldb : i*g.ldb+g.k]
		var v float32
		if g.xs == 1 {
			v = vek.Dot(g.a[:g.k], row)
		} else {
			for p := 0; p < g.k; p++ {
				v += g.a[p*g.xs] * row[p]
			}
		}
		v *= g.alpha
		if g.firstOverwrite {
			g.c[i] = v
		} else {
			g.c[i] += v
		}
		return
	}
	j0 := i * g.chunk
	j1 := g.n
	if g.nChunks > 1 {
		j1 = min(j0+g.chunk, g.n)
	}
	y := g.c[j0:j1]
	if g.firstOverwrite {
		clear(y)
	}
	for p := 0; p < g.k; p++ {
		xv := g.alpha * g.a[p*g.xs]
		if xv == 0 {
			continue
		}
		vek.Axpy(y, g.b[p*g.ldb+j0:p*g.ldb+j1], xv)
	}
}

func gemv(transA, transB bool, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, firstOverwrite bool, c []float32, workers int) {
	xs := 1
	if transA {
		xs = lda // x stored as a column
	}
	g := gemvTaskPool.Get().(*gemvTask)
	*g = gemvTask{a: a, b: b, c: c, ldb: ldb, k: k, xs: xs, n: n,
		alpha: alpha, firstOverwrite: firstOverwrite}
	if transB {
		// y[j] = alpha * dot(x, B[j,:])
		g.dot = true
		par.Run(n, max(1, minTaskMACs/max(k, 1)), g)
	} else {
		// y[j0:j1] = alpha * Σ_p x[p] * B[p][j0:j1]. Chunks of y stay in L1
		// across the k loop; enough chunks to spread B's bandwidth over the
		// workers.
		chunk := 2048
		if workers > 1 {
			chunk = min(chunk, max(256, n/(2*workers)))
		}
		g.nChunks = (n + chunk - 1) / chunk
		if workers <= 1 {
			g.nChunks = 1
			chunk = n
		}
		g.chunk = chunk
		par.Run(g.nChunks, 1, g)
	}
	*g = gemvTask{}
	gemvTaskPool.Put(g)
}

// PanelKernel exposes the micro-kernel for fused consumers (Winograd conv):
// C[MR×NR] (+)= Aᵖ·Bᵖ for one packed A panel (kc steps × MR rows, packAPanel
// layout) against one packed B panel (kc steps × NR columns). C row stride ldc
// must accommodate MR rows and NR columns — callers own full tiles.
func PanelKernel(kc int, apanel, bpanel, c []float32, ldc int, acc bool) {
	microKernel(kc, apanel, bpanel, c, ldc, acc)
}

// PackAPanels packs row-major A[m×k] (k ≤ KC) into packAPanel layout —
// ceil(m/MR) panels of KC·MR floats — for use with PanelKernel.
func PackAPanels(m, k int, a []float32, lda int) []float32 {
	if k > KC {
		panic("gemm: PackAPanels requires k <= KC")
	}
	mPanels := (m + MR - 1) / MR
	out := make([]float32, mPanels*KC*MR)
	for ip := 0; ip < mPanels; ip++ {
		packAPanel(k, min(MR, m-ip*MR), a[ip*MR*lda:], lda, out[ip*KC*MR:])
	}
	return out
}
