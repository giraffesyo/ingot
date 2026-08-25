package gemm

import (
	"sync"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/sme"
)

// Quantized GEMM: C_s32 = A_u8 · B_s8 (raw dot products — zero-point
// compensation is the caller's, per the ONNX QLinear* decomposition where
// weights are symmetric and the activation zero-point folds into a
// per-output-channel bias: Σ x·w − zx·Σw).
//
// Micro-kernel: 8×12 s32 tile via USMMLA (FEAT_I8MM), 8 k-steps per group,
// 32 MACs per instruction. K is padded to groups of 8 with zeros on both
// operands (contributing nothing).

const (
	qMR = 8
	qNR = 12
	qKG = 8 // k-steps per packed group
)

// QgemmU8S8 computes C[m×n] (s32, row-major, stride ldc) = A[m×k] (u8, stride
// lda) · B[k×n] (s8, stride ldb).
func QgemmU8S8(m, n, k int, a []uint8, lda int, b []int8, ldb int, c []int32, ldc int) {
	if m == 0 || n == 0 {
		return
	}
	if k == 0 {
		for i := 0; i < m; i++ {
			clear(c[i*ldc : i*ldc+n])
		}
		return
	}
	kg := (k + qKG - 1) / qKG
	mp := (m + qMR - 1) / qMR
	np := (n + qNR - 1) / qNR
	// Pack A: panel ip = kg groups × 8 rows × 4 bytes.
	ap := make([]uint8, mp*kg*qMR*qKG)
	par.For(mp, max(1, 8192/max(k, 1)), func(ip, _ int) {
		i0 := ip * qMR
		rows := min(qMR, m-i0)
		dst := ap[ip*kg*qMR*qKG : (ip+1)*kg*qMR*qKG]
		clear(dst)
		for r := 0; r < rows; r++ {
			src := a[(i0+r)*lda : (i0+r)*lda+k]
			for p, v := range src {
				dst[(p/qKG)*qMR*qKG+r*qKG+p%qKG] = v
			}
		}
	})
	workers := par.Workers()
	if w := int(int64(m) * int64(n) * int64(k) / (4 * minTaskMACs)); w < workers {
		workers = max(w, 1)
	}
	need := kg * qNR * qKG
	serial := workers <= 1 || np == 1
	var bufs []*qwork
	if serial {
		bufs = []*qwork{getQwork(need)}
	} else {
		bufs = make([]*qwork, par.Workers())
	}
	task := func(jp, wk int) {
		wb := bufs[wk]
		if wb == nil {
			wb = getQwork(need)
			bufs[wk] = wb
		}
		bp := wb.bp
		j0 := jp * qNR
		cols := min(qNR, n-j0)
		qpackBPanel(bp, b, ldb, k, kg, j0, cols)
		for ip := 0; ip < mp; ip++ {
			i0 := ip * qMR
			rows := min(qMR, m-i0)
			qkernel(int64(kg), &ap[ip*kg*qMR*qKG], &bp[0], &wb.ct[0])
			qscatterTile(wb.ct[:], c[(i0)*ldc+j0:], ldc, rows, cols)
		}
	}
	if serial {
		for jp := 0; jp < np; jp++ {
			task(jp, 0)
		}
	} else {
		par.For(np, max(1, (4*macroTaskMACs)/max(qNR*m*k, 1)), task)
	}
	for _, w := range bufs {
		if w != nil {
			putQwork(w)
		}
	}
}

// qscatterTile writes the micro-kernel's 8×12 accumulator tile (2×2-block
// layout: reg bi*6+bj, lane l is row 2bi+l/2, col 2bj+l%2) into row-major C.
// Full tiles take the NEON uzp kernel; edge tiles scatter scalar.
func qscatterTile(ct []int32, c []int32, ldc, rows, cols int) {
	if qctRowMajor {
		for r := 0; r < rows; r++ {
			copy(c[r*ldc:r*ldc+cols], ct[r*qNR:r*qNR+cols])
		}
		return
	}
	if hasQpackAsm && rows == qMR && cols == qNR {
		qscatter(&ct[0], &c[0], int64(ldc))
		return
	}
	for r := 0; r < rows; r++ {
		dst := c[r*ldc : r*ldc+cols]
		bi := r >> 1
		lr := (r & 1) << 1
		for j := 0; j < cols; j++ {
			dst[j] = ct[(bi*6+j>>1)*4+lr+j&1]
		}
	}
}

// qpackBPanel packs one qNR-column panel of row-major B (k×n, leading dim ldb)
// into group-major [kg][qNR][qKG] layout: bp[g*qNR*qKG+j*qKG+o] = B[g*qKG+o][j0+j].
// Full panels take the NEON zip-transpose kernel (it loads 16 bytes per row, so
// it needs j0+16 <= ldb and whole k-groups); edges fall back to the scalar loop.
func qpackBPanel(bp []int8, b []int8, ldb, k, kg, j0, cols int) {
	if qpackQuad {
		// Quad-major layout ([g][q(2)][j(12)][4] bytes): amd64 VNNI and the
		// arm64 SDOT tier. On arm64 the zip-transpose kernel handles full
		// panels; scalar keeps the edges (and amd64 entirely, for now).
		if hasQpackAsm && cols == qNR && k%qKG == 0 && j0+16 <= ldb {
			qpackbq(&bp[0], &b[j0], int64(ldb), int64(kg))
			return
		}
		if cols < qNR || k%qKG != 0 {
			clear(bp[:kg*qNR*qKG])
		}
		for g := 0; g < kg; g++ {
			q0 := g * qKG
			ko := min(qKG, k-q0)
			dst := bp[g*qNR*qKG:]
			for o := 0; o < ko; o++ {
				src := b[(q0+o)*ldb+j0 : (q0+o)*ldb+j0+cols]
				d := dst[(o>>2)*qNR*4+o&3:]
				for j, v := range src {
					d[j*4] = v
				}
			}
		}
		return
	}
	if hasQpackAsm && cols == qNR && k%qKG == 0 && j0+16 <= ldb {
		qpackb(&bp[0], &b[j0], int64(ldb), int64(kg))
		return
	}
	if cols < qNR || k%qKG != 0 {
		clear(bp[:kg*qNR*qKG])
	}
	for g := 0; g < kg; g++ {
		q0 := g * qKG
		ko := min(qKG, k-q0)
		dst := bp[g*qNR*qKG:]
		for o := 0; o < ko; o++ {
			src := b[(q0+o)*ldb+j0 : (q0+o)*ldb+j0+cols]
			d := dst[o:]
			for j, v := range src {
				d[j*qKG] = v
			}
		}
	}
}

// QgemmRef is the scalar reference (the oracle in tests).
func QgemmRef(m, n, k int, a []uint8, lda int, b []int8, ldb int, c []int32, ldc int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s int32
			for p := 0; p < k; p++ {
				s += int32(a[i*lda+p]) * int32(b[p*ldb+j])
			}
			c[i*ldc+j] = s
		}
	}
}

// QPackedA is s8 weights W[m×k] pre-packed for the SMMLA kernel.
type QPackedA struct {
	m, k int
	data []int8        // [panel][k-group][8 rows × 8 bytes]
	sme  *sme.QPackedA // additionally packed for the SME kernel (nil unless enabled+eligible)
}

// Rows and Cols return the logical dimensions.
func (p *QPackedA) Rows() int { return p.m }
func (p *QPackedA) Cols() int { return p.k }

// QPackA packs row-major s8 A[m×k] (row stride lda) for QgemmPackedS8.
func QPackA(m, k int, a []int8, lda int) *QPackedA {
	kg := (k + qKG - 1) / qKG
	mp := (m + qMR - 1) / qMR
	p := &QPackedA{m: m, k: k, data: make([]int8, mp*kg*qMR*qKG)}
	if qsmeEligible(m, k) {
		p.sme = qsmePackA(m, k, a, lda)
	}
	for ip := 0; ip < mp; ip++ {
		i0 := ip * qMR
		rows := min(qMR, m-i0)
		dst := p.data[ip*kg*qMR*qKG:]
		for r := 0; r < rows; r++ {
			src := a[(i0+r)*lda : (i0+r)*lda+k]
			for q, v := range src {
				dst[(q/qKG)*qMR*qKG+r*qKG+q%qKG] = v
			}
		}
	}
	return p
}

// qwork holds a worker's packed-B panel and output tile, pooled on a plain
// free list (like gemmCtx) so per-tile serial calls from conv chunk loops
// allocate nothing in steady state.
type qwork struct {
	bp []int8
	ct [qMR * qNR]int32
}

// sync.Pool: per-P caches, so the per-chunk serial calls from conv loops on
// 18 workers don't contend on one mutex (they did). Buffers are small (a few
// KB), so GC clearing them is fine.
var qworkPool = sync.Pool{New: func() any { return &qwork{} }}

func getQwork(need int) *qwork {
	w := qworkPool.Get().(*qwork)
	// +16: the VNNI kernel's 64-byte loads read 16 bytes past each 48-byte
	// B quad; the slack keeps the final load in-bounds.
	if cap(w.bp) < need+16 {
		w.bp = make([]int8, need+16)
	}
	w.bp = w.bp[:need]
	return w
}

func putQwork(w *qwork) { qworkPool.Put(w) }

// QgemmPackedS8 computes C[m×n] (s32) = pa · B[k×n] (s8, stride ldb) — raw
// dot products; zero-point compensation is the caller's.
func QgemmPackedS8(pa *QPackedA, n int, b []int8, ldb int, c []int32, ldc int, parallel bool) {
	m, k := pa.m, pa.k
	if m == 0 || n == 0 {
		return
	}
	if pa.sme != nil && n >= 128 {
		qsmeGemm(pa.sme, n, b, ldb, c, ldc, parallel)
		return
	}
	if k == 0 {
		for i := 0; i < m; i++ {
			clear(c[i*ldc : i*ldc+n])
		}
		return
	}
	kg := (k + qKG - 1) / qKG
	mp := (m + qMR - 1) / qMR
	np := (n + qNR - 1) / qNR
	need := kg * qNR * qKG
	serial := !parallel || np == 1 || m*n*k < 4*minTaskMACs
	var bufs []*qwork
	if serial {
		bufs = []*qwork{getQwork(need)}
	} else {
		bufs = make([]*qwork, par.Workers())
	}
	task := func(jp, wk int) {
		wb := bufs[wk]
		if wb == nil {
			wb = getQwork(need)
			bufs[wk] = wb
		}
		bp := wb.bp
		j0 := jp * qNR
		cols := min(qNR, n-j0)
		qpackBPanel(bp, b, ldb, k, kg, j0, cols)
		for ip := 0; ip < mp; ip++ {
			i0 := ip * qMR
			rows := min(qMR, m-i0)
			qkernelS8(int64(kg), &pa.data[ip*kg*qMR*qKG], &bp[0], &wb.ct[0])
			qscatterTile(wb.ct[:], c[i0*ldc+j0:], ldc, rows, cols)
		}
	}
	if serial {
		for jp := 0; jp < np; jp++ {
			task(jp, 0)
		}
	} else {
		par.For(np, max(1, (4*macroTaskMACs)/max(qNR*m*k, 1)), task)
	}
	for _, w := range bufs {
		if w != nil {
			putQwork(w)
		}
	}
}
