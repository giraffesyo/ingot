package gemm

import (
	"math"
	"sync"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/vek"
)

// bf16 GEMM: bf16 storage, exact f32 products (8-bit mantissas), f32
// accumulation via BFMMLA. The packed layouts are byte-identical to the int8
// MMLA ones with a k-group of 4 bf16 (the same 64-byte A tiles and 96-byte
// group-major B panels), so the pack/scatter machinery is shared.

const bKG = 4 // bf16 k-steps per packed group

// F32ToBF16 rounds one float32 to bfloat16 bits (round-to-nearest-even;
// NaN kept quiet).
func F32ToBF16(f float32) uint16 {
	b := math.Float32bits(f)
	if b&0x7FFFFFFF > 0x7F800000 { // NaN: keep it one
		return uint16(b>>16) | 0x40
	}
	return uint16((b + 0x7FFF + (b>>16)&1) >> 16)
}

// BF16ToF32 widens bfloat16 bits to float32.
func BF16ToF32(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }

// BPackedA is an m×k f32 matrix converted to bf16 and packed for bkernelBF16
// ([mp][kg][qMR rows][bKG] elements). Weights convert once at pack time.
type BPackedA struct {
	m, k int
	data []uint16
}

// HasBFMMLA reports whether the bf16 kernel is available on this CPU.
func HasBFMMLA() bool { return hasBFMMLA }

// HasBF16 reports whether any bf16 GEMM kernel is available (arm64 BFMMLA or
// amd64 VDPBF16PS).
func HasBF16() bool { return hasBFMMLA || hasBF16DP }

// BPackA converts and packs A (row-major m×k f32, leading dim lda).
func BPackA(m, k int, a []float32, lda int) *BPackedA {
	kg := (k + bKG - 1) / bKG
	mp := (m + qMR - 1) / qMR
	ap := make([]uint16, mp*kg*qMR*bKG)
	for ip := 0; ip < mp; ip++ {
		i0 := ip * qMR
		rows := min(qMR, m-i0)
		dst := ap[ip*kg*qMR*bKG:]
		for r := 0; r < rows; r++ {
			row := a[(i0+r)*lda:]
			for p := 0; p < k; p++ {
				dst[(p/bKG)*qMR*bKG+r*bKG+p%bKG] = F32ToBF16(row[p])
			}
		}
	}
	return &BPackedA{m: m, k: k, data: ap}
}

type bwork struct {
	bp []uint16
	ct [qMR * qNR]float32
}

var bworkPool = sync.Pool{New: func() any { return &bwork{} }}

func getBwork(need int) *bwork {
	w := bworkPool.Get().(*bwork)
	// +8: the amd64 kernel's 64-byte loads read 16 bytes past each 48-byte
	// pair block; the slack keeps the final load in-bounds.
	if cap(w.bp) < need+8 {
		w.bp = make([]uint16, need+8)
	}
	w.bp = w.bp[:need]
	return w
}

// BgemmPacked computes C[m×n] = A·B with pre-packed bf16 A and f32 B
// (converted to bf16 per panel), f32 accumulation.
func BgemmPacked(pa *BPackedA, n int, b []float32, ldb int, c []float32, ldc int, parallel bool) {
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
	kg := (k + bKG - 1) / bKG
	mp := (m + qMR - 1) / qMR
	np := (n + qNR - 1) / qNR
	need := kg * qNR * bKG
	serial := !parallel || np == 1 || m*n*k < 4*minTaskMACs
	var bufs []*bwork
	if serial {
		bufs = []*bwork{getBwork(need)}
	} else {
		bufs = make([]*bwork, par.Workers())
	}
	task := func(jp, wk int) {
		wb := bufs[wk]
		if wb == nil {
			wb = getBwork(need)
			bufs[wk] = wb
		}
		bp := wb.bp
		j0 := jp * qNR
		cols := min(qNR, n-j0)
		if cols < qNR || k%bKG != 0 {
			clear(bp)
		}
		if bPairB {
			// Pair-major for the VDPBF16PS kernel: [g][2p][12j][2] bf16.
			for g := 0; g < kg; g++ {
				q0 := g * bKG
				ko := min(bKG, k-q0)
				dst := bp[g*qNR*bKG:]
				for o := 0; o < ko; o++ {
					src := b[(q0+o)*ldb+j0 : (q0+o)*ldb+j0+cols]
					d := dst[(o/2)*2*qNR+o%2:]
					for j, v := range src {
						d[j*2] = F32ToBF16(v)
					}
				}
			}
		} else {
			for g := 0; g < kg; g++ {
				q0 := g * bKG
				ko := min(bKG, k-q0)
				dst := bp[g*qNR*bKG:]
				for o := 0; o < ko; o++ {
					src := b[(q0+o)*ldb+j0 : (q0+o)*ldb+j0+cols]
					d := dst[o:]
					for j, v := range src {
						d[j*bKG] = F32ToBF16(v)
					}
				}
			}
		}
		for ip := 0; ip < mp; ip++ {
			i0 := ip * qMR
			rows := min(qMR, m-i0)
			bkernelBF16(int64(kg), &pa.data[ip*kg*qMR*bKG], &bp[0], &wb.ct[0])
			bscatterTile(wb.ct[:], c[i0*ldc+j0:], ldc, rows, cols)
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
			bworkPool.Put(w)
		}
	}
}

// bscatterTile writes the 2×2-block f32 tile row-major (same layout as the
// int8 kernels' i32 tiles; the NEON qscatter works on either, 4-byte lanes).
func bscatterTile(ct []float32, c []float32, ldc, rows, cols int) {
	if bctRowMajor {
		for r := 0; r < rows; r++ {
			copy(c[r*ldc:r*ldc+cols], ct[r*qNR:r*qNR+cols])
		}
		return
	}
	if hasQpackAsm && rows == qMR && cols == qNR {
		qscatter(f32AsI32(&ct[0]), f32AsI32(&c[0]), int64(ldc))
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

// BConvert converts f32 weights to bf16 bits (round-to-nearest-even) — the
// one-time pack step for bf16 weight storage.
func BConvert(dst []uint16, src []float32) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = F32ToBF16(src[i])
	}
}

// GemvBF16 computes y[n] = W·x for bf16 weights W (n×k row-major bits,
// leading dim ldw) and f32 x — the bandwidth-bound m=1 decode shape, where
// bf16 storage halves the weight traffic that is the entire cost.
func GemvBF16(y []float32, w []uint16, ldw int, x []float32, n, k int) {
	grain := max(1, minTaskMACs/max(k, 1))
	par.For(n, grain, func(j, _ int) {
		y[j] = vek.DotBF16(x[:k], w[j*ldw:j*ldw+k])
	})
}

// BF16Fast reports whether the bf16 kernel is a speedup on this CPU (the
// amd64 VDPBF16PS kernel; arm64 BFMMLA runs at quarter rate on Apple parts
// and is not worth opting into).
func BF16Fast() bool { return hasBF16DP }

// BPackedB is a k×n f32 weight matrix converted to bf16 and packed whole in
// the kernel's B panel layout, for operands reused across calls (Linear /
// MatMul weights). transB packs an [n×k] source (torch Linear, Gemm transB).
type BPackedB struct {
	k, n, kg, np int
	data         []uint16 // [jp][g][qNR*bKG] (+ tail slack for 64B loads)
}

// Rows and Cols return the logical dimensions of the packed matrix.
func (p *BPackedB) Rows() int { return p.k }
func (p *BPackedB) Cols() int { return p.n }

// BPackB converts and packs B[k×n] (row stride ldw; transB: stored [n×k]).
func BPackB(transB bool, k, n int, w []float32, ldw int) *BPackedB {
	kg := (k + bKG - 1) / bKG
	np := (n + qNR - 1) / qNR
	p := &BPackedB{k: k, n: n, kg: kg, np: np, data: make([]uint16, np*kg*qNR*bKG+8)}
	step := bKG
	if bPairB {
		step = 2
	}
	for jp := 0; jp < np; jp++ {
		j0 := jp * qNR
		cols := min(qNR, n-j0)
		panel := p.data[jp*kg*qNR*bKG:]
		for g := 0; g < kg; g++ {
			q0 := g * bKG
			ko := min(bKG, k-q0)
			dst := panel[g*qNR*bKG:]
			for o := 0; o < ko; o++ {
				d := dst[o:]
				if bPairB {
					d = dst[(o/2)*2*qNR+o%2:]
				}
				for j := 0; j < cols; j++ {
					var v float32
					if transB {
						v = w[(j0+j)*ldw+q0+o]
					} else {
						v = w[(q0+o)*ldw+j0+j]
					}
					d[j*step] = F32ToBF16(v)
				}
			}
		}
	}
	return p
}

var bpackAPool = sync.Pool{New: func() any { return new([]uint16) }}

// BgemmWeights computes y[m×n] = x·W against pre-packed bf16 weights. x is
// converted to a row-major bf16 image once per call (pure SIMD conversion —
// no packing scatter; the DP kernel reads A through row pointers), and the
// kernel consumes the stored B panels. y is written row-major (beta = 0).
func BgemmWeights(m int, x []float32, ldx int, pb *BPackedB, y []float32, ldy int, parallel bool) {
	k, n := pb.k, pb.n
	if m == 0 || n == 0 {
		return
	}
	if k == 0 {
		for i := 0; i < m; i++ {
			clear(y[i*ldy : i*ldy+n])
		}
		return
	}
	kg, np := pb.kg, pb.np
	mp := (m + qMR - 1) / qMR
	if !hasBF16DP {
		bgemmWeightsPacked(m, x, ldx, pb, y, ldy, parallel)
		return
	}
	kAl := (kg*bKG + 31) &^ 31
	need := m*kAl + kAl // + one zero row for edge tiles
	xbn := bpackAPool.Get().(*[]uint16)
	if cap(*xbn) < need {
		*xbn = make([]uint16, need)
	}
	xb := (*xbn)[:need]
	zero := xb[m*kAl:]
	clear(zero)
	tail := kg*bKG - k // padded k positions the kernel reads past x
	cvt := func(r int) {
		row := xb[r*kAl:]
		bf16Row(row[:k], x[r*ldx:r*ldx+k])
		if tail > 0 {
			clear(row[k : k+tail])
		}
	}
	serial := !parallel || np == 1 || m*n*k < 4*minTaskMACs
	if serial || m < 2*qMR {
		for r := 0; r < m; r++ {
			cvt(r)
		}
	} else {
		par.For(m, max(1, 16384/max(k, 1)), func(r, _ int) { cvt(r) })
	}
	var bufs []*bwork
	if serial {
		bufs = []*bwork{getBwork(0)}
	} else {
		bufs = make([]*bwork, par.Workers())
	}
	task := func(jp, wk int) {
		wb := bufs[wk]
		if wb == nil {
			wb = getBwork(0)
			bufs[wk] = wb
		}
		j0 := jp * qNR
		cols := min(qNR, n-j0)
		bp := &pb.data[jp*kg*qNR*bKG]
		var rows [qMR]*uint16
		for ip := 0; ip < mp; ip++ {
			i0 := ip * qMR
			rc := min(qMR, m-i0)
			for r := 0; r < qMR; r++ {
				if r < rc {
					rows[r] = &xb[(i0+r)*kAl]
				} else {
					rows[r] = &zero[0]
				}
			}
			bkernelBF16Rows(int64(kg), &rows, bp, &wb.ct[0])
			bscatterTile(wb.ct[:], y[i0*ldy+j0:], ldy, rc, cols)
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
			bworkPool.Put(w)
		}
	}
	bpackAPool.Put(xbn)
}

// bgemmWeightsPacked is BgemmWeights for the packed-A kernels (arm64 BFMMLA):
// x converts into the [g][r][bKG] panel layout per call.
func bgemmWeightsPacked(m int, x []float32, ldx int, pb *BPackedB, y []float32, ldy int, parallel bool) {
	k, n := pb.k, pb.n
	kg, np := pb.kg, pb.np
	mp := (m + qMR - 1) / qMR
	apn := bpackAPool.Get().(*[]uint16)
	if cap(*apn) < mp*kg*qMR*bKG {
		*apn = make([]uint16, mp*kg*qMR*bKG)
	}
	ap := (*apn)[:mp*kg*qMR*bKG]
	packA := func(ip, _ int) {
		i0 := ip * qMR
		rows := min(qMR, m-i0)
		dst := ap[ip*kg*qMR*bKG:]
		if rows < qMR || k%bKG != 0 {
			clear(dst[:kg*qMR*bKG])
		}
		for r := 0; r < rows; r++ {
			row := x[(i0+r)*ldx:]
			for p := 0; p < k; p++ {
				dst[(p/bKG)*qMR*bKG+r*bKG+p%bKG] = F32ToBF16(row[p])
			}
		}
	}
	serial := !parallel || np == 1 || m*n*k < 4*minTaskMACs
	if serial || mp == 1 {
		for ip := 0; ip < mp; ip++ {
			packA(ip, 0)
		}
	} else {
		par.For(mp, max(1, 8192/max(k, 1)), packA)
	}
	var bufs []*bwork
	if serial {
		bufs = []*bwork{getBwork(0)}
	} else {
		bufs = make([]*bwork, par.Workers())
	}
	task := func(jp, wk int) {
		wb := bufs[wk]
		if wb == nil {
			wb = getBwork(0)
			bufs[wk] = wb
		}
		j0 := jp * qNR
		cols := min(qNR, n-j0)
		bp := &pb.data[jp*kg*qNR*bKG]
		for ip := 0; ip < mp; ip++ {
			i0 := ip * qMR
			rows := min(qMR, m-i0)
			bkernelBF16(int64(kg), &ap[ip*kg*qMR*bKG], bp, &wb.ct[0])
			bscatterTile(wb.ct[:], y[i0*ldy+j0:], ldy, rows, cols)
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
			bworkPool.Put(w)
		}
	}
	bpackAPool.Put(apn)
}
