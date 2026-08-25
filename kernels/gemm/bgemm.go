package gemm

import (
	"math"
	"sync"

	"github.com/giraffesyo/ingot/kernels/par"
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
	if cap(w.bp) < need {
		w.bp = make([]uint16, need)
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
