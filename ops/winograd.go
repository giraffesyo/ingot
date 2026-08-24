package ops

import (
	"os"
	"sync"

	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/vek"
	"github.com/giraffesyo/ingot/tensor"
)

// Winograd F(2×2, 3×3): each 4×4 input tile (overlapping, stride 2) yields a
// 2×2 output tile with 16 multiplies per channel pair instead of 36 — 2.25×
// fewer MACs than im2col, and no column matrix.
//
//	V = Bᵀ·d·B   (input tile transform, 4×4)
//	U = G·g·Gᵀ   (weight transform, once per op)
//	M[k] = U[k]·V[k]  for k = 0..15  (16 GEMMs [Cout×Cin]·[Cin×tiles])
//	Y = Aᵀ·M·A   (+ bias, + fused epilogue)
//
// Constants (exact in binary, so no extra rounding vs im2col):
//
//	Bᵀ = ⎡1  0 -1  0⎤   G = ⎡ 1    0    0 ⎤   Aᵀ = ⎡1 1  1  0⎤
//	     ⎢0  1  1  0⎥       ⎢0.5  0.5  0.5⎥        ⎣0 1 -1 -1⎦
//	     ⎢0 -1  1  0⎥       ⎢0.5 -0.5  0.5⎥
//	     ⎣0  1  0 -1⎦       ⎣ 0    0    1 ⎦
//
// Work is split into bands of tile rows: a task transforms its band's input
// tiles for all Cin, runs the 16 GEMMs against pre-packed U (cached on the op)
// and inverse-transforms into the rows of the output plane it exclusively
// owns, applying bias and the fused epilogue while the band is cache-resident.

// winogradEnabled: on by default since the fused per-block rewrite (the first
// version materialised V/M per band, ~600 KB × workers, and lost in-model to
// shared-cache eviction; the fused pipeline's working set is one ~250 KB
// block). OCR_NO_WINOGRAD=1 disables. Eligible convs additionally yield to
// the SME unit when the dispatch policy would take the equivalent im2col GEMM
// (measured 1T order: SME > Winograd > im2col).
var winogradEnabled = os.Getenv("OCR_NO_WINOGRAD") == "" || os.Getenv("OCR_WINOGRAD") == "1"

// winogradOK reports whether this conv should take the Winograd path.
func (o *convOp) winogradOK(G, Cg, Mg, KH, KW, OH, OW int) bool {
	return winogradEnabled &&
		KH == 3 && KW == 3 && G == 1 &&
		o.strides == [2]int{1, 1} && o.dilations == [2]int{1, 1} &&
		Cg >= 16 && Cg <= gemm.KC && OH >= 4 && OW >= 4 &&
		!gemm.PrefersSME(Mg, OH*OW, Cg*9)
}

// winogradWeights returns the 16 transformed-weight matrices U[k] = (G·g·Gᵀ)[k]
// [Cout×Cin], each packed into micro-kernel A panels (gemm.PackAPanels),
// cached while the weight storage is unchanged.
func (o *convOp) winogradWeights(wf []float32, Cout, Cin int) [][]float32 {
	o.wgMu.Lock()
	defer o.wgMu.Unlock()
	if o.wgSrc == &wf[0] && o.wgLen == len(wf) {
		return o.wgU
	}
	uk := make([][]float32, 16)
	for k := range uk {
		uk[k] = make([]float32, Cout*Cin)
	}
	var g4 [4][3]float32
	for co := 0; co < Cout; co++ {
		for ci := 0; ci < Cin; ci++ {
			g := wf[(co*Cin+ci)*9:]
			for j := 0; j < 3; j++ {
				g4[0][j] = g[j]
				g4[1][j] = 0.5 * (g[j] + g[3+j] + g[6+j])
				g4[2][j] = 0.5 * (g[j] - g[3+j] + g[6+j])
				g4[3][j] = g[6+j]
			}
			for i := 0; i < 4; i++ {
				r := g4[i]
				uk[i*4+0][co*Cin+ci] = r[0]
				uk[i*4+1][co*Cin+ci] = 0.5 * (r[0] + r[1] + r[2])
				uk[i*4+2][co*Cin+ci] = 0.5 * (r[0] - r[1] + r[2])
				uk[i*4+3][co*Cin+ci] = r[2]
			}
		}
	}
	pk := make([][]float32, 16)
	for k := range pk {
		pk[k] = gemm.PackAPanels(Cout, Cin, uk[k], Cin)
	}
	o.wgU, o.wgSrc, o.wgLen = pk, &wf[0], len(wf)
	return pk
}

// winograd runs the fused F(2×2,3×3) conv. Called with G == 1 and Cin ≤ gemm.KC.
//
// The first version materialised V (16·Cin·tiles) and M (16·Cout·tiles) per
// band (~600 KB/worker) and lost in-model to shared-cache eviction. This one
// is fused per column block of a tile row: the column transform writes each
// 12-tile group directly in the micro-kernel's packed-B panel layout, the 16
// tiny GEMMs consume it cache-hot, and the inverse transform runs immediately
// — the whole working set is one block's V + M (~250 KB).
func (o *convOp) winograd(ctx *Ctx, xf, wf, bias, of []float32, N, Cin, Cout, H, W, OH, OW int, pads [4]int) {
	pt, pl := pads[0], pads[1]
	pk := o.winogradWeights(wf, Cout, Cin)
	TH := (OH + 1) / 2
	TW := (OW + 1) / 2
	CoutP := (Cout + gemm.MR - 1) / gemm.MR * gemm.MR
	mPanels := CoutP / gemm.MR
	// Block width in tiles (multiple of NR): V+M ≈ 16·(Cin+CoutP)·blk floats,
	// target ≤ ~64K floats (256 KB) so a block stays cache-resident.
	blk := (1 << 16) / (16 * (Cin + CoutP)) / gemm.NR * gemm.NR
	blk = max(gemm.NR, min(blk, (TW+gemm.NR-1)/gemm.NR*gemm.NR))
	nBlocks := (TW + blk - 1) / blk
	Whb := blk + 1 // E/O half-row width for one block
	// Per-worker scratch: V [16][panel][Cin·NR], M [16][CoutP·blk],
	// E/O rows 8·Whb, row-combined 8·Whb, inverse rows 12·blk.
	per := 16*Cin*blk + 16*CoutP*blk + 16*Whb + 12*blk
	workers := par.Workers()
	scratch := ctx.NewUninit(tensor.F32, workers, per)
	sf := scratch.F32()
	par.For(N*TH*nBlocks, 1, func(task, wk int) {
		n := task / (TH * nBlocks)
		th := (task / nBlocks) % TH
		bi := task % nBlocks
		t0 := bi * blk
		nt := min(blk, TW-t0) // tiles in this block
		nps := (nt + gemm.NR - 1) / gemm.NR
		buf := sf[wk*per : (wk+1)*per]
		V := buf[:16*Cin*blk]
		M := buf[16*Cin*blk : 16*Cin*blk+16*CoutP*blk]
		rows := buf[16*Cin*blk+16*CoutP*blk:]
		eo := rows[:8*Whb]
		te := rows[8*Whb : 12*Whb]
		to := rows[12*Whb : 16*Whb]
		inv := rows[16*Whb:]
		in := xf[n*Cin*H*W:]
		ih0 := 2*th - pt
		iwBase := 2*t0 - pl // input col of the block's first tile

		// --- input transform: per ci, per 12-tile group, packed-B layout ---
		// V[k][jp][ci·NR .. +nrt] with tile columns de-interleaved so both
		// transform stages are shifted whole-row vector ops.
		if nt < blk {
			clear(V) // partial block: pad B panels with zeros
		}
		for ci := 0; ci < Cin; ci++ {
			plane := in[ci*H*W : (ci+1)*H*W]
			// de-interleave the 4 input rows for this block's columns
			for r := 0; r < 4; r++ {
				E := eo[r*Whb : (r+1)*Whb]
				O := eo[(4+r)*Whb : (5+r)*Whb]
				ih := ih0 + r
				if ih < 0 || ih >= H {
					clear(E[:nt+1])
					clear(O[:nt+1])
					continue
				}
				row := plane[ih*W : (ih+1)*W]
				for i := 0; i <= nt; i++ {
					iw := iwBase + 2*i
					if iw >= 0 && iw < W {
						E[i] = row[iw]
					} else {
						E[i] = 0
					}
					if iw+1 >= 0 && iw+1 < W {
						O[i] = row[iw+1]
					} else {
						O[i] = 0
					}
				}
			}
			e0, e1, e2, e3 := eo[0:Whb], eo[Whb:2*Whb], eo[2*Whb:3*Whb], eo[3*Whb:4*Whb]
			o0, o1, o2, o3 := eo[4*Whb:5*Whb], eo[5*Whb:6*Whb], eo[6*Whb:7*Whb], eo[7*Whb:8*Whb]
			w1 := nt + 1
			vek.Sub(te[0:w1], e0, e2)
			vek.Add(te[Whb:Whb+w1], e1, e2)
			vek.Sub(te[2*Whb:2*Whb+w1], e2, e1)
			vek.Sub(te[3*Whb:3*Whb+w1], e1, e3)
			vek.Sub(to[0:w1], o0, o2)
			vek.Add(to[Whb:Whb+w1], o1, o2)
			vek.Sub(to[2*Whb:2*Whb+w1], o2, o1)
			vek.Sub(to[3*Whb:3*Whb+w1], o1, o3)
			for jp := 0; jp < nps; jp++ {
				c0 := jp * gemm.NR
				nrt := min(gemm.NR, nt-c0)
				for i := 0; i < 4; i++ {
					tE := te[i*Whb+c0 : i*Whb+c0+nrt+1]
					tO := to[i*Whb+c0 : i*Whb+c0+nrt+1]
					base := (4*i)*Cin*blk + jp*Cin*gemm.NR + ci*gemm.NR
					step := Cin * blk
					vek.Sub(V[base:base+nrt], tE[:nrt], tE[1:])
					vek.Add(V[base+step:base+step+nrt], tO[:nrt], tE[1:])
					vek.Sub(V[base+2*step:base+2*step+nrt], tE[1:], tO[:nrt])
					vek.Sub(V[base+3*step:base+3*step+nrt], tO[:nrt], tO[1:])
				}
			}
		}
		// --- 16 × nps × mPanels micro-kernels, all cache-hot ---
		for k := 0; k < 16; k++ {
			vk := V[k*Cin*blk:]
			mk := M[k*CoutP*blk:]
			for jp := 0; jp < nps; jp++ {
				bp := vk[jp*Cin*gemm.NR : (jp+1)*Cin*gemm.NR]
				for mp := 0; mp < mPanels; mp++ {
					gemm.PanelKernel(Cin, pk[k][mp*gemm.KC*gemm.MR:], bp, mk[mp*gemm.MR*blk+jp*gemm.NR:], blk, false)
				}
			}
		}
		// --- inverse transform: blk-wide rows, then interleave into C ---
		a0 := [4][]float32{inv[0:blk], inv[blk : 2*blk], inv[2*blk : 3*blk], inv[3*blk : 4*blk]}
		a1 := [4][]float32{inv[4*blk : 5*blk], inv[5*blk : 6*blk], inv[6*blk : 7*blk], inv[7*blk : 8*blk]}
		y := [4][]float32{inv[8*blk : 9*blk], inv[9*blk : 10*blk], inv[10*blk : 11*blk], inv[11*blk : 12*blk]}
		oh := 2 * th
		for co := 0; co < Cout; co++ {
			var b float32
			if bias != nil {
				b = bias[co]
			}
			mrow := func(k int) []float32 {
				base := k*CoutP*blk + co*blk
				return M[base : base+nt]
			}
			for c := 0; c < 4; c++ {
				vek.Add(a0[c][:nt], mrow(c), mrow(4+c))
				vek.Add(a0[c][:nt], a0[c][:nt], mrow(8+c))
				vek.Sub(a1[c][:nt], mrow(4+c), mrow(8+c))
				vek.Sub(a1[c][:nt], a1[c][:nt], mrow(12+c))
			}
			vek.Add(y[0][:nt], a0[0][:nt], a0[1][:nt])
			vek.Add(y[0][:nt], y[0][:nt], a0[2][:nt])
			vek.Sub(y[1][:nt], a0[1][:nt], a0[2][:nt])
			vek.Sub(y[1][:nt], y[1][:nt], a0[3][:nt])
			vek.Add(y[2][:nt], a1[0][:nt], a1[1][:nt])
			vek.Add(y[2][:nt], y[2][:nt], a1[2][:nt])
			vek.Sub(y[3][:nt], a1[1][:nt], a1[2][:nt])
			vek.Sub(y[3][:nt], y[3][:nt], a1[3][:nt])
			out := of[(n*Cout+co)*OH*OW:]
			orow := out[oh*OW : (oh+1)*OW]
			for t := 0; t < nt; t++ {
				ow := 2 * (t0 + t)
				orow[ow] = y[0][t] + b
				if ow+1 < OW {
					orow[ow+1] = y[1][t] + b
				}
			}
			if oh+1 < OH {
				orow = out[(oh+1)*OW : (oh+2)*OW]
				for t := 0; t < nt; t++ {
					ow := 2 * (t0 + t)
					orow[ow] = y[2][t] + b
					if ow+1 < OW {
						orow[ow+1] = y[3][t] + b
					}
				}
			}
			if o.epi.active() {
				lo, hi := 2*t0, min(2*(t0+nt), OW)
				o.epi.apply(out[oh*OW+lo : oh*OW+hi])
				if oh+1 < OH {
					o.epi.apply(out[(oh+1)*OW+lo : (oh+1)*OW+hi])
				}
			}
		}
	})
	if ctx.Pool != nil {
		ctx.Pool.Put(scratch)
	}
}

// winograd weight cache fields live here to keep conv.go's struct readable.
type winogradCache struct {
	wgMu  sync.Mutex
	wgU   [][]float32 // 16 × packed A panels (gemm.PackAPanels)
	wgSrc *float32
	wgLen int
}
