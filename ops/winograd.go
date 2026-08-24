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

// winogradEnabled: the Winograd path is opt-in (OCR_WINOGRAD=1) for now. In
// isolation it beats the tiled-im2col conv on the shapes it targets (det head
// 96→24 3×3 @160²: 1.19 vs 1.56 ms), but inside a model its per-worker V/M
// scratch (16·(Cin+Cout)·tiles floats × workers, ~11 MB concurrent on the det
// head) evicts the shared cache and slows every neighbouring op — the model
// ends up net slower. To make it default-on: block the GEMM over Cin with
// beta=1 accumulation so the live V slab shrinks ~4×, and/or a fused
// transform-GEMM (NCHWc) layout. Tests force-enable it; correctness is
// oracle-verified either way.
var winogradEnabled = os.Getenv("OCR_WINOGRAD") == "1"

// winogradOK reports whether this conv should take the Winograd path.
func (o *convOp) winogradOK(G, Cg, Mg, KH, KW, OH, OW int) bool {
	return winogradEnabled &&
		KH == 3 && KW == 3 && G == 1 &&
		o.strides == [2]int{1, 1} && o.dilations == [2]int{1, 1} &&
		Cg >= 16 && OH >= 4 && OW >= 4 &&
		gemm.PackFits(Mg, Cg)
}

// winogradWeights returns the 16 pre-packed U[k] = (G·g·Gᵀ)[k] matrices
// [Cout×Cin], cached while the weight storage is unchanged.
func (o *convOp) winogradWeights(wf []float32, Cout, Cin int) *[16]*gemm.PackedA {
	o.wgMu.Lock()
	defer o.wgMu.Unlock()
	if o.wgSrc == &wf[0] && o.wgLen == len(wf) {
		return o.wgU
	}
	// u = G·g (4×3), U = u·Gᵀ (4×4), per (cout, cin).
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
	var pk [16]*gemm.PackedA
	for k := range pk {
		pk[k] = gemm.PackA(false, Cout, Cin, uk[k], Cin)
	}
	o.wgU, o.wgSrc, o.wgLen = &pk, &wf[0], len(wf)
	return o.wgU
}

// winograd runs the F(2×2,3×3) conv. Called with G == 1.
func (o *convOp) winograd(ctx *Ctx, xf, wf, bias, of []float32, N, Cin, Cout, H, W, OH, OW int, pads [4]int) {
	pt, pl := pads[0], pads[1]
	pk := o.winogradWeights(wf, Cout, Cin)
	TH := (OH + 1) / 2 // tile rows
	TW := (OW + 1) / 2 // tile cols
	// Band height in tile rows: V+M for one band ≈ 16·(Cin+Cout)·bh·TW floats;
	// target ≤ ~256 KB so the band stays L2-resident.
	bh := max(1, (1<<16)/max(16*(Cin+Cout)*TW/4*4, 1))
	// Ensure enough tasks for the pool when the problem allows.
	nBands := (TH + bh - 1) / bh
	if want := 2 * par.Workers(); N*nBands < want && TH > 1 {
		bh = max(1, min(bh, TH*N/want))
		nBands = (TH + bh - 1) / bh
	}
	Tc := bh * TW // max tiles per band
	// Per-band scratch: V, M, plus row workspaces:
	//   E/O: 4 input rows de-interleaved (Wh each), 8 rows row-combined (tE/tO),
	//   inverse: 8 Aᵀ-combined rows + 4 output-column rows (TW each).
	Wh := TW + 1
	rowsExtra := 16*Wh + 12*TW
	per := 16*(Cin+Cout)*Tc + rowsExtra
	workers := par.Workers()
	scratch := ctx.NewUninit(tensor.F32, workers, per)
	sf := scratch.F32()
	par.For(N*nBands, 1, func(task, wk int) {
		n := task / nBands
		band := task % nBands
		th0 := band * bh
		th1 := min(th0+bh, TH)
		nrows := th1 - th0
		nt := nrows * TW
		buf := sf[wk*per : (wk+1)*per]
		V := buf[:16*Cin*nt]
		M := buf[16*Cin*nt : 16*Cin*nt+16*Cout*nt]
		rows := buf[16*(Cin+Cout)*Tc:]
		eo := rows[:8*Wh]     // E0..E3, O0..O3
		tc := rows[8*Wh:]     // tE0..tE3, tO0..tO3 (Wh) then inverse rows (TW)
		te := tc[:4*Wh]       // row-combined even halves
		to := tc[4*Wh : 8*Wh] // row-combined odd halves
		inv := rows[16*Wh:]   // 12 TW-wide rows for the inverse transform
		in := xf[n*Cin*H*W:]

		// --- input transform ---
		// Tiles in a row overlap by 2 columns, so with each padded input row
		// split into even/odd column halves, tile t's 4 columns are
		// E[t], O[t], E[t+1], O[t+1] — and both the row combination (Bᵀ·d) and
		// the column combination (·B) become whole-row vector ops with
		// contiguous stores into each frequency plane.
		for ci := 0; ci < Cin; ci++ {
			plane := in[ci*H*W : (ci+1)*H*W]
			for th := th0; th < th1; th++ {
				ih0 := 2*th - pt
				// de-interleave 4 padded input rows into E/O halves
				for r := 0; r < 4; r++ {
					E := eo[r*Wh : (r+1)*Wh]
					O := eo[(4+r)*Wh : (5+r)*Wh]
					ih := ih0 + r
					if ih < 0 || ih >= H {
						clear(E)
						clear(O)
						continue
					}
					row := plane[ih*W : (ih+1)*W]
					for i := 0; i < Wh; i++ {
						iw := 2*i - pl
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
				e0, e1, e2, e3 := eo[0:Wh], eo[Wh:2*Wh], eo[2*Wh:3*Wh], eo[3*Wh:4*Wh]
				o0, o1, o2, o3 := eo[4*Wh:5*Wh], eo[5*Wh:6*Wh], eo[6*Wh:7*Wh], eo[7*Wh:8*Wh]
				// Bᵀ·d row combinations, on both halves
				vek.Sub(te[0:Wh], e0, e2)
				vek.Add(te[Wh:2*Wh], e1, e2)
				vek.Sub(te[2*Wh:3*Wh], e2, e1)
				vek.Sub(te[3*Wh:4*Wh], e1, e3)
				vek.Sub(to[0:Wh], o0, o2)
				vek.Add(to[Wh:2*Wh], o1, o2)
				vek.Sub(to[2*Wh:3*Wh], o2, o1)
				vek.Sub(to[3*Wh:4*Wh], o1, o3)
				// ·B column combinations: contiguous TW-wide stores into V
				tb := (th - th0) * TW
				for i := 0; i < 4; i++ {
					tE := te[i*Wh : (i+1)*Wh]
					tO := to[i*Wh : (i+1)*Wh]
					base := ci*nt + tb
					vek.Sub(V[(4*i+0)*Cin*nt+base:(4*i+0)*Cin*nt+base+TW], tE[:TW], tE[1:])
					vek.Add(V[(4*i+1)*Cin*nt+base:(4*i+1)*Cin*nt+base+TW], tO[:TW], tE[1:])
					vek.Sub(V[(4*i+2)*Cin*nt+base:(4*i+2)*Cin*nt+base+TW], tE[1:], tO[:TW])
					vek.Sub(V[(4*i+3)*Cin*nt+base:(4*i+3)*Cin*nt+base+TW], tO[:TW], tO[1:])
				}
			}
		}
		// --- 16 GEMMs: M[k] = U[k]·V[k] ---
		for k := 0; k < 16; k++ {
			gemm.SgemmPackedA(pk[k], nt, V[k*Cin*nt:(k+1)*Cin*nt], nt, 0, M[k*Cout*nt:(k+1)*Cout*nt], nt, false)
		}
		// --- inverse transform: Y = Aᵀ·M·A, whole tile-rows at a time ---
		a0 := [4][]float32{inv[0:TW], inv[TW : 2*TW], inv[2*TW : 3*TW], inv[3*TW : 4*TW]}
		a1 := [4][]float32{inv[4*TW : 5*TW], inv[5*TW : 6*TW], inv[6*TW : 7*TW], inv[7*TW : 8*TW]}
		y := [4][]float32{inv[8*TW : 9*TW], inv[9*TW : 10*TW], inv[10*TW : 11*TW], inv[11*TW : 12*TW]}
		for co := 0; co < Cout; co++ {
			var b float32
			if bias != nil {
				b = bias[co]
			}
			out := of[(n*Cout+co)*OH*OW:]
			for th := th0; th < th1; th++ {
				tb := (th - th0) * TW
				mrow := func(k int) []float32 {
					base := k*Cout*nt + co*nt + tb
					return M[base : base+TW]
				}
				// Aᵀ·M: per column c of the 4×4, combine the three row planes
				for c := 0; c < 4; c++ {
					vek.Add(a0[c], mrow(c), mrow(4+c))
					vek.Add(a0[c], a0[c], mrow(8+c))
					vek.Sub(a1[c], mrow(4+c), mrow(8+c))
					vek.Sub(a1[c], a1[c], mrow(12+c))
				}
				// ·A: 2×2 output values per tile, as TW-wide rows
				vek.Add(y[0], a0[0], a0[1])
				vek.Add(y[0], y[0], a0[2])
				vek.Sub(y[1], a0[1], a0[2])
				vek.Sub(y[1], y[1], a0[3])
				vek.Add(y[2], a1[0], a1[1])
				vek.Add(y[2], y[2], a1[2])
				vek.Sub(y[3], a1[1], a1[2])
				vek.Sub(y[3], y[3], a1[3])
				// interleave into the two output rows (+bias)
				oh := 2 * th
				orow := out[oh*OW : (oh+1)*OW]
				for t := 0; t < TW; t++ {
					ow := 2 * t
					orow[ow] = y[0][t] + b
					if ow+1 < OW {
						orow[ow+1] = y[1][t] + b
					}
				}
				if oh+1 < OH {
					orow = out[(oh+1)*OW : (oh+2)*OW]
					for t := 0; t < TW; t++ {
						ow := 2 * t
						orow[ow] = y[2][t] + b
						if ow+1 < OW {
							orow[ow+1] = y[3][t] + b
						}
					}
				}
			}
			if o.epi.active() {
				o.epi.apply(out[2*th0*OW : min(2*th1, OH)*OW])
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
	wgU   *[16]*gemm.PackedA
	wgSrc *float32
	wgLen int
}
