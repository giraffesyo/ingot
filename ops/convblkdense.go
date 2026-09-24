package ops

import (
	"sync"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/vek"
	"github.com/giraffesyo/ingot/tensor"
)

// Dense KxK convolution (stride 1 or 2) over nChw8c activations, as KH·KW
// blocked pointwise products (vek.PwBlk6x16) over shifted views of a
// padded input.
//
// The input is copied once into zero-padded blocked planes. Stride 1: one
// plane of row length Wq = Wp; stride 2: the padded plane split into its
// four (row, column) parity phases, each Hq×Wq (half size, rounded up).
// Output position q = oy·Wq + ox (ox < Wq; the columns past OW are junk)
// reads tap (ky, kx) from phase (ky%S, kx%S) at q + (ky/S)·Wq + kx/S — a
// constant offset — so 6-position tiles run straight across rows with no
// bounds checks, even on 4×4 planes. Each tap's tile lands in a scratch
// pair and is added into the accumulator; the valid columns are then
// extracted with bias and epilogue.

// denseBlkWeights packs w [M][C][KH][KW] per tap as the pointwise tile's
// [pair][C][16] layout: out[t][pair][ci][16].
func denseBlkWeights(w []float32, M, C, KH, KW int) []float32 {
	pairs := (M/blkC + 1) / 2
	taps := KH * KW
	wp := make([]float32, taps*pairs*C*2*blkC)
	for t := 0; t < taps; t++ {
		for pr := 0; pr < pairs; pr++ {
			for ci := 0; ci < C; ci++ {
				for oc := 0; oc < 2*blkC; oc++ {
					m := pr*2*blkC + oc
					if m < M {
						wp[((t*pairs+pr)*C+ci)*2*blkC+oc] = w[(m*C+ci)*taps+t]
					}
				}
			}
		}
	}
	return wp
}

// denseGeom is a blocked dense conv's padded/phase geometry.
type denseGeom struct {
	S, Hp, Wp, Hq, Wq, OH, OW int
	phase                     int // floats per phase plane (with slack)
	plane                     int // floats per channel block (all phases)
	accPlane                  int // floats per output block of the accumulator
}

func newDenseGeom(H, W, KH, KW, S int, pads [4]int) denseGeom {
	g := denseGeom{S: S, Hp: H + pads[0] + pads[2], Wp: W + pads[1] + pads[3]}
	g.Hq, g.Wq = (g.Hp+S-1)/S, (g.Wp+S-1)/S
	g.OH, g.OW = (g.Hp-KH)/S+1, (g.Wp-KW)/S+1
	// The last tile of the last row reads up to (KW-1)/S + 5 positions past
	// the phase plane's end.
	g.phase = (g.Hq*g.Wq + KW + 6) * blkC
	g.plane = S * S * g.phase
	g.accPlane = (g.OH*g.Wq + 6) * blkC
	return g
}

// denseBlkScratch returns the pad and accumulator sizes (floats).
func denseBlkScratch(N, C, M int, g denseGeom) (pad, acc int) {
	pairs := (M/blkC + 1) / 2
	return N * (C / blkC) * g.plane, N * 2 * pairs * g.accPlane
}

// denseBlkConv computes out [N][M/8][OH][OW][8] = conv(x [N][C/8][H][W][8])
// + bias, then the epilogue (nil: none), with stride g.S and padding pads
// (top, left, bottom, right). wp comes from denseBlkWeights; pad and acc
// are scratch of denseBlkScratch floats.
func denseBlkConv(x, wp, bias, out, pad, acc []float32, N, C, H, W, M, KH, KW int, pads [4]int, g denseGeom, epi *epilogue) {
	CB := C / blkC
	MB := M / blkC
	pairs := (MB + 1) / 2
	S := g.S
	// 1. Pad (and split into stride phases).
	par.For(N*CB, max(1, unaryChunk/max(1, g.Hp*g.Wp*blkC)), func(nc, _ int) {
		dst := pad[nc*g.plane : (nc+1)*g.plane]
		clear(dst)
		src := x[nc*H*W*blkC:]
		for y := 0; y < H; y++ {
			py := y + pads[0]
			row := src[y*W*blkC : (y+1)*W*blkC]
			if S == 1 {
				copy(dst[(py*g.Wq+pads[1])*blkC:], row)
				continue
			}
			for xx := 0; xx < W; xx++ {
				px := xx + pads[1]
				ph := (py%S)*S + px%S
				o := ph*g.phase + ((py/S)*g.Wq+px/S)*blkC
				copy(dst[o:o+blkC], row[xx*blkC:(xx+1)*blkC])
			}
		}
	})
	// 2. Per (image, output pair, tile chunk): KH·KW pointwise tiles.
	Q := g.OH * g.Wq // output positions including junk columns
	nTiles := (Q + 5) / 6
	chunk := max(1, (1<<14)/max(1, C*2*blkC*KH*KW/6)) // ~16K MACs·taps per chunk
	chunk = min(chunk, nTiles)
	nChunks := (nTiles + chunk - 1) / chunk
	par.For(N*pairs*nChunks, 1, func(t, _ int) {
		ch := t % nChunks
		np := t / nChunks
		n, pr := np/pairs, np%pairs
		var tmp [2][6 * blkC]float32
		a0 := acc[(n*2*pairs+2*pr)*g.accPlane:]
		a1 := acc[(n*2*pairs+2*pr+1)*g.accPlane:]
		xb := pad[n*CB*g.plane:]
		for ti := ch * chunk; ti < min((ch+1)*chunk, nTiles); ti++ {
			q0 := ti * 6
			d0, d1 := a0[q0*blkC:q0*blkC+6*blkC], a1[q0*blkC:q0*blkC+6*blkC]
			for tap := 0; tap < KH*KW; tap++ {
				ky, kx := tap/KW, tap%KW
				off := ((ky%S)*S+kx%S)*g.phase + ((ky/S)*g.Wq+kx/S+q0)*blkC
				w := wp[((tap*pairs)+pr)*C*2*blkC:]
				if tap == 0 {
					vek.PwBlk6x16(d0, d1, xb[off:], w, C, g.plane*4)
					continue
				}
				vek.PwBlk6x16(tmp[0][:], tmp[1][:], xb[off:], w, C, g.plane*4)
				vek.Add(d0, d0, tmp[0][:])
				vek.Add(d1, d1, tmp[1][:])
			}
		}
	})
	// 3. Extract valid columns, bias, epilogue.
	OH, OW := g.OH, g.OW
	par.For(N*MB, max(1, unaryChunk/max(1, OH*OW*blkC)), func(nm, _ int) {
		n, mb := nm/MB, nm%MB
		src := acc[(n*2*pairs+mb)*g.accPlane:]
		dst := out[nm*OH*OW*blkC : (nm+1)*OH*OW*blkC]
		for y := 0; y < OH; y++ {
			copy(dst[y*OW*blkC:(y+1)*OW*blkC], src[y*g.Wq*blkC:])
		}
		if bias != nil {
			b := bias[mb*blkC : (mb+1)*blkC]
			for j := range dst {
				dst[j] += b[j&(blkC-1)]
			}
		}
		if epi != nil {
			epi.apply(dst)
		}
	})
}

// convDenseBlkOp: dense KxK stride-1/2 conv over nChw8c activations
// (denseBlkConv; weights packed per tap on first use).
type convDenseBlkOp struct {
	n    NodeInfo
	k, s int
	pads [4]int
	epi  epilogue

	packMu sync.Mutex
	wp     []float32
	wSrc   *float32
}

func (o *convDenseBlkOp) weights(w []float32, M, C int) []float32 {
	o.packMu.Lock()
	defer o.packMu.Unlock()
	if o.wSrc != &w[0] {
		o.wp, o.wSrc = denseBlkWeights(w, M, C, o.k, o.k), &w[0]
	}
	return o.wp
}

func (o *convDenseBlkOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x, wt := in[0], in[1]
	xs, ws := x.Shape(), wt.Shape()
	if len(xs) != 5 || xs[4] != blkC {
		return nil, o.n.Errorf("ConvDenseBlk: want blocked input, got %v", xs)
	}
	N, CB, H, W := xs[0], xs[1], xs[2], xs[3]
	C := CB * blkC
	if len(ws) != 4 || ws[1] != C || ws[0]%blkC != 0 || ws[2] != o.k || ws[3] != o.k {
		return nil, o.n.Errorf("ConvDenseBlk: weights %v vs C=%d K=%d", ws, C, o.k)
	}
	M := ws[0]
	var bias []float32
	if len(in) > 2 && in[2] != nil {
		bias = in[2].F32()
	}
	g := newDenseGeom(H, W, o.k, o.k, o.s, o.pads)
	if g.OH <= 0 || g.OW <= 0 {
		return nil, o.n.Errorf("ConvDenseBlk: empty output %dx%d", g.OH, g.OW)
	}
	np, na := denseBlkScratch(N, C, M, g)
	pad, acc := ctx.NewUninit(tensor.F32, np), ctx.NewUninit(tensor.F32, na)
	out := ctx.NewUninit(tensor.F32, N, M/blkC, g.OH, g.OW, blkC)
	var epi *epilogue
	if o.epi.active() {
		epi = &o.epi
	}
	denseBlkConv(x.F32(), o.weights(wt.F32(), M, C), bias, out.F32(), pad.F32(), acc.F32(), N, C, H, W, M, o.k, o.k, o.pads, g, epi)
	if ctx.Pool != nil {
		ctx.Pool.Put(pad)
		ctx.Pool.Put(acc)
	}
	return ctx.Out(out), nil
}

func init() {
	Register("ingot", "ConvDenseBlk", 1, func(n NodeInfo) (Op, error) {
		o := &convDenseBlkOp{n: n, k: int(n.Attrs.Int("kernel", 3)), s: int(n.Attrs.Int("stride", 1))}
		pads := n.Attrs.Ints("pads", []int64{0, 0, 0, 0})
		if len(pads) != 4 || (o.s != 1 && o.s != 2) {
			return nil, n.Errorf("ConvDenseBlk: need 4 pads and stride 1 or 2")
		}
		for i, p := range pads {
			o.pads[i] = int(p)
		}
		var err error
		if o.epi, err = parseEpilogue(n); err != nil {
			return nil, err
		}
		return o, nil
	})
}
