package ops

import (
	"sync"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/vek"
	"github.com/giraffesyo/ingot/tensor"
)

// Channel-blocked (nChw8c) execution: activations live as [N][C/8][H][W][8]
// between blocked convs, converted at chain edges by ToBlk8/FromBlk8. See
// docs/DESIGN-nchwc.md.

const blkC = 8

// toBlk8Op converts NCHW → nChw8c (C must be a multiple of 8; the layout
// pass only inserts conversions where that holds).
type toBlk8Op struct{ n NodeInfo }

func (o *toBlk8Op) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	xs := x.Shape()
	if len(xs) != 4 || xs[1]%blkC != 0 {
		return nil, o.n.Errorf("ToBlk8: want NCHW with C%%8==0, got %v", xs)
	}
	N, C, H, W := xs[0], xs[1], xs[2], xs[3]
	P := H * W
	out := ctx.NewUninit(tensor.F32, N, C/blkC, H, W, blkC)
	xf, of := x.F32(), out.F32()
	// Chunk over spatial ranges too: N*C/8 alone starves a wide pool at the
	// region entry (mv2's single ToBlk8 sits at 112²x32 = 4 plane tasks).
	// For a fixed block the destination range [p0,p1) is contiguous.
	NCB := N * C / blkC
	chunks := max(1, min(P/2048, 2*par.Workers()/max(1, NCB)))
	cp := (P + chunks - 1) / chunks
	par.For(NCB*chunks, 1, func(t, _ int) {
		ncb, ch := t/chunks, t%chunks
		n, cb := ncb/(C/blkC), ncb%(C/blkC)
		p0, p1 := ch*cp, min((ch+1)*cp, P)
		dst := of[ncb*P*blkC:]
		for c := 0; c < blkC; c++ {
			src := xf[(n*C+cb*blkC+c)*P:]
			for p := p0; p < p1; p++ {
				dst[p*blkC+c] = src[p]
			}
		}
	})
	return ctx.Out(out), nil
}

// fromBlk8Op converts nChw8c → NCHW.
type fromBlk8Op struct{ n NodeInfo }

func (o *fromBlk8Op) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	xs := x.Shape()
	if len(xs) != 5 || xs[4] != blkC {
		return nil, o.n.Errorf("FromBlk8: want [N,C/8,H,W,8], got %v", xs)
	}
	N, CB, H, W := xs[0], xs[1], xs[2], xs[3]
	P := H * W
	out := ctx.NewUninit(tensor.F32, N, CB*blkC, H, W)
	xf, of := x.F32(), out.F32()
	// Spatial chunking, mirroring ToBlk8: per-plane channel writes are
	// disjoint for disjoint [p0,p1) ranges.
	chunks := max(1, min(P/2048, 2*par.Workers()/max(1, N*CB)))
	cp := (P + chunks - 1) / chunks
	par.For(N*CB*chunks, 1, func(t, _ int) {
		ncb, ch := t/chunks, t%chunks
		n, cb := ncb/CB, ncb%CB
		p0, p1 := ch*cp, min((ch+1)*cp, P)
		src := xf[ncb*P*blkC:]
		for c := 0; c < blkC; c++ {
			dst := of[(n*CB*blkC+cb*blkC+c)*P:]
			for p := p0; p < p1; p++ {
				dst[p] = src[p*blkC+c]
			}
		}
	})
	return ctx.Out(out), nil
}

// convDwBlkOp: depthwise KxK stride-S conv over nChw8c activations
// (vek.DwBlk8 row kernels; taps packed [C/8][K*K][8] on first use).
type convDwBlkOp struct {
	n    NodeInfo
	k, s int
	pads [4]int
	epi  epilogue

	packMu sync.Mutex
	wp     []float32
	wSrc   *float32
}

func (o *convDwBlkOp) taps(w []float32, C int) []float32 {
	o.packMu.Lock()
	defer o.packMu.Unlock()
	if o.wSrc == &w[0] {
		return o.wp
	}
	K := o.k
	wp := make([]float32, C*K*K)
	for cb := 0; cb < C/blkC; cb++ {
		for t := 0; t < K*K; t++ {
			for c := 0; c < blkC; c++ {
				wp[(cb*K*K+t)*blkC+c] = w[(cb*blkC+c)*K*K+t]
			}
		}
	}
	o.wp, o.wSrc = wp, &w[0]
	return wp
}

func (o *convDwBlkOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x, wt := in[0], in[1]
	xs := x.Shape()
	if len(xs) != 5 || xs[4] != blkC {
		return nil, o.n.Errorf("ConvDwBlk: want blocked input, got %v", xs)
	}
	N, CB, H, W := xs[0], xs[1], xs[2], xs[3]
	C := CB * blkC
	K, S := o.k, o.s
	pt, pl, pb, pr := o.pads[0], o.pads[1], o.pads[2], o.pads[3]
	OH := (H+pt+pb-K)/S + 1
	OW := (W+pl+pr-K)/S + 1
	var bias []float32
	if len(in) > 2 && in[2] != nil {
		bias = in[2].F32()
	}
	out := ctx.NewUninit(tensor.F32, N, CB, OH, OW, blkC)
	xf, of := x.F32(), out.F32()
	wp := o.taps(wt.F32(), C)
	Wp := W + pl + pr
	workers := par.Workers()
	// Row-strip tasks: whole planes starve a wide pool at small C (mv2's
	// early dw convs are C=32 → 4 tasks on 32 workers), and the padded copy
	// then serialises behind those few streams. Each strip copies only its
	// input window (K-1 halo rows per strip).
	strips := min(OH, max(1, 2*workers/max(1, N*CB)))
	sr := (OH + strips - 1) / strips
	strips = (OH + sr - 1) / sr
	inRows := (sr-1)*S + K
	scratch := ctx.NewUninit(tensor.F32, workers, inRows*Wp*blkC)
	sf := scratch.F32()
	par.For(N*CB*strips, 1, func(t, wk int) {
		ncb, st := t/strips, t%strips
		cb := ncb % CB
		r0 := st * sr
		r1 := min(r0+sr, OH)
		in0 := r0*S - pt // first input row the strip reads (may be out of range)
		p := sf[wk*inRows*Wp*blkC:]
		for i := 0; i < (r1-1-r0)*S+K; i++ {
			row := p[i*Wp*blkC:]
			src := in0 + i
			if src < 0 || src >= H {
				clear(row[:Wp*blkC])
				continue
			}
			clear(row[:pl*blkC])
			copy(row[pl*blkC:(pl+W)*blkC], xf[(ncb*H+src)*W*blkC:])
			clear(row[(pl+W)*blkC : Wp*blkC])
		}
		taps := wp[cb*K*K*blkC:]
		for i := r0; i < r1; i++ {
			dst := of[(ncb*OH+i)*OW*blkC : (ncb*OH+i+1)*OW*blkC]
			vek.DwBlk8(dst, p[(i-r0)*S*Wp*blkC:], taps, OW, Wp, K, S)
			if bias != nil {
				b := bias[cb*blkC : (cb+1)*blkC]
				for j := 0; j < OW; j++ {
					for c := 0; c < blkC; c++ {
						dst[j*blkC+c] += b[c]
					}
				}
			}
			o.epi.apply(dst)
		}
	})
	if ctx.Pool != nil {
		ctx.Pool.Put(scratch)
	}
	return ctx.Out(out), nil
}

// convPwBlkOp: 1x1 stride-1 conv over nChw8c (vek.PwBlk6x16 tiles; weights
// packed [Mpair][C][16] on first use — an odd trailing 8-block computes a
// discarded upper half).
type convPwBlkOp struct {
	n   NodeInfo
	epi epilogue

	packMu sync.Mutex
	wp     []float32
	wSrc   *float32
}

func (o *convPwBlkOp) weights(w []float32, M, C int) []float32 {
	o.packMu.Lock()
	defer o.packMu.Unlock()
	if o.wSrc == &w[0] {
		return o.wp
	}
	pairs := (M/blkC + 1) / 2
	wp := make([]float32, pairs*C*2*blkC)
	for pr := 0; pr < pairs; pr++ {
		for ci := 0; ci < C; ci++ {
			for oc := 0; oc < 2*blkC; oc++ {
				m := pr*2*blkC + oc
				if m < M {
					wp[(pr*C+ci)*2*blkC+oc] = w[m*C+ci]
				}
			}
		}
	}
	o.wp, o.wSrc = wp, &w[0]
	return wp
}

func (o *convPwBlkOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x, wt := in[0], in[1]
	xs := x.Shape()
	if len(xs) != 5 || xs[4] != blkC {
		return nil, o.n.Errorf("ConvPwBlk: want blocked input, got %v", xs)
	}
	N, CB, H, W := xs[0], xs[1], xs[2], xs[3]
	C := CB * blkC
	ws := wt.Shape()
	M := ws[0]
	if ws[1] != C || M%blkC != 0 {
		return nil, o.n.Errorf("ConvPwBlk: weights %v vs C=%d", ws, C)
	}
	P := H * W
	var bias []float32
	if len(in) > 2 && in[2] != nil {
		bias = in[2].F32()
	}
	// Optional fused residual (fuse-blk-res): same-shape blocked tensor added
	// after the epilogue, exactly Add(epilogued conv, r) — read while the
	// output chunk is cache-hot instead of a separate pass and pool region.
	var res []float32
	if len(in) > 3 && in[3] != nil {
		r := in[3]
		if r.Numel() != N*M*P {
			return nil, o.n.Errorf("ConvPwBlk: residual %v vs out [%d,%d,%d]", r.Shape(), N, M, P)
		}
		res = r.F32()
	}
	out := ctx.NewUninit(tensor.F32, N, M/blkC, H, W, blkC)
	xf, of := x.F32(), out.F32()
	if P < 6 {
		// Tiny spatial (the 6-position tile can't form): direct loops.
		wf := wt.F32()
		for n := 0; n < N; n++ {
			for m := 0; m < M; m++ {
				var b float32
				if bias != nil {
					b = bias[m]
				}
				for p := 0; p < P; p++ {
					var acc float32
					for ci := 0; ci < C; ci++ {
						acc += wf[m*C+ci] * xf[((n*CB+ci/blkC)*P+p)*blkC+ci%blkC]
					}
					of[((n*(M/blkC)+m/blkC)*P+p)*blkC+m%blkC] = acc + b
				}
			}
			for mb := 0; mb < M/blkC; mb++ {
				seg := of[(n*(M/blkC)+mb)*P*blkC : (n*(M/blkC)+mb+1)*P*blkC]
				o.epi.apply(seg)
				if res != nil {
					vek.Add(seg, seg, res[(n*(M/blkC)+mb)*P*blkC:(n*(M/blkC)+mb+1)*P*blkC])
				}
			}
		}
		return ctx.Out(out), nil
	}
	wp := o.weights(wt.F32(), M, C)
	pairs := (M/blkC + 1) / 2
	odd := M/blkC%2 == 1
	nTiles := (P + 5) / 6
	workers := par.Workers()
	chunk := max(1, nTiles/max(1, 2*workers/max(1, N*pairs)))
	nChunks := (nTiles + chunk - 1) / chunk
	// The ragged last tile overlaps its predecessor (p0 clamps to P-6). Keep
	// those two tiles in one chunk so bias/epilogue can run per chunk: within
	// a task the overlap is written (twice, identical raw values) before
	// post-processing; across tasks it would race. If the last chunk would
	// hold only the ragged tile, fold it into the previous chunk.
	if P%6 != 0 && nChunks > 1 && (nTiles-1)%chunk == 0 {
		nChunks--
	}
	scratch := ctx.NewUninit(tensor.F32, workers, 6*blkC)
	sf := scratch.F32()
	doEpi := bias != nil || o.epi.active() || res != nil
	par.For(N*pairs*nChunks, 1, func(t, wk int) {
		ch := t % nChunks
		np := t / nChunks
		n, pr := np/pairs, np%pairs
		xb := xf[n*CB*P*blkC:]
		d0 := of[(n*(M/blkC)+2*pr)*P*blkC:]
		discard := odd && pr == pairs-1
		var d1 []float32
		if discard {
			d1 = sf[wk*6*blkC:] // discarded upper half
		} else {
			d1 = of[(n*(M/blkC)+2*pr+1)*P*blkC:]
		}
		tEnd := min((ch+1)*chunk, nTiles)
		if ch == nChunks-1 {
			tEnd = nTiles // absorbs a folded ragged tile
		}
		t0 := ch * chunk
		w := wp[pr*C*2*blkC:]
		if discard {
			// Odd trailing pair: dst1 is a 6x8 scratch, so run per tile.
			for ti := t0; ti < tEnd; ti++ {
				p0 := ti * 6
				if p0 > P-6 {
					p0 = P - 6 // overlap tail: rewrites identical raw values
				}
				vek.PwBlk6x16(d0[p0*blkC:], d1, xb[p0*blkC:], w, C, P*blkC*4)
			}
		} else {
			// Full tiles in one asm-looped call; the ragged last tile (p0
			// clamps to P-6, overlapping its predecessor) runs singly.
			full := tEnd - t0
			ragged := tEnd == nTiles && P%6 != 0
			if ragged {
				full--
			}
			if full > 0 {
				vek.PwBlk6x16Tiles(d0[t0*6*blkC:], d1[t0*6*blkC:], xb[t0*6*blkC:], w, C, P*blkC*4, full)
			}
			if ragged {
				p0 := P - 6
				vek.PwBlk6x16(d0[p0*blkC:], d1[p0*blkC:], xb[p0*blkC:], w, C, P*blkC*4)
			}
		}
		if !doEpi {
			return
		}
		// Bias + epilogue on this chunk's position range, in cache while hot.
		// Every write to these positions happened above (tail overlap stays
		// within the chunk), and each position belongs to exactly one chunk.
		p0, p1 := ch*chunk*6, min(tEnd*6, P)
		for half := 0; half < 2; half++ {
			if half == 1 && discard {
				break
			}
			d := d0
			if half == 1 {
				d = d1
			}
			seg := d[p0*blkC : p1*blkC]
			if bias != nil {
				b := bias[(2*pr+half)*blkC : (2*pr+half+1)*blkC]
				for j := range seg {
					seg[j] += b[j&(blkC-1)]
				}
			}
			o.epi.apply(seg)
			if res != nil {
				rp := res[(n*(M/blkC)+2*pr+half)*P*blkC:]
				vek.Add(seg, seg, rp[p0*blkC:p1*blkC])
			}
		}
	})
	if ctx.Pool != nil {
		ctx.Pool.Put(scratch)
	}
	return ctx.Out(out), nil
}

func init() {
	Register("ingot", "ToBlk8", 1, func(n NodeInfo) (Op, error) { return &toBlk8Op{n}, nil })
	Register("ingot", "FromBlk8", 1, func(n NodeInfo) (Op, error) { return &fromBlk8Op{n}, nil })
	Register("ingot", "ConvDwBlk", 1, func(n NodeInfo) (Op, error) {
		o := &convDwBlkOp{n: n, k: int(n.Attrs.Int("kernel", 3)), s: int(n.Attrs.Int("stride", 1))}
		pads := n.Attrs.Ints("pads", []int64{0, 0, 0, 0})
		if len(pads) != 4 {
			return nil, n.Errorf("ConvDwBlk: need 4 pads")
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
	Register("ingot", "ConvPwBlk", 1, func(n NodeInfo) (Op, error) {
		o := &convPwBlkOp{n: n}
		var err error
		if o.epi, err = parseEpilogue(n); err != nil {
			return nil, err
		}
		return o, nil
	})
}
