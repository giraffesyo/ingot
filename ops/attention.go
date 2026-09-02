package ops

import (
	"math"
	"sync"

	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/vek"
	"github.com/giraffesyo/ingot/tensor"
)

// mhaOp is the fused multi-head attention produced by the graph optimizer's
// fuse-attention passes: input is packed QKV — layout 0 [3, B, H, T, dh]
// (legacy exporter: head-split permute then unbind) or layout 1
// [B, T, 3, H, dh] (dynamo exporter: the qkv Linear's output viewed 5-D, read
// strided so the permute never runs); the op computes softmax(scale·Q·Kᵀ)·V
// per (batch, head) and writes the result directly in [B, T, H, dh] order
// (the final transpose of the exported pattern folds into the output GEMM's
// leading dimension).
type mhaOp struct {
	n      NodeInfo
	scale  float32
	packed bool // layout 1
}

func (o *mhaOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 1 || in[0] == nil {
		return nil, o.n.Errorf("MHA: missing input")
	}
	x := in[0]
	xs := x.Shape()
	var B, H, T, dh int
	// Per-(b,h) operand addressing: q/k/v row t at base + t*ld.
	var qOff, kOff, vOff func(b, h int) int
	var ld int
	if o.packed {
		if x.DType() != tensor.F32 || len(xs) != 5 || xs[2] != 3 {
			return nil, o.n.Errorf("MHA: want f32 [B,T,3,H,dh], got %s %v", x.DType(), xs)
		}
		B, T, H, dh = xs[0], xs[1], xs[3], xs[4]
		ld = 3 * H * dh
		qOff = func(b, h int) int { return b*T*ld + h*dh }
		kOff = func(b, h int) int { return b*T*ld + H*dh + h*dh }
		vOff = func(b, h int) int { return b*T*ld + 2*H*dh + h*dh }
	} else {
		if x.DType() != tensor.F32 || len(xs) != 5 || xs[0] != 3 {
			return nil, o.n.Errorf("MHA: want f32 [3,B,H,T,dh], got %s %v", x.DType(), xs)
		}
		B, H, T, dh = xs[1], xs[2], xs[3], xs[4]
		ld = dh
		plane := T * dh
		qOff = func(b, h int) int { return (b*H + h) * plane }
		kOff = func(b, h int) int { return B*H*plane + (b*H+h)*plane }
		vOff = func(b, h int) int { return 2*B*H*plane + (b*H+h)*plane }
	}
	out := ctx.NewUninit(tensor.F32, B, T, H, dh)
	xf, of := x.F32(), out.F32()
	workers := par.Workers()
	rows, nRow := attnRows(B, H, T, T, dh, workers)
	gemmT := gemm.SgemmTSerial // see sdpaOp: nested per-head fan-out never pays
	if nRow == 1 && B*H < workers && T*T*dh >= 1<<22 {
		gemmT = gemm.SgemmT
	}
	sT := ctx.NewUninit(tensor.F32, workers, rows*T)
	sAll := sT.F32()
	par.For(B*H*nRow, attnGrain(rows, T, dh), func(task, wk int) {
		bh, rc := task/nRow, task%nRow
		b, h := bh/H, bh%H
		t0 := rc * rows
		tc := min(rows, T-t0)
		q := xf[qOff(b, h)+t0*ld:]
		k := xf[kOff(b, h):]
		v := xf[vOff(b, h):]
		s := sAll[wk*rows*T:]
		// scores = scale · Q·Kᵀ (K stays row-major: the transposed-B GEMM).
		gemmT(false, true, tc, T, dh, o.scale, q, ld, k, ld, 0, s, T)
		for t := 0; t < tc; t++ {
			softmaxRow(s[t*T:(t+1)*T], s[t*T:(t+1)*T], false)
		}
		// out[b, t0:, h, :] = probs·V, written straight into [B,T,H,dh] via ldc.
		gemmT(false, false, tc, dh, T, 1, s, T, v, ld, 0, of[((b*T+t0)*H+h)*dh:], H*dh)
	})
	if ctx.Pool != nil {
		ctx.Pool.Put(sT)
	}
	return ctx.Out(out), nil
}

// sdpaOp is the generic fused scaled-dot-product-attention core produced by
// fuse-sdpa: scores = scale·A·B (A [B,H,T,dh], B [B,H,dh,T] — however the
// graph produced it), optional additive mask, row softmax in-tile, then
// probs·V ([B,H,T,dh]). With strideOut the result is written directly in
// [B,T,H,dh] order (folding the usual trailing transpose).
type sdpaOp struct {
	n                NodeInfo
	scale            float32
	strideOut        bool
	cache            bool // decode form: K/V append to the Ctx.Decode slot
	kLay             int  // decode form: 1 = kNew arrives [1,Tn,H,dh]
	aLay, bLay, vLay int  // input layouts, see fuse-sdpa (0 = as-declared)

	// Cached classification of the (constant) mask into (rows × sdpaBk)
	// blocks: 0 = clean (all zero), 1 = mixed, 2 = fully masked (skipped
	// entirely — the upper triangle of a causal mask is ~half the work).
	stMu   sync.Mutex
	stMask *float32
	stR    int
	stTk   int
	st     []uint8
}

// sdpaBk is the flash path's key-block width.
const sdpaBk = 128

const (
	blkClean = iota
	blkMixed
	blkMasked
)

// blockStates classifies mask blocks for the (rows, Tk) geometry, cached on
// the op (the mask is a graph constant).
func (o *sdpaOp) blockStates(mask []float32, T, Tk, rows int) []uint8 {
	o.stMu.Lock()
	defer o.stMu.Unlock()
	if o.stMask == &mask[0] && o.stR == rows && o.stTk == Tk {
		return o.st
	}
	nRow := (T + rows - 1) / rows
	nKb := (Tk + sdpaBk - 1) / sdpaBk
	st := make([]uint8, nRow*nKb)
	negInf := float32(math.Inf(-1))
	for rt := 0; rt < nRow; rt++ {
		t0 := rt * rows
		tcnt := min(rows, T-t0)
		for kb := 0; kb < nKb; kb++ {
			k0 := kb * sdpaBk
			kc := min(sdpaBk, Tk-k0)
			masked, clean := true, true
			for t := t0; t < t0+tcnt && (masked || clean); t++ {
				row := mask[t*Tk+k0 : t*Tk+k0+kc]
				for _, v := range row {
					if v != 0 {
						clean = false
					}
					if v != negInf {
						masked = false
					}
					if !masked && !clean {
						break
					}
				}
			}
			switch {
			case masked:
				st[rt*nKb+kb] = blkMasked
			case clean:
				st[rt*nKb+kb] = blkClean
			default:
				st[rt*nKb+kb] = blkMixed
			}
		}
	}
	o.stMask, o.stR, o.stTk, o.st = &mask[0], rows, Tk, st
	return st
}

func (o *sdpaOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 3 || in[0] == nil || in[1] == nil || in[2] == nil {
		return nil, o.n.Errorf("SDPA: need A, B, V")
	}
	if o.cache {
		return o.runCached(ctx, in)
	}
	a, bm, v := in[0], in[1], in[2]
	var mask []float32
	as, bs, vs := a.Shape(), bm.Shape(), v.Shape()
	// Rank-3 operands (batch and heads pre-flattened — dynamo bmm exports,
	// e.g. PARSeq): identical math as [1, B·H, T, dh]. Default layouts only;
	// the layout/stride variants are rank-4 exporter patterns.
	rank3 := len(as) == 3 && len(bs) == 3 && len(vs) == 3 &&
		o.aLay == 0 && o.bLay == 0 && o.vLay == 0 && !o.strideOut
	if rank3 {
		as = append([]int{1}, as...)
		bs = append([]int{1}, bs...)
		vs = append([]int{1}, vs...)
	}
	if a.DType() != tensor.F32 || len(as) != 4 || len(bs) != 4 || len(vs) != 4 {
		return nil, o.n.Errorf("SDPA: want 4-D f32, got %v %v %v", as, bs, vs)
	}
	var B, H, T, dh int
	if o.aLay == 1 { // A stored [B,T,H,dh]
		B, T, H, dh = as[0], as[1], as[2], as[3]
	} else { // [B,H,T,dh]
		B, H, T, dh = as[0], as[1], as[2], as[3]
	}
	var Tk int
	switch o.bLay {
	case 1: // [B,Tk,H,dh]
		Tk = bs[1]
		if bs[0] != B || bs[2] != H || bs[3] != dh {
			return nil, o.n.Errorf("SDPA: B layout 1 mismatch A%v B%v", as, bs)
		}
	case 2: // [B,H,Tk,dh]
		Tk = bs[2]
		if bs[0] != B || bs[1] != H || bs[3] != dh {
			return nil, o.n.Errorf("SDPA: B layout 2 mismatch A%v B%v", as, bs)
		}
	default: // [B,H,dh,Tk]
		Tk = bs[3]
		if bs[0] != B || bs[1] != H || bs[2] != dh {
			return nil, o.n.Errorf("SDPA: shape mismatch A%v B%v", as, bs)
		}
	}
	if o.vLay == 1 { // [B,Tk,H,dh]
		if vs[0] != B || vs[1] != Tk || vs[2] != H || vs[3] != dh {
			return nil, o.n.Errorf("SDPA: V layout 1 mismatch V%v", vs)
		}
	} else if vs[0] != B || vs[1] != H || vs[2] != Tk || vs[3] != dh {
		return nil, o.n.Errorf("SDPA: shape mismatch V%v", vs)
	}
	if len(in) > 3 && in[3] != nil {
		m := in[3]
		if m.DType() != tensor.F32 || m.Numel() != T*Tk {
			return nil, o.n.Errorf("SDPA: mask must be f32 [T,Tk], got %s %v", m.DType(), m.Shape())
		}
		mask = m.F32()
	}
	var out *tensor.Tensor
	if o.strideOut {
		out = ctx.NewUninit(tensor.F32, B, T, H, dh)
	} else if rank3 {
		out = ctx.NewUninit(tensor.F32, H, T, dh) // same flat layout, B == 1
	} else {
		out = ctx.NewUninit(tensor.F32, B, H, T, dh)
	}
	af, bf, vf, of := a.F32(), bm.F32(), v.F32(), out.F32()
	workers := par.Workers()
	rows, nRow := attnRows(B, H, T, Tk, dh, workers)
	// Per-tile GEMMs run serially inside the parallel region. A head only
	// fans out its own GEMMs when it is both huge and alone: B·H·nRow tasks
	// already fill the pool otherwise, and a nested region per head just adds
	// claim traffic — PARSeq at batch 8 (48 heads of 1M MACs on 12 workers)
	// spent 2.3 ms per SDPA that way vs 0.25 ms serial.
	gemmT := gemm.SgemmTSerial
	if nRow == 1 && B*H < workers && T*Tk*dh >= 1<<22 {
		gemmT = gemm.SgemmT
	}
	// Flash path: masked attention at long Tk streams key blocks with an
	// online softmax — fully-masked blocks (the causal upper triangle,
	// ~half the work) are skipped outright, and the working set per tile
	// drops from rows×Tk to rows×Bk.
	flash := mask != nil && Tk >= 4*sdpaBk // fewer blocks: skip savings < online-softmax overhead (T=256 measured flat-to-worse)
	var st []uint8
	scratchPer := rows * Tk
	if flash {
		st = o.blockStates(mask, T, Tk, rows)
		scratchPer = rows*sdpaBk + rows*dh + 2*rows
	}
	sT := ctx.NewUninit(tensor.F32, workers, scratchPer)
	sAll := sT.F32()
	par.For(B*H*nRow, attnGrain(rows, Tk, dh), func(task, wk int) {
		bh, rc := task/nRow, task%nRow
		b, h := bh/H, bh%H
		t0 := rc * rows
		tc := min(rows, T-t0)
		ab, alda := af[((b*H+h)*T+t0)*dh:], dh
		if o.aLay == 1 {
			ab, alda = af[((b*T+t0)*H+h)*dh:], H*dh
		}
		var bb []float32
		var bldb int
		bT := false
		switch o.bLay {
		case 1:
			bb, bldb, bT = bf[(b*Tk*H+h)*dh:], H*dh, true
		case 2:
			bb, bldb, bT = bf[(b*H+h)*Tk*dh:], dh, true
		default:
			bb, bldb = bf[(b*H+h)*dh*Tk:], Tk
		}
		vb, vldb := vf[(b*H+h)*Tk*dh:], dh
		if o.vLay == 1 {
			vb, vldb = vf[(b*Tk*H+h)*dh:], H*dh
		}
		if flash {
			buf := sAll[wk*scratchPer:]
			sblk := buf[:rows*sdpaBk]
			acc := buf[rows*sdpaBk : rows*sdpaBk+rows*dh]
			mrow := buf[rows*sdpaBk+rows*dh : rows*sdpaBk+rows*dh+rows]
			lrow := buf[rows*sdpaBk+rows*dh+rows:]
			clear(acc[:tc*dh])
			nKb := (Tk + sdpaBk - 1) / sdpaBk
			stRow := st[rc*nKb:]
			for i := 0; i < tc; i++ {
				mrow[i] = float32(math.Inf(-1))
				lrow[i] = 0
			}
			for kb := 0; kb < nKb; kb++ {
				state := stRow[kb]
				if state == blkMasked {
					continue
				}
				k0 := kb * sdpaBk
				kc := min(sdpaBk, Tk-k0)
				var bkb []float32
				switch o.bLay {
				case 1:
					bkb = bb[k0*H*dh:]
				case 2:
					bkb = bb[k0*dh:]
				default:
					bkb = bb[k0:]
				}
				gemmT(false, bT, tc, kc, dh, o.scale, ab, alda, bkb, bldb, 0, sblk, kc)
				for i := 0; i < tc; i++ {
					row := sblk[i*kc : (i+1)*kc]
					if state == blkMixed {
						vek.Add(row, row, mask[(t0+i)*Tk+k0:(t0+i)*Tk+k0+kc])
					}
					mb := row[0]
					for _, v := range row[1:] {
						if v > mb {
							mb = v
						}
					}
					mnew := mrow[i]
					if mb > mnew {
						mnew = mb
					}
					vek.AddScalar(row, row, -mnew)
					vek.Exp(row, row)
					// Flush the saturated exp of masked entries (1.2e-38) to
					// zero in the sum pass — their products in the acc GEMM
					// would be subnormal on x86 (the softmax ambush).
					var sum float32
					for j, v := range row {
						if v < 1e-30 {
							row[j] = 0
							continue
						}
						sum += v
					}
					if mrow[i] != mnew {
						if lrow[i] != 0 {
							sc := expf(mrow[i] - mnew)
							lrow[i] *= sc
							vek.MulScalar(acc[i*dh:(i+1)*dh], acc[i*dh:(i+1)*dh], sc)
						}
						mrow[i] = mnew
					}
					lrow[i] += sum
				}
				var vkb []float32
				if o.vLay == 1 {
					vkb = vb[k0*H*dh:]
				} else {
					vkb = vb[k0*dh:]
				}
				gemmT(false, false, tc, dh, kc, 1, sblk, kc, vkb, vldb, 1, acc, dh)
			}
			for i := 0; i < tc; i++ {
				inv := float32(0)
				if lrow[i] != 0 {
					inv = 1 / lrow[i]
				}
				var dst []float32
				if o.strideOut {
					dst = of[((b*T+t0+i)*H+h)*dh:]
				} else {
					dst = of[((b*H+h)*T+t0+i)*dh:]
				}
				vek.MulScalar(dst[:dh], acc[i*dh:(i+1)*dh], inv)
			}
			return
		}
		s := sAll[wk*rows*Tk:]
		gemmT(false, bT, tc, Tk, dh, o.scale, ab, alda, bb, bldb, 0, s, Tk)
		for t := 0; t < tc; t++ {
			row := s[t*Tk : (t+1)*Tk]
			if mask != nil {
				vek.Add(row, row, mask[(t0+t)*Tk:(t0+t+1)*Tk])
			}
			softmaxRow(row, row, false)
		}
		if o.strideOut {
			gemmT(false, false, tc, dh, Tk, 1, s, Tk, vb, vldb, 0, of[((b*T+t0)*H+h)*dh:], H*dh)
		} else {
			gemmT(false, false, tc, dh, Tk, 1, s, Tk, vb, vldb, 0, of[((b*H+h)*T+t0)*dh:], dh)
		}
	})
	if ctx.Pool != nil {
		ctx.Pool.Put(sT)
	}
	return ctx.Out(out), nil
}

// attnRows picks the query-row tile for attention's parallel tasks. Small
// shapes keep one tile per head (rows = T); large T is split so B·H·nRow
// gives every worker at least ~2 tiles (flash-style row tiling: the score
// tile stays cache-resident, and utilization no longer caps at B·H workers).
func attnRows(b, h, t, tk, dh, workers int) (rows, nRow int) {
	rows = t
	// Split a head into row tiles only when the per-tile GEMM stays big
	// enough to amortise re-packing K each tile (≥512K MACs after halving:
	// a 128×128×64 head split to 32-row tiles ran −45%, ViT-B's 197-token
	// heads −21%; the old 2M floor left B·H=6 tasks on an 18-wide pool).
	for b*h*((t+rows-1)/rows) < 2*workers && rows > 32 && rows*tk*dh >= 1<<19 {
		rows = (rows + 1) / 2
	}
	return rows, (t + rows - 1) / rows
}

// attnGrain sizes attention's parallel chunks so one chunk carries a few µs
// of work: microscopic tiles run inline on the caller instead of waking the
// worker pool, which costs more than the math.
func attnGrain(rows, tk, dh int) int {
	const minMACs = 1 << 15 // two GEMMs per tile, ~µs-scale at small sizes
	w := 2 * rows * tk * dh
	if w >= minMACs {
		return 1
	}
	return (minMACs + w - 1) / w
}

func init() {
	Register("ingot", "MHA", 1, func(n NodeInfo) (Op, error) {
		return &mhaOp{n: n, scale: n.Attrs.Float("scale", 1), packed: n.Attrs.Int("layout", 0) == 1}, nil
	})
	Register("ingot", "SDPA", 1, func(n NodeInfo) (Op, error) {
		return &sdpaOp{n: n, scale: n.Attrs.Float("scale", 1), strideOut: n.Attrs.Int("stride_out", 0) != 0,
			cache: n.Attrs.Int("cache", 0) != 0,
			kLay:  int(n.Attrs.Int("k_layout", 0)),
			aLay:  int(n.Attrs.Int("a_layout", 0)), bLay: int(n.Attrs.Int("b_layout", 0)), vLay: int(n.Attrs.Int("v_layout", 0))}, nil
	})
}

func expf(x float32) float32 { return float32(math.Exp(float64(x))) }

// runCached is the decode form: inputs are q, kNew, vNew — each
// [1, H, Tnew, dh] — appended to this node's KV slot; queries attend
// causally over [0, Pos+i]. See docs/DESIGN-kvcache.md.
func (o *sdpaOp) runCached(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if ctx.Decode == nil {
		return nil, o.n.Errorf("SDPA(cache): no decode state on this run")
	}
	st := ctx.Decode
	q, kn, vn := in[0], in[1], in[2]
	qs := q.Shape()
	if len(qs) != 4 || qs[0] != 1 {
		return nil, o.n.Errorf("SDPA(cache): want [1,H,T,dh] q, got %v", qs)
	}
	H, Tn, dh := qs[1], qs[2], qs[3]
	slot := st.Slots[o.n.Name]
	if slot == nil {
		// First touch: sizes are only known at Run time. The executor is
		// sequential within a Run, and each Decode owns its state.
		slot = &DecodeSlot{}
		if st.BF16 {
			slot.K16 = make([]uint16, H*st.MaxT*dh)
			slot.V16 = make([]uint16, H*st.MaxT*dh)
		} else {
			slot.K = make([]float32, H*st.MaxT*dh)
			slot.V = make([]float32, H*st.MaxT*dh)
		}
		st.Slots[o.n.Name] = slot
	}
	if kn.Numel() != H*Tn*dh || vn.Numel() != H*Tn*dh {
		return nil, o.n.Errorf("SDPA(cache): K/V shape mismatch %v/%v", kn.Shape(), vn.Shape())
	}
	if st.Pos+Tn > st.MaxT {
		return nil, o.n.Errorf("SDPA(cache): %d+%d exceeds MaxT %d", st.Pos, Tn, st.MaxT)
	}
	qf, kf, vf := q.F32(), kn.F32(), vn.F32()
	// Append the new positions (converting once when the cache is bf16).
	for h := 0; h < H; h++ {
		if st.BF16 {
			if o.kLay == 1 { // kNew is [1,Tn,H,dh]
				for t := 0; t < Tn; t++ {
					gemm.BF16Row(slot.K16[(h*st.MaxT+st.Pos+t)*dh:(h*st.MaxT+st.Pos+t)*dh+dh], kf[(t*H+h)*dh:(t*H+h)*dh+dh])
				}
			} else {
				gemm.BF16Row(slot.K16[(h*st.MaxT+st.Pos)*dh:(h*st.MaxT+st.Pos+Tn)*dh], kf[h*Tn*dh:(h+1)*Tn*dh])
			}
			gemm.BF16Row(slot.V16[(h*st.MaxT+st.Pos)*dh:(h*st.MaxT+st.Pos+Tn)*dh], vf[h*Tn*dh:(h+1)*Tn*dh])
			continue
		}
		if o.kLay == 1 { // kNew is [1,Tn,H,dh]
			for t := 0; t < Tn; t++ {
				copy(slot.K[(h*st.MaxT+st.Pos+t)*dh:(h*st.MaxT+st.Pos+t)*dh+dh], kf[(t*H+h)*dh:(t*H+h)*dh+dh])
			}
		} else {
			copy(slot.K[(h*st.MaxT+st.Pos)*dh:], kf[h*Tn*dh:(h+1)*Tn*dh])
		}
		copy(slot.V[(h*st.MaxT+st.Pos)*dh:], vf[h*Tn*dh:(h+1)*Tn*dh])
	}
	out := ctx.NewUninit(tensor.F32, 1, H, Tn, dh)
	of := out.F32()
	workers := par.Workers()
	scratch := ctx.NewUninit(tensor.F32, workers, st.Pos+Tn)
	sAll := scratch.F32()
	grain := max(1, (1<<15)/max(1, 2*(st.Pos+Tn)*dh))
	par.For(H*Tn, grain, func(ht, wk int) {
		h, t := ht/Tn, ht%Tn
		kv := st.Pos + t + 1 // causal: attend over [0, Pos+t]
		row := sAll[wk*(st.Pos+Tn) : wk*(st.Pos+Tn)+kv]
		qrow := qf[(h*Tn+t)*dh : (h*Tn+t+1)*dh]
		if st.BF16 {
			K := slot.K16[h*st.MaxT*dh:]
			V := slot.V16[h*st.MaxT*dh:]
			for j := 0; j < kv; j++ {
				row[j] = o.scale * vek.DotBF16(qrow, K[j*dh:(j+1)*dh])
			}
			softmaxRow(row, row, false)
			dst := of[(h*Tn+t)*dh : (h*Tn+t+1)*dh]
			clear(dst)
			for j := 0; j < kv; j++ {
				if row[j] != 0 {
					vek.AxpyBF16(dst, V[j*dh:(j+1)*dh], row[j])
				}
			}
			return
		}
		K := slot.K[h*st.MaxT*dh:]
		V := slot.V[h*st.MaxT*dh:]
		// scores = scale·q·Kᵀ over the cached range (GEMV at Tn==1).
		gemm.SgemmTSerial(false, true, 1, kv, dh, o.scale, qrow, dh, K, dh, 0, row, kv)
		softmaxRow(row, row, false)
		gemm.SgemmTSerial(false, false, 1, dh, kv, 1, row, kv, V, dh, 0, of[(h*Tn+t)*dh:], dh)
	})
	if ctx.Pool != nil {
		ctx.Pool.Put(scratch)
	}
	return ctx.Out(out), nil
}
