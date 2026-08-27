package ops

import (
	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/vek"
	"github.com/giraffesyo/ingot/tensor"
)

// mhaOp is the fused multi-head attention produced by the graph optimizer's
// fuse-attention pass: input is packed QKV [3, B, H, T, dh]; the op computes
// softmax(scale·Q·Kᵀ)·V per (batch, head) and writes the result directly in
// [B, T, H, dh] order (the final transpose of the exported pattern folds into
// the output GEMM's leading dimension).
type mhaOp struct {
	n     NodeInfo
	scale float32
}

func (o *mhaOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 1 || in[0] == nil {
		return nil, o.n.Errorf("MHA: missing input")
	}
	x := in[0]
	xs := x.Shape()
	if x.DType() != tensor.F32 || len(xs) != 5 || xs[0] != 3 {
		return nil, o.n.Errorf("MHA: want f32 [3,B,H,T,dh], got %s %v", x.DType(), xs)
	}
	B, H, T, dh := xs[1], xs[2], xs[3], xs[4]
	out := ctx.NewUninit(tensor.F32, B, T, H, dh)
	xf, of := x.F32(), out.F32()
	plane := T * dh
	qBase, kBase, vBase := 0, B*H*plane, 2*B*H*plane
	workers := par.Workers()
	rows, nRow := attnRows(B, H, T, T, dh, workers)
	gemmT := gemm.SgemmTSerial
	if nRow == 1 {
		gemmT = gemm.SgemmT // untiled: let big per-head GEMMs fan out
	}
	sT := ctx.NewUninit(tensor.F32, workers, rows*T)
	sAll := sT.F32()
	par.For(B*H*nRow, attnGrain(rows, T, dh), func(task, wk int) {
		bh, rc := task/nRow, task%nRow
		b, h := bh/H, bh%H
		t0 := rc * rows
		tc := min(rows, T-t0)
		off := (b*H + h) * plane
		q := xf[qBase+off+t0*dh:]
		k := xf[kBase+off : kBase+off+plane]
		v := xf[vBase+off : vBase+off+plane]
		s := sAll[wk*rows*T:]
		// scores = scale · Q·Kᵀ (K stays row-major: the transposed-B GEMM).
		gemmT(false, true, tc, T, dh, o.scale, q, dh, k, dh, 0, s, T)
		for t := 0; t < tc; t++ {
			softmaxRow(s[t*T:(t+1)*T], s[t*T:(t+1)*T], false)
		}
		// out[b, t0:, h, :] = probs·V, written straight into [B,T,H,dh] via ldc.
		gemmT(false, false, tc, dh, T, 1, s, T, v, dh, 0, of[((b*T+t0)*H+h)*dh:], H*dh)
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
	aLay, bLay, vLay int // input layouts, see fuse-sdpa (0 = as-declared)
}

func (o *sdpaOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 3 || in[0] == nil || in[1] == nil || in[2] == nil {
		return nil, o.n.Errorf("SDPA: need A, B, V")
	}
	a, bm, v := in[0], in[1], in[2]
	var mask []float32
	as, bs, vs := a.Shape(), bm.Shape(), v.Shape()
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
	} else {
		out = ctx.NewUninit(tensor.F32, B, H, T, dh)
	}
	af, bf, vf, of := a.F32(), bm.F32(), v.F32(), out.F32()
	workers := par.Workers()
	rows, nRow := attnRows(B, H, T, Tk, dh, workers)
	gemmT := gemm.SgemmTSerial
	if nRow == 1 {
		gemmT = gemm.SgemmT // untiled: let big per-head GEMMs fan out
	}
	sT := ctx.NewUninit(tensor.F32, workers, rows*Tk)
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
	// enough to amortise re-packing K each tile (~2M MACs before halving).
	for b*h*((t+rows-1)/rows) < 2*workers && rows > 32 && rows*tk*dh >= 1<<21 {
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
		return &mhaOp{n: n, scale: n.Attrs.Float("scale", 1)}, nil
	})
	Register("ingot", "SDPA", 1, func(n NodeInfo) (Op, error) {
		return &sdpaOp{n: n, scale: n.Attrs.Float("scale", 1), strideOut: n.Attrs.Int("stride_out", 0) != 0,
			aLay: int(n.Attrs.Int("a_layout", 0)), bLay: int(n.Attrs.Int("b_layout", 0)), vLay: int(n.Attrs.Int("v_layout", 0))}, nil
	})
}
