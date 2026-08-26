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
	sT := ctx.NewUninit(tensor.F32, workers, T*T)
	sAll := sT.F32()
	par.For(B*H, 1, func(bh, wk int) {
		b, h := bh/H, bh%H
		off := (b*H + h) * plane
		q := xf[qBase+off : qBase+off+plane]
		k := xf[kBase+off : kBase+off+plane]
		v := xf[vBase+off : vBase+off+plane]
		s := sAll[wk*T*T : (wk+1)*T*T]
		// scores = scale · Q·Kᵀ (K stays row-major: the transposed-B GEMM).
		gemm.SgemmT(false, true, T, T, dh, o.scale, q, dh, k, dh, 0, s, T)
		for t := 0; t < T; t++ {
			softmaxRow(s[t*T:(t+1)*T], s[t*T:(t+1)*T], false)
		}
		// out[b, :, h, :] = probs·V, written straight into [B,T,H,dh] via ldc.
		gemm.SgemmT(false, false, T, dh, T, 1, s, T, v, dh, 0, of[(b*T*H+h)*dh:], H*dh)
	})
	if ctx.Pool != nil {
		ctx.Pool.Put(sT)
	}
	return []*tensor.Tensor{out}, nil
}

// sdpaOp is the generic fused scaled-dot-product-attention core produced by
// fuse-sdpa: scores = scale·A·B (A [B,H,T,dh], B [B,H,dh,T] — however the
// graph produced it), optional additive mask, row softmax in-tile, then
// probs·V ([B,H,T,dh]). With strideOut the result is written directly in
// [B,T,H,dh] order (folding the usual trailing transpose).
type sdpaOp struct {
	n         NodeInfo
	scale     float32
	strideOut bool
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
	B, H, T, dh := as[0], as[1], as[2], as[3]
	Tk := bs[3]
	if bs[0] != B || bs[1] != H || bs[2] != dh || vs[0] != B || vs[1] != H || vs[2] != Tk || vs[3] != dh {
		return nil, o.n.Errorf("SDPA: shape mismatch A%v B%v V%v", as, bs, vs)
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
	sT := ctx.NewUninit(tensor.F32, workers, T*Tk)
	sAll := sT.F32()
	par.For(B*H, 1, func(bh, wk int) {
		b, h := bh/H, bh%H
		ab := af[(b*H+h)*T*dh:]
		bb := bf[(b*H+h)*dh*Tk:]
		vb := vf[(b*H+h)*Tk*dh:]
		s := sAll[wk*T*Tk : (wk+1)*T*Tk]
		gemm.SgemmT(false, false, T, Tk, dh, o.scale, ab, dh, bb, Tk, 0, s, Tk)
		for t := 0; t < T; t++ {
			row := s[t*Tk : (t+1)*Tk]
			if mask != nil {
				vek.Add(row, row, mask[t*Tk:(t+1)*Tk])
			}
			softmaxRow(row, row, false)
		}
		if o.strideOut {
			gemm.SgemmT(false, false, T, dh, Tk, 1, s, Tk, vb, dh, 0, of[(b*T*H+h)*dh:], H*dh)
		} else {
			gemm.SgemmT(false, false, T, dh, Tk, 1, s, Tk, vb, dh, 0, of[(b*H+h)*T*dh:], dh)
		}
	})
	if ctx.Pool != nil {
		ctx.Pool.Put(sT)
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	Register("ingot", "MHA", 1, func(n NodeInfo) (Op, error) {
		return &mhaOp{n: n, scale: n.Attrs.Float("scale", 1)}, nil
	})
	Register("ingot", "SDPA", 1, func(n NodeInfo) (Op, error) {
		return &sdpaOp{n: n, scale: n.Attrs.Float("scale", 1), strideOut: n.Attrs.Int("stride_out", 0) != 0}, nil
	})
}
