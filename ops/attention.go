package ops

import (
	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/kernels/par"
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

func init() {
	Register("ingot", "MHA", 1, func(n NodeInfo) (Op, error) {
		return &mhaOp{n: n, scale: n.Attrs.Float("scale", 1)}, nil
	})
}
