package ops

import (
	"math"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/tensor"
)

// snakeOp (ingot.Snake) is the periodic activation of BigVGAN-family
// vocoders over channel-first x [N, C, …]:
//
//	y = x + scale[c] · sin²(freq[c] · x)
//
// Snake uses scale = 1/freq; SnakeBeta learns both in log space, which the
// model folds at build time (freq = e^α, scale = 1/(e^β + ε)). One pass
// replaces the broadcast Mul/Sin/Mul/Mul/Add chain.
type snakeOp struct{ n NodeInfo }

func (o *snakeOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) != 3 || in[0] == nil || in[1] == nil || in[2] == nil {
		return nil, o.n.Errorf("need x, freq, scale")
	}
	x, fr, sc := in[0], in[1], in[2]
	if x.DType() != tensor.F32 || fr.DType() != tensor.F32 || sc.DType() != tensor.F32 {
		return nil, o.n.Errorf("want f32 inputs")
	}
	xs := x.Shape()
	if len(xs) < 2 {
		return nil, o.n.Errorf("x rank %d < 2", len(xs))
	}
	C := xs[1]
	if fr.Numel() != C || sc.Numel() != C {
		return nil, o.n.Errorf("x %v needs freq/scale [%d], got %v %v", xs, C, fr.Shape(), sc.Shape())
	}
	inner := x.Numel() / max(1, xs[0]*C)
	out := ctx.NewUninit(tensor.F32, xs...)
	xf, ff, sf, of := x.F32(), fr.F32(), sc.F32(), out.F32()
	grain := max(1, 16384/max(inner, 1))
	par.For(xs[0]*C, grain, func(r, _ int) {
		a, b := ff[r%C], sf[r%C]
		src, dst := xf[r*inner:(r+1)*inner], of[r*inner:(r+1)*inner]
		for i, v := range src {
			s := float32(math.Sin(float64(v * a)))
			dst[i] = v + b*(s*s)
		}
	})
	return ctx.Out(out), nil
}

func init() {
	Register("ingot", "Snake", 1, func(n NodeInfo) (Op, error) { return &snakeOp{n: n}, nil })
}
