package ops

import (
	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/tensor"
)

// ropeOp (ingot.RoPE) rotates x [T, …, dh] by per-token angles: cos, sin
// [T, dh/2] are shared by every row (head) of token t. layout 0 rotates
// interleaved channel pairs (2j, 2j+1) — the complex view; 1 rotates
// (j, j+dh/2) — rotate_half. One pass replaces the reshape/split/mul/add/
// concat chain an exporter emits.
type ropeOp struct {
	n    NodeInfo
	half bool
}

func (o *ropeOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) != 3 || in[0] == nil || in[1] == nil || in[2] == nil {
		return nil, o.n.Errorf("need x, cos, sin")
	}
	x, c, s := in[0], in[1], in[2]
	if x.DType() != tensor.F32 || c.DType() != tensor.F32 || s.DType() != tensor.F32 {
		return nil, o.n.Errorf("want f32 inputs")
	}
	xs := x.Shape()
	if len(xs) < 2 {
		return nil, o.n.Errorf("x rank %d < 2", len(xs))
	}
	T, dh := xs[0], xs[len(xs)-1]
	if dh%2 != 0 || c.Numel() != T*dh/2 || s.Numel() != T*dh/2 {
		return nil, o.n.Errorf("x %v needs cos/sin [%d, %d], got %v %v", xs, T, dh/2, c.Shape(), s.Shape())
	}
	rows := x.Numel() / (T * dh) // rows (heads) per token
	out := ctx.NewUninit(tensor.F32, xs...)
	xf, cf, sf, of := x.F32(), c.F32(), s.F32(), out.F32()
	hd := dh / 2
	grain := max(1, 16384/max(rows*dh, 1))
	par.For(T, grain, func(t, _ int) {
		ct, st := cf[t*hd:(t+1)*hd], sf[t*hd:(t+1)*hd]
		for r := range rows {
			base := (t*rows + r) * dh
			src, dst := xf[base:base+dh], of[base:base+dh]
			if o.half {
				for j := range hd {
					a, b := src[j], src[j+hd]
					dst[j] = a*ct[j] - b*st[j]
					dst[j+hd] = b*ct[j] + a*st[j]
				}
				continue
			}
			for j := range hd {
				a, b := src[2*j], src[2*j+1]
				dst[2*j] = a*ct[j] - b*st[j]
				dst[2*j+1] = a*st[j] + b*ct[j]
			}
		}
	})
	return ctx.Out(out), nil
}

func init() {
	Register("ingot", "RoPE", 1, func(n NodeInfo) (Op, error) {
		return &ropeOp{n: n, half: n.Attrs.Int("layout", 0) == 1}, nil
	})
}
