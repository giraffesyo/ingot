package ops

import (
	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/vek"
	"github.com/giraffesyo/ingot/tensor"
)

// seOp is the fused squeeze-excite island produced by fuse-se:
// GlobalAveragePool → 1x1 Conv (+act) → 1x1 Conv (+act) → Mul back onto the
// input. The tiny FC chain runs inline between two parallel regions (pool,
// scale) instead of four ops' worth of dispatch and buffer traffic.
// Inputs: x, W1 [Cr,C,1,1], B1 (opt), W2 [C,Cr,1,1], B2 (opt).
type seOp struct {
	n    NodeInfo
	epi1 epilogue
	epi2 epilogue
}

// parseEpiloguePrefix reads an epilogue configured under prefixed attribute
// names (fused ops carry one per stage).
func parseEpiloguePrefix(n NodeInfo, prefix string) (epilogue, error) {
	a := n.Attrs
	e := epilogue{
		act:   a.String(prefix+"act", ""),
		alpha: a.Float(prefix+"act_alpha", 0),
		beta:  a.Float(prefix+"act_beta", 0),
	}
	switch e.act {
	case "", "relu", "hardswish", "hardsigmoid", "sigmoid", "silu", "clip", "leakyrelu":
	default:
		return e, n.Errorf("unknown %sact %q", prefix, e.act)
	}
	if a.Has(prefix+"post_scale") || a.Has(prefix+"post_shift") {
		e.post = true
		e.scale = a.Float(prefix+"post_scale", 1)
		e.shift = a.Float(prefix+"post_shift", 0)
	}
	return e, nil
}

func buildSE(n NodeInfo) (Op, error) {
	o := &seOp{n: n}
	var err error
	if o.epi1, err = parseEpiloguePrefix(n, "se1_"); err != nil {
		return nil, err
	}
	if o.epi2, err = parseEpiloguePrefix(n, "se2_"); err != nil {
		return nil, err
	}
	return o, nil
}

func (o *seOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 5 || in[0] == nil || in[1] == nil || in[3] == nil {
		return nil, o.n.Errorf("SE: need x, W1, (B1), W2, (B2)")
	}
	x, w1t, w2t := in[0], in[1], in[3]
	if x.DType() != tensor.F32 {
		return nil, o.n.Errorf("SE: want f32, got %s", x.DType())
	}
	xs := x.Shape()
	if len(xs) != 4 {
		return nil, o.n.Errorf("SE: want NCHW, got %v", xs)
	}
	N, C, H, W := xs[0], xs[1], xs[2], xs[3]
	Cr := w1t.Shape()[0]
	if w1t.Numel() != Cr*C || w2t.Numel() != C*Cr || w2t.Shape()[0] != C {
		return nil, o.n.Errorf("SE: weight shapes %v/%v vs C=%d", w1t.Shape(), w2t.Shape(), C)
	}
	var b1, b2 []float32
	if in[2] != nil {
		b1 = in[2].F32()
	}
	if len(in) > 4 && in[4] != nil {
		b2 = in[4].F32()
	}
	P := H * W
	out := ctx.NewUninit(tensor.F32, xs...)
	xf, of := x.F32(), out.F32()
	w1, w2 := w1t.F32(), w2t.F32()

	scr := ctx.NewUninit(tensor.F32, N*C+Cr)
	sm := scr.F32()[:N*C] // per-(n,c) means, then overwritten with scales
	z1 := scr.F32()[N*C:]

	inv := 1 / float32(P)
	par.For(N*C, max(1, unaryChunk/max(P, 1)), func(nc, _ int) {
		row := xf[nc*P : (nc+1)*P]
		var s0, s1, s2, s3 float32
		i := 0
		for ; i+4 <= len(row); i += 4 {
			s0 += row[i]
			s1 += row[i+1]
			s2 += row[i+2]
			s3 += row[i+3]
		}
		sum := (s0 + s1) + (s2 + s3)
		for ; i < len(row); i++ {
			sum += row[i]
		}
		sm[nc] = sum * inv
	})
	for n := 0; n < N; n++ {
		s := sm[n*C : (n+1)*C]
		for r := 0; r < Cr; r++ {
			v := vek.Dot(w1[r*C:(r+1)*C], s)
			if b1 != nil {
				v += b1[r]
			}
			z1[r] = v
		}
		o.epi1.apply(z1)
		// Second FC overwrites the mean slice in place: each output scale
		// depends only on z1.
		for c := 0; c < C; c++ {
			v := vek.Dot(w2[c*Cr:(c+1)*Cr], z1)
			if b2 != nil {
				v += b2[c]
			}
			s[c] = v
		}
		o.epi2.apply(s)
	}
	par.For(N*C, max(1, unaryChunk/max(P, 1)), func(nc, _ int) {
		vek.MulScalar(of[nc*P:(nc+1)*P], xf[nc*P:(nc+1)*P], sm[nc])
	})
	if ctx.Pool != nil {
		ctx.Pool.Put(scr)
	}
	return ctx.Out(out), nil
}

func init() {
	Register("ingot", "SE", 1, buildSE)
}
