package ops

import (
	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/tensor"
)

// convTransposeOp implements 2-D ConvTranspose (NCHW). Weight layout is
// [C_in, C_out/group, kH, kW]. Computed by scatter-accumulate, parallelised
// over output channels so writes never race.
type convTransposeOp struct {
	n         NodeInfo
	group     int
	strides   [2]int
	dilations [2]int
	pads      [4]int
	outPad    [2]int
	outShape  []int64
	autoPad   string
}

func buildConvTranspose(n NodeInfo) (Op, error) {
	o := &convTransposeOp{n: n, group: int(n.Attrs.Int("group", 1)), autoPad: n.Attrs.String("auto_pad", "NOTSET")}
	st := n.Attrs.Ints("strides", []int64{1, 1})
	di := n.Attrs.Ints("dilations", []int64{1, 1})
	pa := n.Attrs.Ints("pads", []int64{0, 0, 0, 0})
	op := n.Attrs.Ints("output_padding", []int64{0, 0})
	if len(st) != 2 || len(di) != 2 || len(pa) != 4 || len(op) != 2 {
		return nil, n.Errorf("only 2-D ConvTranspose supported")
	}
	o.strides = [2]int{int(st[0]), int(st[1])}
	o.dilations = [2]int{int(di[0]), int(di[1])}
	o.pads = [4]int{int(pa[0]), int(pa[1]), int(pa[2]), int(pa[3])}
	o.outPad = [2]int{int(op[0]), int(op[1])}
	o.outShape = n.Attrs.Ints("output_shape", nil)
	return o, nil
}

func (o *convTransposeOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("need X and W")
	}
	x, w := in[0], in[1]
	if x.DType() != tensor.F32 || w.DType() != tensor.F32 {
		return nil, o.n.Errorf("only f32")
	}
	xs, ws := x.Shape(), w.Shape()
	if len(xs) != 4 || len(ws) != 4 {
		return nil, o.n.Errorf("only 2-D (X %v, W %v)", xs, ws)
	}
	var bias []float32
	if len(in) > 2 && in[2] != nil {
		bias = in[2].F32()
	}
	N, Cin, H, W := xs[0], xs[1], xs[2], xs[3]
	G := o.group
	CoutG, KH, KW := ws[1], ws[2], ws[3]
	if ws[0] != Cin {
		return nil, o.n.Errorf("weight in-channels %d != X channels %d", ws[0], Cin)
	}
	Cout := CoutG * G
	CinG := Cin / G
	sh, sw := o.strides[0], o.strides[1]
	dh, dw := o.dilations[0], o.dilations[1]
	pt, pl, pb, pr := o.pads[0], o.pads[1], o.pads[2], o.pads[3]
	OH := (H-1)*sh - pt - pb + dh*(KH-1) + o.outPad[0] + 1
	OW := (W-1)*sw - pl - pr + dw*(KW-1) + o.outPad[1] + 1
	if len(o.outShape) == 4 {
		OH, OW = int(o.outShape[2]), int(o.outShape[3])
	}
	if OH <= 0 || OW <= 0 {
		return nil, o.n.Errorf("non-positive output %dx%d", OH, OW)
	}
	out := ctx.New(tensor.F32, N, Cout, OH, OW) // zeroed: we scatter-add
	xf, wf, of := x.F32(), w.F32(), out.F32()

	// Parallel over (n, out-channel): each task owns one output plane.
	par.For(N*Cout, 1, func(idx, _ int) {
		n := idx / Cout
		oc := idx % Cout
		g := oc / CoutG
		ocg := oc % CoutG
		op := of[(n*Cout+oc)*OH*OW : (n*Cout+oc+1)*OH*OW]
		if bias != nil {
			b := bias[oc]
			for i := range op {
				op[i] = b
			}
		}
		for icg := 0; icg < CinG; icg++ {
			ic := g*CinG + icg
			ip := xf[(n*Cin+ic)*H*W:]
			// weight for (ic, ocg): [KH, KW]
			wp := wf[((ic*CoutG)+ocg)*KH*KW:]
			for ih := 0; ih < H; ih++ {
				for iw := 0; iw < W; iw++ {
					v := ip[ih*W+iw]
					if v == 0 {
						continue
					}
					for kh := 0; kh < KH; kh++ {
						oh := ih*sh - pt + kh*dh
						if oh < 0 || oh >= OH {
							continue
						}
						wr := wp[kh*KW : kh*KW+KW]
						orow := op[oh*OW:]
						for kw := 0; kw < KW; kw++ {
							ow := iw*sw - pl + kw*dw
							if ow < 0 || ow >= OW {
								continue
							}
							orow[ow] += v * wr[kw]
						}
					}
				}
			}
		}
	})
	return []*tensor.Tensor{out}, nil
}

func init() {
	Register("", "ConvTranspose", 1, buildConvTranspose)
}
