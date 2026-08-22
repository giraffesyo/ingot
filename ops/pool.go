package ops

import (
	"math"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/tensor"
)

type poolOp struct {
	n               NodeInfo
	kind            string // "max" | "avg"
	kernel, strides [2]int
	pads            [4]int
	autoPad         string
	ceilMode        bool
	countIncludePad bool
}

func buildPool(kind string) Builder {
	return func(n NodeInfo) (Op, error) {
		ks := n.Attrs.Ints("kernel_shape", nil)
		if len(ks) != 2 {
			return nil, n.Errorf("only 2-D pooling supported (kernel_shape=%v)", ks)
		}
		st := n.Attrs.Ints("strides", []int64{1, 1})
		pa := n.Attrs.Ints("pads", []int64{0, 0, 0, 0})
		if di := n.Attrs.Ints("dilations", nil); di != nil && (di[0] != 1 || di[1] != 1) {
			return nil, n.Errorf("dilations not supported")
		}
		if len(st) != 2 || len(pa) != 4 {
			return nil, n.Errorf("bad strides/pads")
		}
		return &poolOp{
			n: n, kind: kind,
			kernel:          [2]int{int(ks[0]), int(ks[1])},
			strides:         [2]int{int(st[0]), int(st[1])},
			pads:            [4]int{int(pa[0]), int(pa[1]), int(pa[2]), int(pa[3])},
			autoPad:         n.Attrs.String("auto_pad", "NOTSET"),
			ceilMode:        n.Attrs.Int("ceil_mode", 0) == 1,
			countIncludePad: n.Attrs.Int("count_include_pad", 0) == 1,
		}, nil
	}
}

func (o *poolOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 1 || in[0] == nil || in[0].DType() != tensor.F32 {
		return nil, o.n.Errorf("need f32 input")
	}
	x := in[0]
	xs := x.Shape()
	if len(xs) != 4 {
		return nil, o.n.Errorf("only NCHW supported, got %v", xs)
	}
	N, C, H, W := xs[0], xs[1], xs[2], xs[3]
	KH, KW := o.kernel[0], o.kernel[1]
	sh, sw := o.strides[0], o.strides[1]
	pads := o.pads
	switch o.autoPad {
	case "NOTSET", "":
	case "VALID":
		pads = [4]int{}
	case "SAME_UPPER", "SAME_LOWER":
		for d, in := range [2]int{H, W} {
			k, s := o.kernel[d], o.strides[d]
			out := (in + s - 1) / s
			total := max(0, (out-1)*s+k-in)
			a := total / 2
			if o.autoPad == "SAME_LOWER" {
				a = (total + 1) / 2
			}
			pads[d], pads[d+2] = a, total-a
		}
	default:
		return nil, o.n.Errorf("unsupported auto_pad %q", o.autoPad)
	}
	outDim := func(in, k, s, pa, pb int) int {
		if o.ceilMode {
			v := (in+pa+pb-k+s-1)/s + 1
			// last window must start inside the (padded) input
			if (v-1)*s >= in+pa {
				v--
			}
			return v
		}
		return (in+pa+pb-k)/s + 1
	}
	OH := outDim(H, KH, sh, pads[0], pads[2])
	OW := outDim(W, KW, sw, pads[1], pads[3])
	out := ctx.NewUninit(tensor.F32, N, C, OH, OW)
	xf, of := x.F32(), out.F32()
	pt, pl := pads[0], pads[1]
	isMax := o.kind == "max"
	par.For(N*C, 1, func(nc, _ int) {
		xc := xf[nc*H*W : (nc+1)*H*W]
		oc := of[nc*OH*OW : (nc+1)*OH*OW]
		for oh := 0; oh < OH; oh++ {
			h0 := oh*sh - pt
			for ow := 0; ow < OW; ow++ {
				w0 := ow*sw - pl
				if isMax {
					m := float32(math.Inf(-1))
					for kh := 0; kh < KH; kh++ {
						ih := h0 + kh
						if ih < 0 || ih >= H {
							continue
						}
						row := xc[ih*W : (ih+1)*W]
						for kw := 0; kw < KW; kw++ {
							iw := w0 + kw
							if iw >= 0 && iw < W && row[iw] > m {
								m = row[iw]
							}
						}
					}
					oc[oh*OW+ow] = m
				} else {
					var sum float32
					cnt := 0
					for kh := 0; kh < KH; kh++ {
						ih := h0 + kh
						if ih < 0 || ih >= H {
							continue
						}
						row := xc[ih*W : (ih+1)*W]
						for kw := 0; kw < KW; kw++ {
							iw := w0 + kw
							if iw >= 0 && iw < W {
								sum += row[iw]
								cnt++
							}
						}
					}
					if o.countIncludePad {
						// window clipped to padded region (pads only, not beyond)
						hA, hB := max(h0, -pt), min(h0+KH, H+pads[2])
						wA, wB := max(w0, -pl), min(w0+KW, W+pads[3])
						cnt = (hB - hA) * (wB - wA)
					}
					if cnt > 0 {
						oc[oh*OW+ow] = sum / float32(cnt)
					}
				}
			}
		}
	})
	return []*tensor.Tensor{out}, nil
}

type globalPoolOp struct {
	n     NodeInfo
	isMax bool
}

func (o *globalPoolOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 1 || in[0] == nil || in[0].DType() != tensor.F32 {
		return nil, o.n.Errorf("need f32 input")
	}
	x := in[0]
	xs := x.Shape()
	if len(xs) < 3 {
		return nil, o.n.Errorf("need rank>=3, got %v", xs)
	}
	N, C := xs[0], xs[1]
	P := 1
	for _, d := range xs[2:] {
		P *= d
	}
	oshape := make(tensor.Shape, len(xs))
	oshape[0], oshape[1] = N, C
	for i := 2; i < len(xs); i++ {
		oshape[i] = 1
	}
	out := ctx.NewUninit(tensor.F32, oshape...)
	xf, of := x.F32(), out.F32()
	par.For(N*C, 8, func(nc, _ int) {
		src := xf[nc*P : (nc+1)*P]
		if o.isMax {
			m := src[0]
			for _, v := range src[1:] {
				if v > m {
					m = v
				}
			}
			of[nc] = m
			return
		}
		// 8 independent accumulators: a single chain is latency-bound.
		var s0, s1, s2, s3, s4, s5, s6, s7 float32
		i := 0
		for ; i+8 <= len(src); i += 8 {
			s0 += src[i]
			s1 += src[i+1]
			s2 += src[i+2]
			s3 += src[i+3]
			s4 += src[i+4]
			s5 += src[i+5]
			s6 += src[i+6]
			s7 += src[i+7]
		}
		sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
		for ; i < len(src); i++ {
			sum += src[i]
		}
		of[nc] = sum / float32(P)
	})
	return []*tensor.Tensor{out}, nil
}

func init() {
	Register("", "MaxPool", 1, buildPool("max"))
	Register("", "AveragePool", 1, buildPool("avg"))
	Register("", "GlobalAveragePool", 1, func(n NodeInfo) (Op, error) { return &globalPoolOp{n, false}, nil })
	Register("", "GlobalMaxPool", 1, func(n NodeInfo) (Op, error) { return &globalPoolOp{n, true}, nil })
}
