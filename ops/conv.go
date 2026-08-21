package ops

import (
	"fmt"
	"sync"

	"github.com/giraffesyo/ocr/kernels/gemm"
	"github.com/giraffesyo/ocr/kernels/par"
	"github.com/giraffesyo/ocr/kernels/vek"
	"github.com/giraffesyo/ocr/tensor"
)

// convOp implements 2-D Conv (NCHW) via im2col + GEMM, with fast paths for
// 1×1/stride-1/no-pad (GEMM directly on the input) and depthwise (direct).
type convOp struct {
	n         NodeInfo
	group     int
	dwPadded  []float32 // lazily packed padded depthwise weights (C * paddedK)
	dwOnce    sync.Once
	strides   [2]int
	dilations [2]int
	pads      [4]int // top, left, bottom, right
	autoPad   string
	kshape    []int64 // may be nil (infer from W)
}

func buildConv(n NodeInfo) (Op, error) {
	o := &convOp{n: n, group: int(n.Attrs.Int("group", 1)), autoPad: n.Attrs.String("auto_pad", "NOTSET")}
	o.kshape = n.Attrs.Ints("kernel_shape", nil)
	st := n.Attrs.Ints("strides", []int64{1, 1})
	di := n.Attrs.Ints("dilations", []int64{1, 1})
	pa := n.Attrs.Ints("pads", []int64{0, 0, 0, 0})
	if len(st) != 2 || len(di) != 2 || len(pa) != 4 {
		return nil, n.Errorf("only 2-D conv supported (strides=%v dilations=%v pads=%v)", st, di, pa)
	}
	o.strides = [2]int{int(st[0]), int(st[1])}
	o.dilations = [2]int{int(di[0]), int(di[1])}
	o.pads = [4]int{int(pa[0]), int(pa[1]), int(pa[2]), int(pa[3])}
	return o, nil
}

// convGeom resolves padding/output size for one spatial dim.
func convOut(in, k, stride, dil, padA, padB int) int {
	eff := dil*(k-1) + 1
	return (in+padA+padB-eff)/stride + 1
}

func (o *convOp) resolvePads(h, w, kh, kw int) (pads [4]int, err error) {
	pads = o.pads
	switch o.autoPad {
	case "NOTSET", "":
	case "VALID":
		pads = [4]int{}
	case "SAME_UPPER", "SAME_LOWER":
		for d, in := range [2]int{h, w} {
			k, s, dl := [2]int{kh, kw}[d], o.strides[d], o.dilations[d]
			out := (in + s - 1) / s
			total := max(0, (out-1)*s+dl*(k-1)+1-in)
			a := total / 2
			if o.autoPad == "SAME_LOWER" {
				a = (total + 1) / 2
			}
			pads[d], pads[d+2] = a, total-a
		}
	default:
		return pads, fmt.Errorf("unsupported auto_pad %q", o.autoPad)
	}
	return pads, nil
}

func (o *convOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("need X and W")
	}
	x, w := in[0], in[1]
	var bias []float32
	if len(in) > 2 && in[2] != nil {
		bias = in[2].F32()
	}
	if x.DType() != tensor.F32 || w.DType() != tensor.F32 {
		return nil, o.n.Errorf("only f32 supported")
	}
	xs, ws := x.Shape(), w.Shape()
	if len(xs) != 4 || len(ws) != 4 {
		return nil, o.n.Errorf("only 2-D conv supported (X %v, W %v)", xs, ws)
	}
	N, C, H, W := xs[0], xs[1], xs[2], xs[3]
	M, Cg, KH, KW := ws[0], ws[1], ws[2], ws[3]
	G := o.group
	if C != Cg*G || M%G != 0 {
		return nil, o.n.Errorf("channel mismatch: X C=%d, W [%d,%d], group=%d", C, M, Cg, G)
	}
	pads, err := o.resolvePads(H, W, KH, KW)
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	OH := convOut(H, KH, o.strides[0], o.dilations[0], pads[0], pads[2])
	OW := convOut(W, KW, o.strides[1], o.dilations[1], pads[1], pads[3])
	if OH <= 0 || OW <= 0 {
		return nil, o.n.Errorf("non-positive output size %dx%d", OH, OW)
	}
	out := ctx.NewUninit(tensor.F32, N, M, OH, OW)
	xf, wf, of := x.F32(), w.F32(), out.F32()
	Mg := M / G
	K := Cg * KH * KW
	P := OH * OW

	depthwise := G == C && Cg == 1 && Mg == 1
	pointwise := KH == 1 && KW == 1 && o.strides == [2]int{1, 1} && pads == [4]int{} && o.dilations == [2]int{1, 1}
	dwFast := depthwise && o.strides == [2]int{1, 1} && o.dilations == [2]int{1, 1} && KH == KW && (KH == 3 || KH == 5)

	switch {
	case dwFast:
		o.depthwiseS1(ctx, xf, wf, bias, of, N, C, H, W, KH, OH, OW, pads)
	case depthwise:
		o.depthwise(xf, wf, bias, of, N, C, H, W, KH, KW, OH, OW, pads)
	case pointwise:
		// out[n][g] = W_g[Mg×K] · X[n][g][K×P]
		for n := 0; n < N; n++ {
			for g := 0; g < G; g++ {
				gemm.Sgemm(Mg, P, K, 1, wf[g*Mg*K:], K, xf[(n*C+g*Cg)*H*W:], P, 0, of[(n*M+g*Mg)*P:], P)
			}
		}
		addBias(of, bias, N, M, P)
	default:
		col := ctx.New(tensor.F32, K, P)
		cf := col.F32()
		for n := 0; n < N; n++ {
			for g := 0; g < G; g++ {
				o.im2col(xf[(n*C+g*Cg)*H*W:], cf, Cg, H, W, KH, KW, OH, OW, pads)
				gemm.Sgemm(Mg, P, K, 1, wf[g*Mg*K:], K, cf, P, 0, of[(n*M+g*Mg)*P:], P)
			}
		}
		if ctx.Pool != nil {
			ctx.Pool.Put(col)
		}
		addBias(of, bias, N, M, P)
	}
	return []*tensor.Tensor{out}, nil
}

func addBias(of, bias []float32, N, M, P int) {
	if bias == nil {
		return
	}
	for n := 0; n < N; n++ {
		for m := 0; m < M; m++ {
			b := bias[m]
			row := of[(n*M+m)*P : (n*M+m+1)*P]
			for i := range row {
				row[i] += b
			}
		}
	}
}

// im2col writes col[(c*KH+kh)*KW+kw][oh*OW+ow] = x[c][oh*s+kh*d-pt][ow*s+kw*d-pl] (0 outside).
// Parallel over the K rows.
func (o *convOp) im2col(x, col []float32, C, H, W, KH, KW, OH, OW int, pads [4]int) {
	sh, sw := o.strides[0], o.strides[1]
	dh, dw := o.dilations[0], o.dilations[1]
	pt, pl := pads[0], pads[1]
	P := OH * OW
	K := C * KH * KW
	grain := max(1, 8192/max(P, 1))
	par.For(K, grain, func(k, _ int) {
		c := k / (KH * KW)
		kh := (k / KW) % KH
		kw := k % KW
		xc := x[c*H*W : (c+1)*H*W]
		row := col[k*P : (k+1)*P]
		for oh := 0; oh < OH; oh++ {
			ih := oh*sh + kh*dh - pt
			dst := row[oh*OW : (oh+1)*OW]
			if ih < 0 || ih >= H {
				clear(dst)
				continue
			}
			src := xc[ih*W : (ih+1)*W]
			if sw == 1 && dw == 1 {
				// contiguous window with edge clipping
				start := kw - pl // iw at ow=0
				for ow := 0; ow < OW; ow++ {
					iw := start + ow
					if iw < 0 || iw >= W {
						dst[ow] = 0
					} else {
						dst[ow] = src[iw]
					}
				}
				continue
			}
			for ow := 0; ow < OW; ow++ {
				iw := ow*sw + kw*dw - pl
				if iw < 0 || iw >= W {
					dst[ow] = 0
				} else {
					dst[ow] = src[iw]
				}
			}
		}
	})
}

// depthwise: one filter per channel. For each output row the kernel taps are
// applied as clipped row-axpys (out[ow0:ow1] += w * x[ih][...]), so the inner
// loops are branch-free and stride-1 for the common stride-1 case.
// Parallel over (n, c) with grain chosen so each task is a few µs.
func (o *convOp) depthwise(x, w, bias, out []float32, N, C, H, W, KH, KW, OH, OW int, pads [4]int) {
	sh, sw := o.strides[0], o.strides[1]
	dh, dw := o.dilations[0], o.dilations[1]
	pt, pl := pads[0], pads[1]
	// Per-kw valid output column range [lo, hi) such that 0 <= iw < W.
	type span struct{ lo, hi, off int }
	spans := make([]span, KW)
	for kw := 0; kw < KW; kw++ {
		off := kw*dw - pl // iw = ow*sw + off
		lo := 0
		if off < 0 {
			lo = (-off + sw - 1) / sw
		}
		hi := OW
		if m := (W - off + sw - 1) / sw; m < hi { // ow*sw+off < W
			hi = max(m, 0)
		}
		spans[kw] = span{lo, max(hi, lo), off}
	}
	work := OH * OW * KH * KW
	grain := max(1, 20000/max(work, 1))
	par.For(N*C, grain, func(nc, _ int) {
		c := nc % C
		xc := x[nc*H*W : (nc+1)*H*W]
		wc := w[c*KH*KW : (c+1)*KH*KW]
		oc := out[nc*OH*OW : (nc+1)*OH*OW]
		var b float32
		if bias != nil {
			b = bias[c]
		}
		for oh := 0; oh < OH; oh++ {
			row := oc[oh*OW : (oh+1)*OW]
			for i := range row {
				row[i] = b
			}
			for kh := 0; kh < KH; kh++ {
				ih := oh*sh + kh*dh - pt
				if ih < 0 || ih >= H {
					continue
				}
				xr := xc[ih*W : (ih+1)*W]
				for kw := 0; kw < KW; kw++ {
					sp := spans[kw]
					if sp.hi <= sp.lo {
						continue
					}
					wv := wc[kh*KW+kw]
					dst := row[sp.lo:sp.hi]
					if sw == 1 {
						vek.Axpy(dst, xr[sp.lo+sp.off:sp.hi+sp.off], wv)
					} else {
						base := sp.lo*sw + sp.off
						for i := range dst {
							dst[i] += wv * xr[base+i*sw]
						}
					}
				}
			}
		}
	})
}

// depthwiseS1 is a stride-1, dilation-1 KxK (K in {3,5}) depthwise conv. Each
// input channel is copied once into a zero-padded scratch plane so every output
// row is fully in-bounds and the NEON row kernel (vek.DwRowKxKS1) covers the
// whole output — no scalar border. Parallel over (n, c); each worker reuses its
// own padded plane (borders stay zero across channels, only the interior is
// rewritten).
func (o *convOp) depthwiseS1(ctx *Ctx, x, w, bias, out []float32, N, C, H, W, K, OH, OW int, pads [4]int) {
	pt, pl, pb, pr := pads[0], pads[1], pads[2], pads[3]
	Hp, Wp := H+pt+pb, W+pl+pr
	paddedK := ((K*K + 3) / 4) * 4
	o.dwOnce.Do(func() {
		o.dwPadded = make([]float32, C*paddedK)
		for c := 0; c < C; c++ {
			copy(o.dwPadded[c*paddedK:c*paddedK+K*K], w[c*K*K:(c+1)*K*K])
		}
	})
	workers := par.Workers()
	scratch := ctx.New(tensor.F32, workers, Hp, Wp) // zeroed: borders stay 0
	sf := scratch.F32()
	plane := Hp * Wp
	grain := max(1, 20000/max(OH*OW*K*K, 1))
	par.For(N*C, grain, func(nc, wk int) {
		c := nc % C
		xc := x[nc*H*W : (nc+1)*H*W]
		oc := out[nc*OH*OW : (nc+1)*OH*OW]
		pad := sf[wk*plane : (wk+1)*plane]
		// Copy this channel into the padded interior.
		for i := 0; i < H; i++ {
			copy(pad[(pt+i)*Wp+pl:(pt+i)*Wp+pl+W], xc[i*W:(i+1)*W])
		}
		var b float32
		if bias != nil {
			b = bias[c]
		}
		wp := o.dwPadded[c*paddedK : c*paddedK+paddedK]
		ncols := OW &^ 3
		for oh := 0; oh < OH; oh++ {
			row := oc[oh*OW : (oh+1)*OW]
			for i := range row {
				row[i] = b
			}
			src := pad[oh*Wp:] // output col 0 reads padded[(oh)*Wp + 0 ..]
			if K == 3 {
				vek.DwRow3x3S1(row, src, wp, ncols, Wp)
			} else {
				vek.DwRow5x5S1(row, src, wp, ncols, Wp)
			}
			// <4 column remainder against the padded plane
			for cc := ncols; cc < OW; cc++ {
				var acc float32
				for kh := 0; kh < K; kh++ {
					for kw := 0; kw < K; kw++ {
						acc += wp[kh*K+kw] * src[kh*Wp+cc+kw]
					}
				}
				row[cc] += acc
			}
		}
	})
}

func init() {
	Register("", "Conv", 1, buildConv)
}
