package ops

import (
	"sync"

	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/vek"
	"github.com/giraffesyo/ingot/tensor"
)

// convTransposeOp implements 2-D ConvTranspose (NCHW). Weight layout is
// [C_in, C_out/group, kH, kW]. Computed as GEMM (Wᵀ·X → column matrix) followed
// by col2im scatter, parallelised over output channels so writes never race.
type convTransposeOp struct {
	n         NodeInfo
	group     int
	strides   [2]int
	dilations [2]int
	pads      [4]int
	outPad    [2]int
	outShape  []int64
	autoPad   string
	epi       epilogue

	packMu  sync.Mutex
	packed  []*gemm.PackedA // per group: Wgᵀ [KK×CinG] pre-packed
	packSrc *float32
	packLen int
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
	var err error
	if o.epi, err = parseEpilogue(n); err != nil {
		return nil, err
	}
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
	HW := H * W
	KK := CoutG * KH * KW
	overlap := KH > sh || KW > sw || pt != 0 || pl != 0 || pb != 0 || pr != 0 || dh != 1 || dw != 1 || len(o.outShape) == 4
	var out *tensor.Tensor
	if overlap {
		out = ctx.New(tensor.F32, N, Cout, OH, OW) // zeroed: we scatter-add
	} else {
		out = ctx.NewUninit(tensor.F32, N, Cout, OH, OW) // every output written exactly once
	}
	xf, wf, of := x.F32(), w.F32(), out.F32()

	// GEMM + col2im (the standard formulation, cf. Caffe/ORT):
	//   col[(ocg*KH+kh)*KW+kw][ih*W+iw] = Σ_ic W[ic][ocg][kh][kw] · X[ic][ih][iw]
	// i.e. col = W_gᵀ[KK×CinG] · X_g[CinG×HW], then each col row is scattered
	// onto the output plane at stride s with offset (kh·d − pad).
	col := ctx.NewUninit(tensor.F32, KK, HW)
	cf := col.F32()
	var pk []*gemm.PackedA
	if gemm.PackFits(KK, CinG) {
		o.packMu.Lock()
		if o.packSrc != &wf[0] || o.packLen != len(wf) {
			o.packed = make([]*gemm.PackedA, G)
			for g := 0; g < G; g++ {
				o.packed[g] = gemm.PackA(true, KK, CinG, wf[g*CinG*KK:], KK)
			}
			o.packSrc, o.packLen = &wf[0], len(wf)
		}
		pk = o.packed
		o.packMu.Unlock()
	}
	for n := 0; n < N; n++ {
		for g := 0; g < G; g++ {
			if pk != nil {
				gemm.SgemmPackedA(pk[g], HW, xf[(n*Cin+g*CinG)*HW:], HW, 0, cf, HW, true)
			} else {
				gemm.SgemmT(true, false, KK, HW, CinG, 1, wf[g*CinG*KK:], KK, xf[(n*Cin+g*CinG)*HW:], HW, 0, cf, HW)
			}
			if !overlap {
				// Non-overlapping k≤s, no pad: out[ih*sh+kh][iw*sw+kw] = col + b. Each
				// output element is written exactly once, so tasks can be (channel,
				// input-row chunk) pairs: chunk c of channel ocg owns input rows
				// [c*rows, (c+1)*rows) and hence output rows [c*rows*sh, ...).
				rows := max(1, (32768 / max(1, KH*KW*W)))
				nChunks := (H + rows - 1) / rows
				par.For(CoutG*nChunks, 1, func(t, _ int) {
					ocg, ch := t/nChunks, t%nChunks
					ih0 := ch * rows
					ih1 := min(ih0+rows, H)
					oc := g*CoutG + ocg
					op := of[(n*Cout+oc)*OH*OW : (n*Cout+oc+1)*OH*OW]
					var b float32
					if bias != nil {
						b = bias[oc]
					}
					for kh := 0; kh < KH; kh++ {
						if sw == 2 && KW == 2 {
							// One interleave writes the whole output row.
							cr0 := cf[((ocg*KH+kh)*KW+0)*HW:]
							cr1 := cf[((ocg*KH+kh)*KW+1)*HW:]
							for ih := ih0; ih < ih1; ih++ {
								vek.Zip2(op[(ih*sh+kh)*OW:(ih*sh+kh)*OW+2*W], cr0[ih*W:(ih+1)*W], cr1[ih*W:(ih+1)*W], b)
							}
							continue
						}
						for kw := 0; kw < KW; kw++ {
							cr := cf[((ocg*KH+kh)*KW+kw)*HW:]
							for ih := ih0; ih < ih1; ih++ {
								orow := op[(ih*sh+kh)*OW+kw:]
								src := cr[ih*W : (ih+1)*W]
								if sw == 1 {
									vek.AddScalar(orow[:W], src, b)
									continue
								}
								for iw, v := range src {
									orow[iw*sw] = v + b
								}
							}
						}
					}
					// Output rows/cols not covered by any tap (KH<sh, KW<sw, or
					// output_padding) must still be written: fill with bias.
					if KH < sh || KW < sw || o.outPad != [2]int{} {
						for oh := ih0 * sh; oh < min(ih1*sh, OH); oh++ {
							kh := oh % sh
							orow := op[oh*OW : (oh+1)*OW]
							if kh < KH {
								for ow := 0; ow < OW; ow++ {
									if ow/sw >= W || ow%sw >= KW {
										orow[ow] = b
									}
								}
								continue
							}
							for i := range orow {
								orow[i] = b
							}
						}
						if ch == nChunks-1 {
							for oh := H * sh; oh < OH; oh++ {
								orow := op[oh*OW : (oh+1)*OW]
								for i := range orow {
									orow[i] = b
								}
							}
						}
					}
					o.epi.apply(op[ih0*sh*OW : min(ih1*sh, OH)*OW])
					if ch == nChunks-1 && H*sh < OH {
						o.epi.apply(op[H*sh*OW:])
					}
				})
				continue
			}
			// Overlapping taps: scatter-accumulate, one task per output plane so
			// writes never race.
			par.For(CoutG, 1, func(ocg, _ int) {
				oc := g*CoutG + ocg
				op := of[(n*Cout+oc)*OH*OW : (n*Cout+oc+1)*OH*OW]
				var b float32
				if bias != nil {
					b = bias[oc]
				}
				if bias != nil {
					for i := range op {
						op[i] = b
					}
				}
				for kh := 0; kh < KH; kh++ {
					for kw := 0; kw < KW; kw++ {
						cr := cf[((ocg*KH+kh)*KW+kw)*HW:]
						for ih := 0; ih < H; ih++ {
							oh := ih*sh - pt + kh*dh
							if oh < 0 || oh >= OH {
								continue
							}
							orow := op[oh*OW : (oh+1)*OW]
							src := cr[ih*W : (ih+1)*W]
							for iw, v := range src {
								ow := iw*sw - pl + kw*dw
								if ow >= 0 && ow < OW {
									orow[ow] += v
								}
							}
						}
					}
				}
				o.epi.apply(op)
			})
		}
	}
	if ctx.Pool != nil {
		ctx.Pool.Put(col)
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	Register("", "ConvTranspose", 1, buildConvTranspose)
}
