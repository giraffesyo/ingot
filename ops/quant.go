// ONNX quantization ops: QuantizeLinear, DequantizeLinear,
// DynamicQuantizeLinear, QLinearConv, QLinearMatMul, MatMulInteger.
//
// Supported subset (loud errors otherwise): activations u8 or s8, weights s8
// with zero_point 0 (what onnxruntime's quantizer emits: asymmetric
// activations, symmetric weights, optionally per-output-channel weight
// scales). The integer core is Σ xs8·w computed by the SMMLA GEMM
// (gemm.QgemmPackedS8) on activations shifted to s8 (u8 → x−128, with the
// zero point adjusted), plus a per-output-channel correction
// (zx′·ΣW − bias); requantization multiplies by x_scale·w_scale/y_scale in
// float32 and rounds half-to-even, matching onnxruntime within 1 quantum.
package ops

import (
	"math"
	"sync"

	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/vek"
	"github.com/giraffesyo/ingot/tensor"
)

// ---- element helpers ----

func satU8(v float32) uint8 {
	r := float32(math.RoundToEven(float64(v)))
	if r <= 0 {
		return 0
	}
	if r >= 255 {
		return 255
	}
	return uint8(r)
}

func satI8(v float32) int8 {
	r := float32(math.RoundToEven(float64(v)))
	if r <= -128 {
		return -128
	}
	if r >= 127 {
		return 127
	}
	return int8(r)
}

// quantParams reads a (scale, zero_point) input pair; zero_point may be nil.
// Returns per-element slices (len 1 for per-tensor) and the quantized dtype.
func quantParams(scale, zp *tensor.Tensor) (sc []float32, zpi []int32, dt tensor.DType, err error) {
	sc = scale.F32()
	dt = tensor.U8
	if zp == nil {
		return sc, []int32{0}, dt, nil
	}
	dt = zp.DType()
	switch dt {
	case tensor.U8:
		for _, v := range zp.U8() {
			zpi = append(zpi, int32(v))
		}
	case tensor.I8:
		for _, v := range zp.I8() {
			zpi = append(zpi, int32(v))
		}
	case tensor.I32:
		zpi = append(zpi, zp.I32()...)
	default:
		return nil, nil, dt, errUnsupported
	}
	return sc, zpi, dt, nil
}

var errUnsupported = &opError{"unsupported quantization parameter dtype"}

type opError struct{ s string }

func (e *opError) Error() string { return e.s }

// ---- QuantizeLinear ----

type quantizeLinearOp struct {
	n    NodeInfo
	axis int
}

func (o *quantizeLinearOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("need x and scale")
	}
	x := in[0]
	if x.DType() != tensor.F32 {
		return nil, o.n.Errorf("only f32 input (got %s)", x.DType())
	}
	var zp *tensor.Tensor
	if len(in) > 2 {
		zp = in[2]
	}
	sc, zpi, dt, err := quantParams(in[1], zp)
	if err != nil || dt == tensor.I32 {
		return nil, o.n.Errorf("unsupported zero_point dtype")
	}
	out := ctx.NewUninit(dt, x.Shape()...)
	xf := x.F32()
	if len(sc) == 1 {
		s0 := sc[0]
		z := float32(zpi[0])
		chunks := max(1, (len(xf)+unaryChunk-1)/unaryChunk)
		par.For(chunks, 1, func(c, _ int) {
			lo := c * unaryChunk
			hi := min(lo+unaryChunk, len(xf))
			if out.DType() == tensor.U8 {
				vek.QuantU8(out.U8()[lo:hi], xf[lo:hi], s0, z)
			} else {
				vek.QuantI8(out.I8()[lo:hi], xf[lo:hi], s0, z)
			}
		})
		return []*tensor.Tensor{out}, nil
	}
	// per-axis
	xs := x.Shape()
	axis, err2 := normAxis(o.axis, len(xs))
	if err2 != nil || xs[axis] != len(sc) {
		return nil, o.n.Errorf("per-axis scale len %d vs dim %v axis %d", len(sc), xs, o.axis)
	}
	inner := 1
	for _, d := range xs[axis+1:] {
		inner *= d
	}
	quantAll(out, xf, func(i int) (float32, float32) {
		c := (i / inner) % len(sc)
		z := float32(0)
		if len(zpi) > 1 {
			z = float32(zpi[c])
		} else if len(zpi) == 1 {
			z = float32(zpi[0])
		}
		return sc[c], z
	})
	return []*tensor.Tensor{out}, nil
}

// quantAll divides by the scale (not multiply-by-reciprocal: the spec is
// x/scale, and the reciprocal is off by an ulp exactly at .5 boundaries).
func quantAll(out *tensor.Tensor, xf []float32, prm func(i int) (scale, z float32)) {
	chunks := max(1, (len(xf)+unaryChunk-1)/unaryChunk)
	par.For(chunks, 1, func(c, _ int) {
		lo := c * unaryChunk
		hi := min(lo+unaryChunk, len(xf))
		if out.DType() == tensor.U8 {
			of := out.U8()
			for i := lo; i < hi; i++ {
				s, z := prm(i)
				of[i] = satU8(xf[i]/s + z)
			}
			return
		}
		of := out.I8()
		for i := lo; i < hi; i++ {
			s, z := prm(i)
			of[i] = satI8(xf[i]/s + z)
		}
	})
}

// ---- DequantizeLinear ----

type dequantizeLinearOp struct {
	n    NodeInfo
	axis int
}

func (o *dequantizeLinearOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("need x and scale")
	}
	x := in[0]
	var zp *tensor.Tensor
	if len(in) > 2 {
		zp = in[2]
	}
	sc, zpi, _, err := quantParams(in[1], zp)
	if err != nil {
		return nil, o.n.Errorf("unsupported zero_point dtype")
	}
	get, n, derr := intGetter(x)
	if derr != nil {
		return nil, o.n.Errorf("unsupported input dtype %s", x.DType())
	}
	out := ctx.NewUninit(tensor.F32, x.Shape()...)
	of := out.F32()
	if len(sc) == 1 {
		s := sc[0]
		z := zpi[0]
		chunks := max(1, (n+unaryChunk-1)/unaryChunk)
		par.For(chunks, 1, func(c, _ int) {
			lo := c * unaryChunk
			hi := min(lo+unaryChunk, n)
			switch x.DType() {
			case tensor.U8:
				vek.DequantU8(of[lo:hi], x.U8()[lo:hi], s, z)
			case tensor.I8:
				vek.DequantI8(of[lo:hi], x.I8()[lo:hi], s, z)
			default:
				for i := lo; i < hi; i++ {
					of[i] = float32(get(i)-z) * s
				}
			}
		})
		return []*tensor.Tensor{out}, nil
	}
	xs := x.Shape()
	axis, err2 := normAxis(o.axis, len(xs))
	if err2 != nil || xs[axis] != len(sc) {
		return nil, o.n.Errorf("per-axis scale len %d vs dims %v axis %d", len(sc), xs, o.axis)
	}
	inner := 1
	for _, d := range xs[axis+1:] {
		inner *= d
	}
	for i := 0; i < n; i++ {
		c := (i / inner) % len(sc)
		z := zpi[0]
		if len(zpi) > 1 {
			z = zpi[c]
		}
		of[i] = float32(get(i)-z) * sc[c]
	}
	return []*tensor.Tensor{out}, nil
}

func intGetter(x *tensor.Tensor) (func(int) int32, int, error) {
	switch x.DType() {
	case tensor.U8:
		v := x.U8()
		return func(i int) int32 { return int32(v[i]) }, len(v), nil
	case tensor.I8:
		v := x.I8()
		return func(i int) int32 { return int32(v[i]) }, len(v), nil
	case tensor.I32:
		v := x.I32()
		return func(i int) int32 { return v[i] }, len(v), nil
	}
	return nil, 0, errUnsupported
}

// ---- DynamicQuantizeLinear ----

type dynamicQuantizeLinearOp struct{ n NodeInfo }

func (o *dynamicQuantizeLinearOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	if x.DType() != tensor.F32 {
		return nil, o.n.Errorf("only f32")
	}
	xf := x.F32()
	lo, hi := float32(0), float32(0) // range always includes 0 per spec
	for _, v := range xf {
		lo = min(lo, v)
		hi = max(hi, v)
	}
	scale := (hi - lo) / 255
	if scale == 0 {
		scale = 1
	}
	zpf := -lo / scale
	zp := satU8(zpf)
	out := ctx.NewUninit(tensor.U8, x.Shape()...)
	of := out.U8()
	z := float32(zp)
	for i, v := range xf {
		of[i] = satU8(v/scale + z)
	}
	st := ctx.NewUninit(tensor.F32)
	st.F32()[0] = scale
	zt := ctx.NewUninit(tensor.U8)
	zt.U8()[0] = zp
	return []*tensor.Tensor{out, st, zt}, nil
}

// ---- integer GEMM core (shared by QLinearConv / QLinearMatMul / MatMulInteger) ----

// s8Weights validates a weight tensor + zero point for the SMMLA core:
// s8 with zero_point 0 (onnxruntime's symmetric weight quantization).
func s8Weights(n NodeInfo, w, wzp *tensor.Tensor) ([]int8, error) {
	if w.DType() != tensor.I8 {
		return nil, n.Errorf("only s8 weights supported (got %s); u8 weights are not emitted by standard quantizers", w.DType())
	}
	if wzp != nil {
		switch wzp.DType() {
		case tensor.I8:
			for _, v := range wzp.I8() {
				if v != 0 {
					return nil, n.Errorf("non-zero weight zero_point unsupported (symmetric s8 weights only)")
				}
			}
		default:
			return nil, n.Errorf("weight zero_point dtype %s unsupported", wzp.DType())
		}
	}
	return w.I8(), nil
}

// shiftToS8 converts an activation tensor to s8 with its zero point adjusted:
// u8 x → x−128 (zx′ = zx−128); s8 passes through.
func shiftToS8(dst []int8, x *tensor.Tensor, zx int32) int32 {
	if x.DType() == tensor.I8 {
		copy(dst, x.I8())
		return zx
	}
	src := x.U8()
	for i, v := range src {
		dst[i] = int8(v ^ 0x80)
	}
	return zx - 128
}

// ---- QLinearConv ----

type qlinearConvOp struct {
	conv *convOp // geometry (strides/pads/dilations/group)

	packMu  sync.Mutex
	packed  []*gemm.QPackedA
	sumW    []int32 // ΣW per output channel
	dwW16   []int16 // depthwise: s16 taps per channel, padded to 8
	packSrc *int8
}

func buildQLinearConv(n NodeInfo) (Op, error) {
	base, err := buildConv(n)
	if err != nil {
		return nil, err
	}
	return &qlinearConvOp{conv: base.(*convOp)}, nil
}

func (o *qlinearConvOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 8 {
		return nil, o.conv.n.Errorf("need 8-9 inputs")
	}
	x, xScale, xZp, w, wScale, wZp, yScale, yZp := in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]
	var bias []int32
	if len(in) > 8 && in[8] != nil {
		bias = in[8].I32()
	}
	wf, err := s8Weights(o.conv.n, w, wZp)
	if err != nil {
		return nil, err
	}
	_, xzpi, _, err1 := quantParams(xScale, xZp)
	_, yzpi, ydt, err2 := quantParams(yScale, yZp)
	if err1 != nil || err2 != nil || len(xzpi) != 1 || len(yzpi) != 1 || ydt == tensor.I32 {
		return nil, o.conv.n.Errorf("per-tensor activation/output quantization only")
	}
	xs, ws := x.Shape(), w.Shape()
	if len(xs) != 4 || len(ws) != 4 {
		return nil, o.conv.n.Errorf("only 2-D conv")
	}
	N, C, H, W := xs[0], xs[1], xs[2], xs[3]
	M, Cg, KH, KW := ws[0], ws[1], ws[2], ws[3]
	G := o.conv.group
	if C != Cg*G || M%G != 0 {
		return nil, o.conv.n.Errorf("channel mismatch")
	}
	pads, err := o.conv.resolvePads(H, W, KH, KW)
	if err != nil {
		return nil, o.conv.n.Errorf("%v", err)
	}
	OH := convOut(H, KH, o.conv.strides[0], o.conv.dilations[0], pads[0], pads[2])
	OW := convOut(W, KW, o.conv.strides[1], o.conv.dilations[1], pads[1], pads[3])
	Mg := M / G
	K := Cg * KH * KW
	P := OH * OW
	// requant multipliers: x_scale·w_scale[m]/y_scale
	xs0 := xScale.F32()[0]
	ys0 := yScale.F32()[0]
	wsc := wScale.F32()
	mult := make([]float32, M)
	for m := 0; m < M; m++ {
		s := wsc[0]
		if len(wsc) > 1 {
			s = wsc[m]
		}
		mult[m] = xs0 * s / ys0
	}
	zy := float32(yzpi[0])
	// shift activations to s8 once (u8 → x−128; one pass)
	xsh := ctx.NewUninit(tensor.I8, xs...)
	zx := shiftToS8(xsh.I8(), x, xzpi[0])
	xf := xsh.I8()
	out := ctx.NewUninit(ydt, N, M, OH, OW)
	q := &qconvRun{
		o: o, ctx: ctx, xf: xf, wf: wf, bias: bias, out: out, ydt: ydt,
		mult: mult, zx: zx, zy: zy,
		N: N, C: C, H: H, W: W, M: M, G: G, Cg: Cg, Mg: Mg, K: K,
		KH: KH, KW: KW, OH: OH, OW: OW, P: P, pads: pads,
	}
	depthwise := G == C && Cg == 1 && Mg == 1
	pointwise := KH == 1 && KW == 1 && o.conv.strides == [2]int{1, 1} && pads == [4]int{} && o.conv.dilations == [2]int{1, 1}
	switch {
	case depthwise:
		q.depthwise()
	case pointwise:
		q.pointwise()
	default:
		q.im2colTiled()
	}
	if ctx.Pool != nil {
		ctx.Pool.Put(xsh)
	}
	return []*tensor.Tensor{out}, nil
}

// qconvRun carries one QLinearConv invocation's state across its paths.
type qconvRun struct {
	o    *qlinearConvOp
	ctx  *Ctx
	xf   []int8
	wf   []int8
	bias []int32
	out  *tensor.Tensor
	ydt  tensor.DType
	mult []float32
	zx   int32
	zy   float32

	N, C, H, W, M, G, Cg, Mg, K int
	KH, KW, OH, OW, P           int
	pads                        [4]int
}

func (q *qconvRun) corr(m int) int32 {
	c := -q.zx * q.o.sumWFor(q.wf, q.G, q.Mg, q.K, q.M)[m]
	if q.bias != nil {
		c += q.bias[m]
	}
	return c
}

// requant writes sat(round((acc+corr)·mult) + zy) for one row segment via the
// vek kernels: (v+corr)·mult + zy = v·mult + (corr·mult + zy) — exact only up
// to f32 association, so corr is folded into the offset the same way the
// kernels compute it.
func (q *qconvRun) requant(dst8 []uint8, dsti8 []int8, acc []int32, corr int32, mult float32) {
	if corr != 0 {
		for i := range acc {
			acc[i] += corr
		}
	}
	if q.ydt == tensor.U8 {
		vek.RequantU8(dst8, acc, mult, q.zy)
		return
	}
	vek.RequantI8(dsti8, acc, mult, q.zy)
}

func (q *qconvRun) dstRow(m, n, off, ln int) ([]uint8, []int8) {
	base := (n*q.M+m)*q.P + off
	if q.ydt == tensor.U8 {
		return q.out.U8()[base : base+ln], nil
	}
	return nil, q.out.I8()[base : base+ln]
}

// depthwise: one channel per task — pad the plane with the shifted zero
// point, widen it to s16 once, accumulate the taps into corr-prefilled s32
// rows (vek.QDwRowS1 for the 3×3/5×5 stride-1/2 dilation-1 shapes, with the
// stride-2 even/odd column de-interleave from the f32 path; scalar taps
// otherwise), and requantize each row in-cache.
func (q *qconvRun) depthwise() {
	sh, sw := q.o.conv.strides[0], q.o.conv.strides[1]
	dh, dw := q.o.conv.dilations[0], q.o.conv.dilations[1]
	pt, pl, pb, pr := q.pads[0], q.pads[1], q.pads[2], q.pads[3]
	Hp, Wp := q.H+pt+pb, q.W+pl+pr
	Wh := (Wp + 1) / 2
	fast1 := sh == 1 && sw == 1 && dh == 1 && dw == 1 && q.KH == q.KW && (q.KH == 3 || q.KH == 5)
	fast2 := sh == 2 && sw == 2 && dh == 1 && dw == 1 && q.KH == q.KW && (q.KH == 3 || q.KH == 5)
	taps := q.KH * q.KW
	pad8 := (taps + 7) / 8 * 8
	var w16 []int16
	if fast1 || fast2 {
		w16 = q.o.dw16For(q.wf, q.C, taps)
	}
	// per-worker scratch: s16 padded plane + optional even/odd halves. One
	// allocation per Run (no i16 pool dtype); amortised across all channels.
	per := Hp*Wp + 2*Hp*Wh
	workers := par.Workers()
	s16All := make([]int16, workers*per)
	accT := q.ctx.NewUninit(tensor.I32, workers, q.OW)
	accF := accT.I32()
	// stride-2 sub-kernel weights (even/odd column taps), built per call: tiny.
	var wEO []int16
	KE, KO := (q.KW+1)/2, q.KW/2
	padE, padO := (q.KH*KE+7)/8*8, (q.KH*KO+7)/8*8
	if fast2 {
		wEO = make([]int16, q.C*(padE+padO))
		for c := 0; c < q.C; c++ {
			for kh := 0; kh < q.KH; kh++ {
				for kw := 0; kw < q.KW; kw++ {
					v := w16[c*pad8+kh*q.KW+kw]
					if kw%2 == 0 {
						wEO[c*(padE+padO)+kh*KE+kw/2] = v
					} else {
						wEO[c*(padE+padO)+padE+kh*KO+kw/2] = v
					}
				}
			}
		}
	}
	pv := int16(q.zx)
	par.For(q.N*q.C, 1, func(nc, wk int) {
		n, c := nc/q.C, nc%q.C
		xc := q.xf[nc*q.H*q.W:]
		s16 := s16All[wk*per : (wk+1)*per]
		plane := s16[:Hp*Wp]
		// pad + widen in one pass
		for i := range plane[:pt*Wp] {
			plane[i] = pv
		}
		for i := (pt + q.H) * Wp; i < len(plane); i++ {
			plane[i] = pv
		}
		for r := 0; r < q.H; r++ {
			row := plane[(pt+r)*Wp : (pt+r+1)*Wp]
			for i := 0; i < pl; i++ {
				row[i] = pv
			}
			src := xc[r*q.W : (r+1)*q.W]
			for i, v := range src {
				row[pl+i] = int16(v)
			}
			for i := pl + q.W; i < Wp; i++ {
				row[i] = pv
			}
		}
		corr := q.corr(c)
		mult := q.mult[c]
		acc := accF[wk*q.OW : (wk+1)*q.OW]
		fill := func() {
			for i := range acc {
				acc[i] = corr
			}
		}
		switch {
		case fast1:
			wp := w16[c*pad8 : c*pad8+pad8]
			for oh := 0; oh < q.OH; oh++ {
				fill()
				vek.QDwRowS1(acc, plane[oh*Wp:], wp, q.OW, Wp, q.KH, q.KW)
				d8, di := q.dstRow(c, n, oh*q.OW, q.OW)
				q.requant(d8, di, acc, 0, mult)
			}
		case fast2:
			// de-interleave columns into even/odd halves
			ev := s16[Hp*Wp : Hp*Wp+Hp*Wh]
			od := s16[Hp*Wp+Hp*Wh : Hp*Wp+2*Hp*Wh]
			for r := 0; r < Hp; r++ {
				prow := plane[r*Wp : (r+1)*Wp]
				er := ev[r*Wh : (r+1)*Wh]
				orow := od[r*Wh : (r+1)*Wh]
				i := 0
				for ; i+1 < Wp; i += 2 {
					er[i/2] = prow[i]
					orow[i/2] = prow[i+1]
				}
				if i < Wp {
					er[i/2] = prow[i]
					orow[i/2] = pv
				}
			}
			we := wEO[c*(padE+padO) : c*(padE+padO)+padE]
			wo := wEO[c*(padE+padO)+padE:]
			for oh := 0; oh < q.OH; oh++ {
				fill()
				vek.QDwRowS1(acc, ev[2*oh*Wh:], we, q.OW, Wh, q.KH, KE)
				vek.QDwRowS1(acc, od[2*oh*Wh:], wo, q.OW, Wh, q.KH, KO)
				d8, di := q.dstRow(c, n, oh*q.OW, q.OW)
				q.requant(d8, di, acc, 0, mult)
			}
		default:
			w := q.wf[c*taps:]
			for oh := 0; oh < q.OH; oh++ {
				clear(acc)
				for kh := 0; kh < q.KH; kh++ {
					src := plane[(oh*sh+kh*dh)*Wp:]
					for kw := 0; kw < q.KW; kw++ {
						wv := int32(w[kh*q.KW+kw])
						if wv == 0 {
							continue
						}
						sr := src[kw*dw:]
						for ow := 0; ow < q.OW; ow++ {
							acc[ow] += wv * int32(sr[ow*sw])
						}
					}
				}
				d8, di := q.dstRow(c, n, oh*q.OW, q.OW)
				q.requant(d8, di, acc, corr, mult)
			}
		}
	})
	if q.ctx.Pool != nil {
		q.ctx.Pool.Put(accT)
	}
}

// pointwise: 1×1 s1 p0 — feed the shifted activations directly as the GEMM's
// B, chunked over columns, requantizing each tile in-cache.
func (q *qconvRun) pointwise() {
	pk := q.o.weights(q.wf, q.G, q.Mg, q.K, q.M)
	Pc := max(64, (1<<17)/max(1, q.Mg*q.K))
	nChunks := (q.P + Pc - 1) / Pc
	workers := par.Workers()
	accT := q.ctx.NewUninit(tensor.I32, workers, q.Mg*Pc)
	accF := accT.I32()
	par.For(q.N*q.G*nChunks, 1, func(t, wk int) {
		ch := t % nChunks
		ng := t / nChunks
		n, g := ng/q.G, ng%q.G
		p0 := ch * Pc
		pc := min(Pc, q.P-p0)
		acc := accF[wk*q.Mg*Pc:]
		gemm.QgemmPackedS8(pk[g], pc, q.xf[(n*q.C+g*q.Cg)*q.P+p0:], q.P, acc, pc, false)
		for mi := 0; mi < q.Mg; mi++ {
			m := g*q.Mg + mi
			d8, di := q.dstRow(m, n, p0, pc)
			q.requant(d8, di, acc[mi*pc:(mi+1)*pc], q.corr(m), q.mult[m])
		}
	})
	if q.ctx.Pool != nil {
		q.ctx.Pool.Put(accT)
	}
}

// im2colTiled: general conv, tiled over output rows with per-worker byte
// column buffers (padded with the shifted zero point) and in-cache requant.
func (q *qconvRun) im2colTiled() {
	pk := q.o.weights(q.wf, q.G, q.Mg, q.K, q.M)
	rows := max(1, (1<<17)/max(1, q.K*q.OW))
	nChunks := (q.OH + rows - 1) / rows
	workers := par.Workers()
	colT := q.ctx.NewUninit(tensor.I8, workers, q.K*rows*q.OW)
	colF := colT.I8()
	accT := q.ctx.NewUninit(tensor.I32, workers, q.Mg*rows*q.OW)
	accF := accT.I32()
	pv := int8(q.zx)
	par.For(q.N*q.G*nChunks, 1, func(t, wk int) {
		ch := t % nChunks
		ng := t / nChunks
		n, g := ng/q.G, ng%q.G
		oh0 := ch * rows
		oh1 := min(oh0+rows, q.OH)
		pc := (oh1 - oh0) * q.OW
		cf := colF[wk*q.K*rows*q.OW:][:q.K*pc]
		q.qim2colRows(q.xf[(n*q.C+g*q.Cg)*q.H*q.W:], cf, pv, oh0, oh1)
		acc := accF[wk*q.Mg*rows*q.OW:]
		gemm.QgemmPackedS8(pk[g], pc, cf, pc, acc, pc, false)
		for mi := 0; mi < q.Mg; mi++ {
			m := g*q.Mg + mi
			d8, di := q.dstRow(m, n, oh0*q.OW, pc)
			q.requant(d8, di, acc[mi*pc:(mi+1)*pc], q.corr(m), q.mult[m])
		}
	})
	if q.ctx.Pool != nil {
		q.ctx.Pool.Put(accT)
		q.ctx.Pool.Put(colT)
	}
}

// qim2colRows fills the byte column matrix for output rows [oh0,oh1).
func (q *qconvRun) qim2colRows(x, col []int8, pv int8, oh0, oh1 int) {
	sh, sw := q.o.conv.strides[0], q.o.conv.strides[1]
	dh, dw := q.o.conv.dilations[0], q.o.conv.dilations[1]
	pt, pl := q.pads[0], q.pads[1]
	pc := (oh1 - oh0) * q.OW
	for k := 0; k < q.K; k++ {
		c := k / (q.KH * q.KW)
		kh := (k / q.KW) % q.KH
		kw := k % q.KW
		xc := x[c*q.H*q.W : (c+1)*q.H*q.W]
		row := col[k*pc : (k+1)*pc]
		for oh := oh0; oh < oh1; oh++ {
			ih := oh*sh + kh*dh - pt
			dst := row[(oh-oh0)*q.OW : (oh-oh0+1)*q.OW]
			if ih < 0 || ih >= q.H {
				for i := range dst {
					dst[i] = pv
				}
				continue
			}
			src := xc[ih*q.W : (ih+1)*q.W]
			for ow := 0; ow < q.OW; ow++ {
				iw := ow*sw + kw*dw - pl
				if iw < 0 || iw >= q.W {
					dst[ow] = pv
				} else {
					dst[ow] = src[iw]
				}
			}
		}
	}
}

func (o *qlinearConvOp) weights(wf []int8, G, Mg, K, M int) []*gemm.QPackedA {
	o.packMu.Lock()
	defer o.packMu.Unlock()
	o.fill(wf, G, Mg, K, M, true)
	return o.packed
}

// dw16For returns the per-channel s16 taps (padded to 8) for the depthwise
// row kernels, cached alongside ΣW/packed panels.
func (o *qlinearConvOp) dw16For(wf []int8, C, taps int) []int16 {
	o.packMu.Lock()
	defer o.packMu.Unlock()
	if o.packSrc != &wf[0] {
		o.packed, o.sumW, o.dwW16, o.packSrc = nil, nil, nil, &wf[0]
	}
	if o.dwW16 == nil {
		pad := (taps + 7) / 8 * 8
		w16 := make([]int16, C*pad)
		for c := 0; c < C; c++ {
			for t := 0; t < taps; t++ {
				w16[c*pad+t] = int16(wf[c*taps+t])
			}
		}
		o.dwW16 = w16
	}
	return o.dwW16
}

func (o *qlinearConvOp) sumWFor(wf []int8, G, Mg, K, M int) []int32 {
	o.packMu.Lock()
	defer o.packMu.Unlock()
	o.fill(wf, G, Mg, K, M, false)
	return o.sumW
}

// fill populates the cached ΣW (always) and packed panels (when needed).
func (o *qlinearConvOp) fill(wf []int8, G, Mg, K, M int, needPack bool) {
	if o.packSrc == &wf[0] && o.sumW != nil && (!needPack || o.packed != nil) {
		return
	}
	if o.packSrc != &wf[0] {
		o.packed, o.sumW, o.dwW16 = nil, nil, nil
	}
	if o.sumW == nil {
		sumW := make([]int32, M)
		for m := 0; m < M; m++ {
			var s int32
			for p := 0; p < K; p++ {
				s += int32(wf[m*K+p])
			}
			sumW[m] = s
		}
		o.sumW = sumW
	}
	if needPack && o.packed == nil {
		pk := make([]*gemm.QPackedA, G)
		for g := 0; g < G; g++ {
			pk[g] = gemm.QPackA(Mg, K, wf[g*Mg*K:], K)
		}
		o.packed = pk
	}
	o.packSrc = &wf[0]
}

// ---- QLinearMatMul / MatMulInteger ----

type qmatmulOp struct {
	n       NodeInfo
	requant bool // QLinearMatMul; false = MatMulInteger (s32 out)

	packMu  sync.Mutex
	packed  *gemm.QPackedA
	colSum  []int32
	packSrc *int8
}

// bWeights packs the s8 B matrix [k×n] (column sums included); cached when B
// is the same storage across calls (constant weights).
func (o *qmatmulOp) bWeights(bf []int8, k, n int) (*gemm.QPackedA, []int32) {
	o.packMu.Lock()
	defer o.packMu.Unlock()
	if o.packSrc == &bf[0] && o.packed != nil && o.packed.Cols() == k && o.packed.Rows() == n {
		return o.packed, o.colSum
	}
	// The SMMLA kernel computes rowsA×rowsPackedA... we need y = a·b, computed
	// as (b>ᵀ·aᵀ)ᵀ? Simpler: pack bᵀ as the kernel's A operand (rows = b's
	// columns), feed a as its B operand (k×m), and the kernel's output is yᵀ.
	bt := make([]int8, n*k)
	for p := 0; p < k; p++ {
		row := bf[p*n : (p+1)*n]
		for j, v := range row {
			bt[j*k+p] = v
		}
	}
	cs := make([]int32, n)
	for j := 0; j < n; j++ {
		var sum int32
		for p := 0; p < k; p++ {
			sum += int32(bt[j*k+p])
		}
		cs[j] = sum
	}
	o.packed, o.colSum, o.packSrc = gemm.QPackA(n, k, bt, k), cs, &bf[0]
	return o.packed, o.colSum
}

func (o *qmatmulOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	var a, b, aZp, bZp *tensor.Tensor
	var aScale, bScale, yScale, yZp *tensor.Tensor
	if o.requant {
		if len(in) < 8 {
			return nil, o.n.Errorf("need 8 inputs")
		}
		a, aScale, aZp, b, bScale, bZp, yScale, yZp = in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]
	} else {
		if len(in) < 2 {
			return nil, o.n.Errorf("need 2-4 inputs")
		}
		a, b = in[0], in[1]
		if len(in) > 2 {
			aZp = in[2]
		}
		if len(in) > 3 {
			bZp = in[3]
		}
	}
	bf, err := s8Weights(o.n, b, bZp)
	if err != nil {
		return nil, err
	}
	as, bs := a.Shape(), b.Shape()
	if len(as) != 2 || len(bs) != 2 || as[1] != bs[0] {
		return nil, o.n.Errorf("2-D matmul only (a %v, b %v)", as, bs)
	}
	m, k, n := as[0], as[1], bs[1]
	var za int32
	if aZp != nil {
		_, zpi, _, zerr := quantParams(tensor.FromF32([]float32{1}), aZp)
		if zerr != nil || len(zpi) != 1 {
			return nil, o.n.Errorf("per-tensor a zero_point only")
		}
		za = zpi[0]
	}
	ash := ctx.NewUninit(tensor.I8, as...)
	za = shiftToS8(ash.I8(), a, za)
	pb, colSum := o.bWeights(bf, k, n)
	// yᵀ[n×m] = packed(bᵀ)·aᵀ — feed aᵀ as the kernel's B via ldb tricks:
	// a is row-major [m×k]; aᵀ is [k×m] with element (p,i) = a[i*k+p], i.e.
	// column-major — QgemmPackedS8 wants row-major B[k×m], stride... aᵀ
	// row p is a's column p (stride k). Instead of materialising aᵀ, note the
	// kernel packs B per panel anyway; here we materialise aᵀ once (m·k
	// bytes) — cheap next to the GEMM.
	at := ctx.NewUninit(tensor.I8, k, m)
	atf := at.I8()
	af := ash.I8()
	for i := 0; i < m; i++ {
		row := af[i*k : (i+1)*k]
		for p, v := range row {
			atf[p*m+i] = v
		}
	}
	yt := ctx.NewUninit(tensor.I32, n, m)
	ytf := yt.I32()
	gemm.QgemmPackedS8(pb, m, atf, m, ytf, m, true)
	out := func() *tensor.Tensor {
		if !o.requant {
			o2 := ctx.NewUninit(tensor.I32, m, n)
			of := o2.I32()
			for j := 0; j < n; j++ {
				corr := -za * colSum[j]
				row := ytf[j*m : (j+1)*m]
				for i, v := range row {
					of[i*n+j] = v + corr
				}
			}
			return o2
		}
		_, yzpi, ydt, e2 := quantParams(yScale, yZp)
		if e2 != nil || len(yzpi) != 1 {
			return nil
		}
		multV := aScale.F32()[0] * bScale.F32()[0] / yScale.F32()[0]
		zy := float32(yzpi[0])
		o2 := ctx.NewUninit(ydt, m, n)
		if ydt == tensor.U8 {
			of := o2.U8()
			for j := 0; j < n; j++ {
				corr := -za * colSum[j]
				row := ytf[j*m : (j+1)*m]
				for i, v := range row {
					of[i*n+j] = satU8(float32(v+corr)*multV + zy)
				}
			}
		} else {
			of := o2.I8()
			for j := 0; j < n; j++ {
				corr := -za * colSum[j]
				row := ytf[j*m : (j+1)*m]
				for i, v := range row {
					of[i*n+j] = satI8(float32(v+corr)*multV + zy)
				}
			}
		}
		return o2
	}()
	if out == nil {
		return nil, o.n.Errorf("unsupported output quantization")
	}
	if ctx.Pool != nil {
		ctx.Pool.Put(yt)
		ctx.Pool.Put(at)
		ctx.Pool.Put(ash)
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	Register("", "QLinearMatMul", 10, func(n NodeInfo) (Op, error) { return &qmatmulOp{n: n, requant: true}, nil })
	Register("", "MatMulInteger", 10, func(n NodeInfo) (Op, error) { return &qmatmulOp{n: n, requant: false}, nil })
	Register("", "QuantizeLinear", 10, func(n NodeInfo) (Op, error) {
		return &quantizeLinearOp{n: n, axis: int(n.Attrs.Int("axis", 1))}, nil
	})
	Register("", "DequantizeLinear", 10, func(n NodeInfo) (Op, error) {
		return &dequantizeLinearOp{n: n, axis: int(n.Attrs.Int("axis", 1))}, nil
	})
	Register("", "DynamicQuantizeLinear", 11, func(n NodeInfo) (Op, error) {
		return &dynamicQuantizeLinearOp{n: n}, nil
	})
	Register("", "QLinearConv", 10, buildQLinearConv)
}
