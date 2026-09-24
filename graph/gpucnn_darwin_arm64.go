//go:build darwin && arm64

package graph

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/ops"
	"github.com/giraffesyo/ingot/tensor"
)

// ---- convolution, pooling, activations (NCHW) ----

// gpuEpilogue is a fused conv activation plus post-affine (the optimizer's
// ingot_act / ingot_post_* attributes).
type gpuEpilogue struct {
	act                       int
	alpha, beta, scale, shift float32
	active                    bool
}

var gpuActs = map[string]int{"": metal.ActNone, "relu": metal.ActRelu, "hardswish": metal.ActHardSwish,
	"hardsigmoid": metal.ActHardSigmoid, "sigmoid": metal.ActSigmoid, "silu": metal.ActSiLU,
	"clip": metal.ActClip, "leakyrelu": metal.ActLeakyRelu}

func epilogueOf(a ops.Attrs) (gpuEpilogue, bool) {
	name := a.String("ingot_act", "")
	act, ok := gpuActs[name]
	if !ok {
		return gpuEpilogue{}, false
	}
	e := gpuEpilogue{act: act, alpha: a.Float("ingot_act_alpha", 0), beta: a.Float("ingot_act_beta", 0),
		scale: a.Float("ingot_post_scale", 1), shift: a.Float("ingot_post_shift", 0)}
	e.active = name != "" || e.scale != 1 || e.shift != 0
	return e, true
}

func (p gpuEpilogue) encode(e *metal.Encoder, r metal.Region, n int) {
	if p.active {
		e.Act(p.act, r, r, n, p.alpha, p.beta, p.scale, p.shift)
	}
}

// samePads resolves ONNX auto_pad for one spatial dim (eff = the dilated
// kernel extent): (before, after) or ok=false for an unknown mode.
func samePads(mode string, in, eff, stride int, pads [2]int) ([2]int, bool) {
	switch mode {
	case "NOTSET", "":
		return pads, true
	case "VALID":
		return [2]int{}, true
	case "SAME_UPPER", "SAME_LOWER":
		out := (in + stride - 1) / stride
		total := max(0, (out-1)*stride+eff-in)
		a := total / 2
		if mode == "SAME_LOWER" {
			a = (total + 1) / 2
		}
		return [2]int{a, total - a}, true
	}
	return pads, false
}

type convGPU struct {
	strides, dils [2]int
	pads          [4]int
	autoPad       string
	group         int
	epi           gpuEpilogue
}

func newConvGPU(a ops.Attrs) gpuOp {
	st, di, pa := a.Ints("strides", []int64{1, 1}), a.Ints("dilations", []int64{1, 1}), a.Ints("pads", []int64{0, 0, 0, 0})
	epi, ok := epilogueOf(a)
	if len(st) != 2 || len(di) != 2 || len(pa) != 4 || !ok {
		return nil
	}
	return convGPU{strides: [2]int{int(st[0]), int(st[1])}, dils: [2]int{int(di[0]), int(di[1])},
		pads: [4]int{int(pa[0]), int(pa[1]), int(pa[2]), int(pa[3])}, autoPad: a.String("auto_pad", "NOTSET"),
		group: int(a.Int("group", 1)), epi: epi}
}

// im2colBudget caps the im2col scratch (floats) per GEMM chunk.
const im2colBudget = 8 << 20

func (o convGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) < 2 || in[0] == nil || in[1] == nil || !allF32(in...) || !onlyFirstOutput(st) {
		return nil, nil, false
	}
	x, w := in[0], in[1]
	var bias *tensor.Tensor
	if len(in) > 2 {
		bias = in[2]
	}
	xs, ws := x.Shape(), w.Shape()
	if len(xs) != 4 || len(ws) != 4 {
		return nil, nil, false
	}
	g := metal.ConvGeom{N: xs[0], C: xs[1], H: xs[2], W: xs[3], M: ws[0], KH: ws[2], KW: ws[3],
		SH: o.strides[0], SW: o.strides[1], DH: o.dils[0], DW: o.dils[1], Group: o.group}
	if g.Group <= 0 || g.C != ws[1]*g.Group || g.M%g.Group != 0 || (bias != nil && bias.Numel() != g.M) {
		return nil, nil, false
	}
	ph, ok1 := samePads(o.autoPad, g.H, g.DH*(g.KH-1)+1, g.SH, [2]int{o.pads[0], o.pads[2]})
	pw, ok2 := samePads(o.autoPad, g.W, g.DW*(g.KW-1)+1, g.SW, [2]int{o.pads[1], o.pads[3]})
	if !ok1 || !ok2 {
		return nil, nil, false
	}
	g.PT, g.PL = ph[0], pw[0]
	g.OH = (g.H+ph[0]+ph[1]-(g.DH*(g.KH-1)+1))/g.SH + 1
	g.OW = (g.W+pw[0]+pw[1]-(g.DW*(g.KW-1)+1))/g.SW + 1
	if g.OH <= 0 || g.OW <= 0 || g.N == 0 {
		return nil, nil, false
	}
	ts := []*tensor.Tensor{x, w}
	if bias != nil {
		ts = append(ts, bias)
	}
	rs, ok := c.regions(ts...)
	if !ok {
		return nil, nil, false
	}
	var rb metal.Region
	if bias != nil {
		rb = rs[2]
	}
	out := c.out(g.N, g.M, g.OH, g.OW)
	ro, ok := c.regions(out)
	if !ok {
		c.release(out)
		return nil, nil, false
	}
	n, P := out.Numel(), g.OH*g.OW
	epi := o.epi
	if g.Group != 1 {
		return []*tensor.Tensor{out}, func(e *metal.Encoder) {
			e.ConvDirect(rs[0], rs[1], rb, ro[0], g)
			epi.encode(e, ro[0], n)
		}, true
	}
	K := g.C * g.KH * g.KW
	direct := g.KH == 1 && g.KW == 1 && g.SH == 1 && g.SW == 1 && g.PT == 0 && g.PL == 0 && P == g.H*g.W
	pc := P
	var rc metal.Region
	if !direct {
		pc = min(P, max(256, im2colBudget/K))
		cols := c.out(K * pc)
		rcs, ok := c.regions(cols)
		// Scratch: back to the pool now; later users run after this
		// node in the same serial command buffer.
		c.release(cols)
		if !ok {
			c.release(out)
			return nil, nil, false
		}
		rc = rcs[0]
	}
	at := func(r metal.Region, floats int) metal.Region { return metal.Region{B: r.B, Off: r.Off + 4*floats} }
	return []*tensor.Tensor{out}, func(e *metal.Encoder) {
		for b := range g.N {
			xb, yb := at(rs[0], b*g.C*g.H*g.W), b*g.M*P
			if direct {
				e.Gemm(metal.Gemm{M: g.M, N: P, K: K, A: rs[1], B: xb, C: at(ro[0], yb)})
				continue
			}
			for p0 := 0; p0 < P; p0 += pc {
				cn := min(pc, P-p0)
				e.Im2ColNCHW(xb, rc, g, p0, cn)
				e.Gemm(metal.Gemm{M: g.M, N: cn, K: K, A: rs[1], B: rc, LDB: cn, C: at(ro[0], yb+p0), LDC: P})
			}
		}
		if bias != nil {
			e.BinaryBcast(metal.OpAdd, ro[0], rb, ro[0], n, 1, n, P, g.M)
		}
		epi.encode(e, ro[0], n)
	}, true
}

type poolGPU struct {
	isMax, ceil, includePad bool
	kernel, strides         [2]int
	pads                    [4]int
	autoPad                 string
}

func newPoolGPU(a ops.Attrs, isMax bool) gpuOp {
	ks, st, pa := a.Ints("kernel_shape", nil), a.Ints("strides", []int64{1, 1}), a.Ints("pads", []int64{0, 0, 0, 0})
	if len(ks) != 2 || len(st) != 2 || len(pa) != 4 {
		return nil
	}
	if di := a.Ints("dilations", nil); di != nil && (len(di) != 2 || di[0] != 1 || di[1] != 1) {
		return nil
	}
	return poolGPU{isMax: isMax, ceil: a.Int("ceil_mode", 0) == 1, includePad: a.Int("count_include_pad", 0) == 1,
		kernel: [2]int{int(ks[0]), int(ks[1])}, strides: [2]int{int(st[0]), int(st[1])},
		pads: [4]int{int(pa[0]), int(pa[1]), int(pa[2]), int(pa[3])}, autoPad: a.String("auto_pad", "NOTSET")}
}

func (o poolGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) < 1 || in[0] == nil || !allF32(in[0]) || !onlyFirstOutput(st) {
		return nil, nil, false
	}
	x := in[0]
	xs := x.Shape()
	if len(xs) != 4 {
		return nil, nil, false
	}
	H, W := xs[2], xs[3]
	KH, KW, SH, SW := o.kernel[0], o.kernel[1], o.strides[0], o.strides[1]
	ph, ok1 := samePads(o.autoPad, H, KH, SH, [2]int{o.pads[0], o.pads[2]})
	pw, ok2 := samePads(o.autoPad, W, KW, SW, [2]int{o.pads[1], o.pads[3]})
	if !ok1 || !ok2 {
		return nil, nil, false
	}
	outDim := func(in, k, s, pa, pb int) int {
		if o.ceil {
			v := (in+pa+pb-k+s-1)/s + 1
			if (v-1)*s >= in+pa {
				v--
			}
			return v
		}
		return (in+pa+pb-k)/s + 1
	}
	OH, OW := outDim(H, KH, SH, ph[0], ph[1]), outDim(W, KW, SW, pw[0], pw[1])
	if OH <= 0 || OW <= 0 {
		return nil, nil, false
	}
	rx, ok := c.regions(x)
	if !ok {
		return nil, nil, false
	}
	out := c.out(xs[0], xs[1], OH, OW)
	ro, ok := c.regions(out)
	if !ok {
		c.release(out)
		return nil, nil, false
	}
	pads := [4]int{ph[0], pw[0], ph[1], pw[1]}
	planes := xs[0] * xs[1]
	return []*tensor.Tensor{out}, func(e *metal.Encoder) {
		e.Pool2D(rx[0], ro[0], planes, H, W, OH, OW, KH, KW, SH, SW, pads, o.isMax, o.includePad)
	}, true
}

// globalAvgPoolGPU is GlobalAveragePool: a mean over each plane.
type globalAvgPoolGPU struct{}

func (globalAvgPoolGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) < 1 || in[0] == nil || !allF32(in[0]) {
		return nil, nil, false
	}
	x := in[0]
	xs := x.Shape()
	if len(xs) < 3 || x.Numel() == 0 {
		return nil, nil, false
	}
	rows := xs[0] * xs[1]
	cols := x.Numel() / rows
	rx, ok := c.regions(x)
	if !ok {
		return nil, nil, false
	}
	oshape := []int{xs[0], xs[1]}
	for range xs[2:] {
		oshape = append(oshape, 1)
	}
	out := c.out(oshape...)
	ro, ok := c.regions(out)
	if !ok {
		c.release(out)
		return nil, nil, false
	}
	return []*tensor.Tensor{out}, func(e *metal.Encoder) { e.ReduceRows(rx[0], ro[0], rows, cols, cols, true) }, true
}

// actGPU is a parameterised activation (HardSwish, HardSigmoid, Clip,
// LeakyRelu). Clip's bounds come from constant inputs (opset ≥ 11) or
// attributes.
type actGPU struct {
	act         int
	alpha, beta float32
	clip        bool
}

func (o actGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) < 1 || in[0] == nil || !allF32(in[0]) || !onlyFirstOutput(st) {
		return nil, nil, false
	}
	alpha, beta := o.alpha, o.beta
	if o.clip {
		for k, dst := range []*float32{&alpha, &beta} {
			if len(in) > k+1 && in[k+1] != nil {
				// Read on the CPU now: only constants are known written.
				if c.s.constVals[st.in[k+1]] == nil {
					return nil, nil, false
				}
				b := in[k+1]
				if b.DType() != tensor.F32 || b.Numel() != 1 {
					return nil, nil, false
				}
				*dst = b.F32()[0]
			}
		}
	}
	x := in[0]
	rx, ok := c.regions(x)
	if !ok {
		return nil, nil, false
	}
	out := c.out(x.Shape()...)
	ro, ok := c.regions(out)
	if !ok {
		c.release(out)
		return nil, nil, false
	}
	n := x.Numel()
	return []*tensor.Tensor{out}, func(e *metal.Encoder) { e.Act(o.act, rx[0], ro[0], n, alpha, beta, 1, 0) }, true
}

func newClipGPU(a ops.Attrs) gpuOp {
	return actGPU{act: metal.ActClip, clip: true,
		alpha: a.Float("min", float32(math.Inf(-1))), beta: a.Float("max", float32(math.Inf(1)))}
}

// resizeGPU is Resize (nearest / linear over NCHW planes): the taps come
// from ops.PlanResize — the CPU op's own mapping — cached per shape.
type resizeGPU struct {
	attrs ops.Attrs
	name  string
}

func (o resizeGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) < 1 || in[0] == nil || !allF32(in[0]) || !onlyFirstOutput(st) {
		return nil, nil, false
	}
	// Scales / sizes are read on the CPU now: only constants (or integer
	// sizes, never GPU outputs) are known written.
	if len(in) > 2 && in[2] != nil && in[2].Numel() > 0 && c.s.constVals[st.in[2]] == nil {
		return nil, nil, false
	}
	tp, err := ops.PlanResize(o.attrs, in)
	if err != nil {
		return nil, nil, false
	}
	x := in[0]
	xs := x.Shape()
	H, W := xs[2], xs[3]
	key := fmt.Sprintf("resize/%s/%dx%d->%dx%d", o.name, H, W, tp.OH, tp.OW)
	ri, ok1 := c.s.table(key+"/i", func() []byte {
		b := make([]byte, 0, 4*(2*tp.OH+2*tp.OW))
		for _, v := range [][]int{tp.Y0, tp.Y1, tp.X0, tp.X1} {
			for _, i := range v {
				b = binary.LittleEndian.AppendUint32(b, uint32(i))
			}
		}
		return b
	})
	rw, ok2 := c.s.table(key+"/w", func() []byte {
		b := make([]byte, 0, 4*(tp.OH+tp.OW))
		for _, v := range [][]float32{tp.WY, tp.WX} {
			for _, f := range v {
				b = binary.LittleEndian.AppendUint32(b, math.Float32bits(f))
			}
		}
		return b
	})
	if !ok1 || !ok2 {
		return nil, nil, false
	}
	rx, ok := c.regions(x)
	if !ok {
		return nil, nil, false
	}
	out := c.out(xs[0], xs[1], tp.OH, tp.OW)
	ro, ok := c.regions(out)
	if !ok {
		c.release(out)
		return nil, nil, false
	}
	planes, OH, OW := xs[0]*xs[1], tp.OH, tp.OW
	return []*tensor.Tensor{out}, func(e *metal.Encoder) { e.ResizeTaps(rx[0], ro[0], ri, rw, planes, H, W, OH, OW) }, true
}

type convTransposeGPU struct {
	strides, dils, outPad [2]int
	pads                  [4]int
	outShape              []int64
	group                 int
	epi                   gpuEpilogue
}

func newConvTransposeGPU(a ops.Attrs) gpuOp {
	st, di, pa := a.Ints("strides", []int64{1, 1}), a.Ints("dilations", []int64{1, 1}), a.Ints("pads", []int64{0, 0, 0, 0})
	op := a.Ints("output_padding", []int64{0, 0})
	epi, ok := epilogueOf(a)
	if len(st) != 2 || len(di) != 2 || len(pa) != 4 || len(op) != 2 || !ok {
		return nil
	}
	if ap := a.String("auto_pad", "NOTSET"); ap != "NOTSET" && ap != "" {
		return nil
	}
	return convTransposeGPU{strides: [2]int{int(st[0]), int(st[1])}, dils: [2]int{int(di[0]), int(di[1])},
		outPad: [2]int{int(op[0]), int(op[1])}, pads: [4]int{int(pa[0]), int(pa[1]), int(pa[2]), int(pa[3])},
		outShape: a.Ints("output_shape", nil), group: int(a.Int("group", 1)), epi: epi}
}

func (o convTransposeGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) < 2 || in[0] == nil || in[1] == nil || !allF32(in...) || !onlyFirstOutput(st) {
		return nil, nil, false
	}
	x, w := in[0], in[1]
	var bias *tensor.Tensor
	if len(in) > 2 {
		bias = in[2]
	}
	xs, ws := x.Shape(), w.Shape()
	if len(xs) != 4 || len(ws) != 4 || ws[0] != xs[1] || o.group <= 0 || xs[1]%o.group != 0 {
		return nil, nil, false
	}
	g := metal.ConvGeom{N: xs[0], C: xs[1], H: xs[2], W: xs[3], M: ws[1] * o.group, KH: ws[2], KW: ws[3],
		SH: o.strides[0], SW: o.strides[1], DH: o.dils[0], DW: o.dils[1], PT: o.pads[0], PL: o.pads[1], Group: o.group}
	if bias != nil && bias.Numel() != g.M {
		return nil, nil, false
	}
	g.OH = (g.H-1)*g.SH - o.pads[0] - o.pads[2] + g.DH*(g.KH-1) + o.outPad[0] + 1
	g.OW = (g.W-1)*g.SW - o.pads[1] - o.pads[3] + g.DW*(g.KW-1) + o.outPad[1] + 1
	if len(o.outShape) == 4 {
		g.OH, g.OW = int(o.outShape[2]), int(o.outShape[3])
	}
	if g.OH <= 0 || g.OW <= 0 || g.N == 0 {
		return nil, nil, false
	}
	ts := []*tensor.Tensor{x, w}
	if bias != nil {
		ts = append(ts, bias)
	}
	rs, ok := c.regions(ts...)
	if !ok {
		return nil, nil, false
	}
	var rb metal.Region
	if bias != nil {
		rb = rs[2]
	}
	out := c.out(g.N, g.M, g.OH, g.OW)
	ro, ok := c.regions(out)
	if !ok {
		c.release(out)
		return nil, nil, false
	}
	n, epi := out.Numel(), o.epi
	return []*tensor.Tensor{out}, func(e *metal.Encoder) {
		e.ConvTransposeDirect(rs[0], rs[1], rb, ro[0], g)
		epi.encode(e, ro[0], n)
	}, true
}
