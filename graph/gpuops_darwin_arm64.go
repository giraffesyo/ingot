//go:build darwin && arm64

package graph

import (
	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/tensor"
)

// gpuOp is a node's Metal implementation. prepare validates the inputs,
// allocates the outputs from the session pool (uninitialised: the kernels
// overwrite them) and returns the encoding; ok false (before allocating
// anything) sends the node to its CPU op.
type gpuOp interface {
	prepare(c *gpuCtx, st *step, in []*tensor.Tensor) (outs []*tensor.Tensor, enc func(e *metal.Encoder), ok bool)
}

type gpuCtx struct {
	s    *GPUSession
	ones *tensor.Tensor // [n] ones, grown on demand (LayerNorm without scale)
}

func (c *gpuCtx) out(shape ...int) *tensor.Tensor { return c.s.pool.GetUninit(tensor.F32, shape...) }

// regions resolves the tensors' Metal regions; false if any is not in
// session memory.
func (c *gpuCtx) regions(ts ...*tensor.Tensor) ([]metal.Region, bool) {
	rs := make([]metal.Region, len(ts))
	for i, t := range ts {
		r, ok := c.s.region(t)
		if !ok {
			return nil, false
		}
		rs[i] = r
	}
	return rs, true
}

// allF32 reports whether every non-nil tensor is contiguous f32.
func allF32(ts ...*tensor.Tensor) bool {
	for _, t := range ts {
		if t != nil && (t.DType() != tensor.F32 || !t.IsContiguous()) {
			return false
		}
	}
	return true
}

// release returns scratch or declined outputs to the pool.
func (c *gpuCtx) release(ts ...*tensor.Tensor) {
	for _, t := range ts {
		c.s.pool.Put(t)
	}
}

// onlyFirstOutput reports whether a node's outputs past the first are
// absent (the GPU ops produce one).
func onlyFirstOutput(st *step) bool {
	for _, id := range st.out[1:] {
		if id >= 0 {
			return false
		}
	}
	return true
}

func gpuOpFor(n *Node) gpuOp {
	a := n.Attrs
	switch n.Domain {
	case "":
		switch n.OpType {
		case "Add":
			return binaryGPU{metal.OpAdd}
		case "Sub":
			return binaryGPU{metal.OpSub}
		case "Mul":
			return binaryGPU{metal.OpMul}
		case "Div":
			return binaryGPU{metal.OpDiv}
		case "Pow":
			return binaryGPU{metal.OpPow}
		case "Max":
			return binaryGPU{metal.OpMax}
		case "Min":
			return binaryGPU{metal.OpMin}
		case "Relu":
			return unaryGPU{metal.UnRelu}
		case "Sigmoid":
			return unaryGPU{metal.UnSigmoid}
		case "Tanh":
			return unaryGPU{metal.UnTanh}
		case "Sqrt":
			return unaryGPU{metal.UnSqrt}
		case "Reciprocal":
			return unaryGPU{metal.UnReciprocal}
		case "Exp":
			return unaryGPU{metal.UnExp}
		case "Neg":
			return unaryGPU{metal.UnNeg}
		case "Abs":
			return unaryGPU{metal.UnAbs}
		case "Erf":
			return unaryGPU{metal.UnErf}
		case "Gelu":
			if a.String("approximate", "none") == "tanh" {
				return unaryGPU{metal.UnGeluTanh}
			}
			return unaryGPU{metal.UnGeluErf}
		case "MatMul":
			if act := a.String("ingot_act", ""); act != "" && act != "gelu" {
				return nil
			}
			return matmulGPU{gelu: a.String("ingot_act", "") == "gelu"}
		case "Gemm":
			if a.Int("transA", 0) != 0 || a.Float("alpha", 1) != 1 {
				return nil
			}
			return gemmGPU{transB: a.Int("transB", 0) != 0, beta: a.Float("beta", 1)}
		case "LayerNormalization":
			return layerNormGPU{axis: int(a.Int("axis", -1)), eps: a.Float("epsilon", 1e-5)}
		case "Softmax":
			return softmaxGPU{axis: int(a.Int("axis", -1))}
		case "ReduceMean":
			return reduceMeanGPU{keep: a.Int("keepdims", 1) == 1, attrAxes: a.Ints("axes", nil)}
		case "Transpose":
			return transposeGPU{perm: a.Ints("perm", nil)}
		case "Gather":
			return gatherGPU{axis: int(a.Int("axis", 0))}
		case "Concat":
			return concatGPU{axis: int(a.Int("axis", 0))}
		}
	case "ingot":
		switch n.OpType {
		case "Gelu":
			return unaryGPU{metal.UnGeluErf}
		case "SiLU":
			return unaryGPU{metal.UnSiLU}
		case "LayerNorm":
			return layerNormGPU{axis: int(a.Int("axis", -1)), eps: a.Float("epsilon", 1e-5)}
		case "AddLayerNorm":
			return addLayerNormGPU{axis: int(a.Int("axis", -1)), eps: a.Float("epsilon", 1e-5)}
		case "SDPA":
			if a.Int("cache", 0) != 0 {
				return nil
			}
			return sdpaGPU{scale: a.Float("scale", 1), aLay: int(a.Int("a_layout", 0)), bLay: int(a.Int("b_layout", 0)),
				vLay: int(a.Int("v_layout", 0)), strideOut: a.Int("stride_out", 0) != 0}
		}
	}
	return nil
}

// ---- elementwise ----

type binaryGPU struct{ op int }

func (o binaryGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) != 2 || in[0] == nil || in[1] == nil || !allF32(in[0], in[1]) {
		return nil, nil, false
	}
	a, b := in[0], in[1]
	oshape, ok := broadcastShapes(a.Shape(), b.Shape())
	if !ok {
		return nil, nil, false
	}
	da, ma, ok1 := bcastIndex(a.Shape(), oshape)
	db, mb, ok2 := bcastIndex(b.Shape(), oshape)
	if !ok1 || !ok2 {
		return nil, nil, false
	}
	rs, ok := c.regions(a, b)
	if !ok {
		return nil, nil, false
	}
	out := c.out(oshape...)
	ro, ok := c.regions(out)
	if !ok {
		c.release(out)
		return nil, nil, false
	}
	n := out.Numel()
	return []*tensor.Tensor{out}, func(e *metal.Encoder) { e.BinaryBcast(o.op, rs[0], rs[1], ro[0], n, da, ma, db, mb) }, true
}

// broadcastShapes is NumPy broadcasting of two shapes.
func broadcastShapes(a, b tensor.Shape) ([]int, bool) {
	r := max(len(a), len(b))
	out := make([]int, r)
	for i := range r {
		da, db := 1, 1
		if j := i - (r - len(a)); j >= 0 {
			da = a[j]
		}
		if j := i - (r - len(b)); j >= 0 {
			db = b[j]
		}
		switch {
		case da == db, db == 1:
			out[i] = da
		case da == 1:
			out[i] = db
		default:
			return nil, false
		}
	}
	return out, true
}

// bcastIndex expresses operand x of output shape out as index (i / inner) %
// span: x's non-1 dims must be one contiguous block of out's dims (x is that
// block repeated around it).
func bcastIndex(x tensor.Shape, out []int) (inner, span int, ok bool) {
	r := len(out)
	lo, hi := -1, -1
	for i := range r {
		d := 1
		if j := i - (r - len(x)); j >= 0 {
			d = x[j]
		}
		if d == 1 {
			continue
		}
		if d != out[i] || (hi >= 0 && hi != i) {
			// A broadcast gap inside the block: only size-1 output dims
			// may sit between non-1 operand dims.
			if d != out[i] {
				return 0, 0, false
			}
			for k := hi; k < i; k++ {
				if out[k] != 1 {
					return 0, 0, false
				}
			}
		}
		if lo < 0 {
			lo = i
		}
		hi = i + 1
	}
	if lo < 0 { // scalar
		return 1, 1, true
	}
	inner, span = 1, 1
	for _, d := range out[hi:] {
		inner *= d
	}
	for _, d := range out[lo:hi] {
		span *= d
	}
	return inner, span, true
}

type unaryGPU struct{ op int }

func (o unaryGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) < 1 || in[0] == nil || !allF32(in[0]) || !onlyFirstOutput(st) {
		return nil, nil, false
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
	return []*tensor.Tensor{out}, func(e *metal.Encoder) { e.Unary(o.op, rx[0], ro[0], n) }, true
}

// ---- matrix products ----

type matmulGPU struct{ gelu bool }

func (o matmulGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) < 2 || in[0] == nil || in[1] == nil || !allF32(in...) {
		return nil, nil, false
	}
	a, b := in[0], in[1]
	var bias *tensor.Tensor
	if len(in) > 2 {
		bias = in[2]
	}
	as, bs := a.Shape(), b.Shape()
	if len(as) < 2 || len(bs) < 2 || as[len(as)-1] != bs[len(bs)-2] {
		return nil, nil, false
	}
	K, N := as[len(as)-1], bs[len(bs)-1]
	M := as[len(as)-2]
	batch := 1
	switch {
	case len(bs) == 2: // weights: fold every leading dim of a into M
		M = a.Numel() / K
	case len(as) == len(bs): // batched: equal batch dims
		for i := range len(as) - 2 {
			if as[i] != bs[i] {
				return nil, nil, false
			}
			batch *= as[i]
		}
	default:
		return nil, nil, false
	}
	if bias != nil && bias.Numel() != N {
		return nil, nil, false
	}
	ts := []*tensor.Tensor{a, b}
	if bias != nil {
		ts = append(ts, bias)
	}
	rs, ok := c.regions(ts...)
	if !ok {
		return nil, nil, false
	}
	oshape := append(append([]int{}, as[:len(as)-1]...), N)
	out := c.out(oshape...)
	ro, ok := c.regions(out)
	if !ok {
		c.release(out)
		return nil, nil, false
	}
	gelu := o.gelu
	return []*tensor.Tensor{out}, func(e *metal.Encoder) {
		for bi := range batch {
			e.Gemm(metal.Gemm{M: M, N: N, K: K,
				A: metal.Region{B: rs[0].B, Off: rs[0].Off + 4*bi*M*K},
				B: metal.Region{B: rs[1].B, Off: rs[1].Off + 4*bi*K*N},
				C: metal.Region{B: ro[0].B, Off: ro[0].Off + 4*bi*M*N}})
		}
		rows := out.Numel() / N
		if bias != nil {
			e.AddBias(ro[0], rs[2], rows, N, N)
		}
		if gelu {
			e.Unary(metal.UnGeluErf, ro[0], ro[0], out.Numel())
		}
	}, true
}

type gemmGPU struct {
	transB bool
	beta   float32
}

func (o gemmGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) < 2 || in[0] == nil || in[1] == nil || !allF32(in...) {
		return nil, nil, false
	}
	a, b := in[0], in[1]
	if a.Shape().Rank() != 2 || b.Shape().Rank() != 2 {
		return nil, nil, false
	}
	M, K := a.Dim(0), a.Dim(1)
	Kb, N := b.Dim(0), b.Dim(1)
	if o.transB {
		Kb, N = N, Kb
	}
	if K != Kb {
		return nil, nil, false
	}
	var bias *tensor.Tensor
	if len(in) > 2 && in[2] != nil && o.beta != 0 {
		if o.beta != 1 || in[2].Numel() != N {
			return nil, nil, false
		}
		bias = in[2]
	}
	ts := []*tensor.Tensor{a, b}
	if bias != nil {
		ts = append(ts, bias)
	}
	rs, ok := c.regions(ts...)
	if !ok {
		return nil, nil, false
	}
	out := c.out(M, N)
	ro, ok := c.regions(out)
	if !ok {
		c.release(out)
		return nil, nil, false
	}
	transB := o.transB
	return []*tensor.Tensor{out}, func(e *metal.Encoder) {
		e.Gemm(metal.Gemm{M: M, N: N, K: K, A: rs[0], B: rs[1], C: ro[0], TransB: transB})
		if bias != nil {
			e.AddBias(ro[0], rs[2], M, N, N)
		}
	}, true
}

// ---- normalisation, softmax, reductions ----

// lastAxis reports whether axis names the last of rank dims.
func lastAxis(axis, rank int) bool { return axis == -1 || axis == rank-1 }

func (c *gpuCtx) onesVec(n int) (metal.Region, bool) {
	if c.ones == nil || c.ones.Numel() < n {
		if c.ones != nil {
			c.release(c.ones)
		}
		c.ones = c.out(n)
		copy(c.ones.F32(), tensor.FromF32(onesSlice(n), n).F32())
	}
	r, ok := c.regions(c.ones)
	if !ok {
		return metal.Region{}, false
	}
	return r[0], true
}

func onesSlice(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = 1
	}
	return s
}

type layerNormGPU struct {
	axis int
	eps  float32
}

func (o layerNormGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) < 1 || in[0] == nil || !allF32(in...) || !onlyFirstOutput(st) {
		return nil, nil, false
	}
	x := in[0]
	xs := x.Shape()
	if !lastAxis(o.axis, len(xs)) {
		return nil, nil, false
	}
	D := xs[len(xs)-1]
	var scale, bias *tensor.Tensor
	if len(in) > 1 {
		scale = in[1]
	}
	if len(in) > 2 {
		bias = in[2]
	}
	if (scale != nil && scale.Numel() != D) || (bias != nil && bias.Numel() != D) {
		return nil, nil, false
	}
	return c.layerNorm(x, nil, scale, bias, D, o.eps)
}

// layerNorm encodes y = LN(x [+ add]) · scale + bias over the last dim D;
// with add, the sum is also an output (AddLayerNorm).
func (c *gpuCtx) layerNorm(x, add, scale, bias *tensor.Tensor, D int, eps float32) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	ts := []*tensor.Tensor{x}
	if add != nil {
		ts = append(ts, add)
	}
	rs, ok := c.regions(ts...)
	if !ok {
		return nil, nil, false
	}
	var rScale, rBias metal.Region
	if scale != nil {
		r, ok := c.regions(scale)
		if !ok {
			return nil, nil, false
		}
		rScale = r[0]
	} else if rScale, ok = c.onesVec(D); !ok {
		return nil, nil, false
	}
	if bias != nil {
		r, ok := c.regions(bias)
		if !ok {
			return nil, nil, false
		}
		rBias = r[0]
	}
	rows := x.Numel() / D
	out := c.out(x.Shape()...)
	var sum *tensor.Tensor
	outs := []*tensor.Tensor{out}
	if add != nil {
		sum = c.out(x.Shape()...)
		outs = []*tensor.Tensor{sum, out}
	}
	ro, ok := c.regions(outs...)
	if !ok {
		c.release(outs...)
		return nil, nil, false
	}
	return outs, func(e *metal.Encoder) {
		src := rs[0]
		dst := ro[0]
		if add != nil {
			e.Binary(metal.OpAdd, rs[0], rs[1], ro[0], x.Numel(), x.Numel(), add.Numel())
			src, dst = ro[0], ro[1]
		}
		e.LayerNormMod(src, dst, rScale, rows, D, D, D, eps)
		if bias != nil {
			e.AddBias(dst, rBias, rows, D, D)
		}
	}, true
}

type addLayerNormGPU struct {
	axis int
	eps  float32
}

func (o addLayerNormGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) < 2 || in[0] == nil || in[1] == nil || !allF32(in...) || !in[0].Shape().Equal(in[1].Shape()) {
		return nil, nil, false
	}
	xs := in[0].Shape()
	if !lastAxis(o.axis, len(xs)) {
		return nil, nil, false
	}
	D := xs[len(xs)-1]
	var scale, bias *tensor.Tensor
	if len(in) > 2 {
		scale = in[2]
	}
	if len(in) > 3 {
		bias = in[3]
	}
	if (scale != nil && scale.Numel() != D) || (bias != nil && bias.Numel() != D) {
		return nil, nil, false
	}
	return c.layerNorm(in[0], in[1], scale, bias, D, o.eps)
}

type softmaxGPU struct{ axis int }

func (o softmaxGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) < 1 || in[0] == nil || !allF32(in[0]) {
		return nil, nil, false
	}
	x := in[0]
	xs := x.Shape()
	// Opset < 13 flattens at axis; only the last-axis form is taken here.
	if len(xs) == 0 || !lastAxis(o.axis, len(xs)) {
		return nil, nil, false
	}
	D := xs[len(xs)-1]
	rx, ok := c.regions(x)
	if !ok {
		return nil, nil, false
	}
	out := c.out(xs...)
	ro, ok := c.regions(out)
	if !ok {
		c.release(out)
		return nil, nil, false
	}
	n := x.Numel()
	return []*tensor.Tensor{out}, func(e *metal.Encoder) {
		e.CopyF32(rx[0], ro[0], n)
		e.SoftmaxRows(ro[0], n/D, D, D, 1)
	}, true
}

type reduceMeanGPU struct {
	keep     bool
	attrAxes []int64
}

func (o reduceMeanGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) < 1 || in[0] == nil || !allF32(in[0]) {
		return nil, nil, false
	}
	x := in[0]
	xs := x.Shape()
	axes := o.attrAxes
	if axes == nil && len(in) > 1 && in[1] != nil {
		if in[1].DType() != tensor.I64 {
			return nil, nil, false
		}
		axes = in[1].I64()
	}
	if len(xs) == 0 || len(axes) != 1 || !lastAxis(int(axes[0]), len(xs)) {
		return nil, nil, false
	}
	D := xs[len(xs)-1]
	rx, ok := c.regions(x)
	if !ok {
		return nil, nil, false
	}
	oshape := append([]int{}, xs[:len(xs)-1]...)
	if o.keep {
		oshape = append(oshape, 1)
	}
	out := c.out(oshape...)
	ro, ok := c.regions(out)
	if !ok {
		c.release(out)
		return nil, nil, false
	}
	rows := x.Numel() / D
	return []*tensor.Tensor{out}, func(e *metal.Encoder) { e.ReduceRows(rx[0], ro[0], rows, D, D, true) }, true
}

type transposeGPU struct{ perm []int64 }

func (o transposeGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) < 1 || in[0] == nil || !allF32(in[0]) {
		return nil, nil, false
	}
	x := in[0]
	xs := x.Shape()
	r := len(xs)
	if r == 0 || r > 6 {
		return nil, nil, false
	}
	perm := make([]int, r)
	for i := range perm {
		perm[i] = r - 1 - i
		if o.perm != nil {
			if len(o.perm) != r {
				return nil, nil, false
			}
			perm[i] = int(o.perm[i])
		}
	}
	rx, ok := c.regions(x)
	if !ok {
		return nil, nil, false
	}
	oshape := make([]int, r)
	for i, p := range perm {
		oshape[i] = xs[p]
	}
	out := c.out(oshape...)
	ro, ok := c.regions(out)
	if !ok {
		c.release(out)
		return nil, nil, false
	}
	dims := append([]int{}, xs...)
	return []*tensor.Tensor{out}, func(e *metal.Encoder) { e.Transpose(rx[0], ro[0], dims, perm) }, true
}

// ---- attention ----

// sdpaGPU is ingot.SDPA as, per (batch, head), S = scale·Q·Kᵀ (+ mask),
// softmax, O = S·V over strided views of the operands in the layouts the
// CPU op accepts (see ops/attention.go).
type sdpaGPU struct {
	scale            float32
	aLay, bLay, vLay int
	strideOut        bool
}

func (o sdpaGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) < 3 || in[0] == nil || in[1] == nil || in[2] == nil || !allF32(in...) {
		return nil, nil, false
	}
	q, k, v := in[0], in[1], in[2]
	qs, ks, vs := q.Shape(), k.Shape(), v.Shape()
	rank3 := len(qs) == 3 && len(ks) == 3 && len(vs) == 3 && o.aLay == 0 && o.bLay == 0 && o.vLay == 0 && !o.strideOut
	if rank3 { // batch and heads pre-flattened: [1, B·H, T, dh]
		qs, ks, vs = tensor.Shape{1, qs[0], qs[1], qs[2]}, tensor.Shape{1, ks[0], ks[1], ks[2]}, tensor.Shape{1, vs[0], vs[1], vs[2]}
	}
	if len(qs) != 4 || len(ks) != 4 || len(vs) != 4 {
		return nil, nil, false
	}
	var B, H, T, dh, Tk int
	if o.aLay == 1 {
		B, T, H, dh = qs[0], qs[1], qs[2], qs[3]
	} else {
		B, H, T, dh = qs[0], qs[1], qs[2], qs[3]
	}
	switch o.bLay {
	case 1:
		Tk = ks[1]
	case 2:
		Tk = ks[2]
	default:
		Tk = ks[3]
	}
	var mask *tensor.Tensor
	if len(in) > 3 && in[3] != nil {
		if in[3].Numel() != T*Tk {
			return nil, nil, false
		}
		mask = in[3]
	}
	ts := []*tensor.Tensor{q, k, v}
	if mask != nil {
		ts = append(ts, mask)
	}
	rs, ok := c.regions(ts...)
	if !ok {
		return nil, nil, false
	}
	oshape := []int{B, H, T, dh}
	if o.strideOut {
		oshape = []int{B, T, H, dh}
	}
	if rank3 {
		oshape = []int{H, T, dh}
	}
	out := c.out(oshape...)
	s := c.out(T, Tk) // one head's scores, reused head by head
	ro, ok := c.regions(out, s)
	if !ok {
		c.release(out, s)
		return nil, nil, false
	}
	// Scores are dead once this node's dispatches are recorded; later
	// writers of the recycled buffer run after them in stream order (and
	// CPU writers only after a flush).
	c.release(s)
	at := func(r metal.Region, elems int) metal.Region { return metal.Region{B: r.B, Off: r.Off + 4*elems} }
	o2 := o
	return []*tensor.Tensor{out}, func(e *metal.Encoder) {
		for b := range B {
			for h := range H {
				var qo, lq, ko, lk, vo, lv, oo, lo int
				if o2.aLay == 1 {
					qo, lq = (b*T*H+h)*dh, H*dh
				} else {
					qo, lq = (b*H+h)*T*dh, dh
				}
				switch o2.bLay {
				case 1:
					ko, lk = (b*Tk*H+h)*dh, H*dh
				case 2:
					ko, lk = (b*H+h)*Tk*dh, dh
				default: // Kᵀ stored [dh, Tk]
					ko, lk = (b*H+h)*dh*Tk, Tk
				}
				if o2.vLay == 1 {
					vo, lv = (b*Tk*H+h)*dh, H*dh
				} else {
					vo, lv = (b*H+h)*Tk*dh, dh
				}
				if o2.strideOut {
					oo, lo = (b*T*H+h)*dh, H*dh
				} else {
					oo, lo = (b*H+h)*T*dh, dh
				}
				e.Gemm(metal.Gemm{M: T, N: Tk, K: dh, A: at(rs[0], qo), LDA: lq, B: at(rs[1], ko), LDB: lk, C: ro[1],
					TransB: o2.bLay != 0})
				if mask != nil {
					e.SoftmaxRowsMasked(ro[1], rs[3], T, Tk, Tk, Tk, o2.scale)
				} else {
					e.SoftmaxRows(ro[1], T, Tk, Tk, o2.scale)
				}
				e.Gemm(metal.Gemm{M: T, N: dh, K: Tk, A: ro[1], B: at(rs[2], vo), LDB: lv, C: at(ro[0], oo), LDC: lo})
			}
		}
	}, true
}

// ---- gathers and concatenation ----

// gatherGPU is Gather with its indices read on the CPU (index tensors are
// never GPU outputs) and passed as constant data. Along axis 0 (embedding
// lookups) it is one row gather of up to 1024 indices; along an inner axis
// (e.g. pooling one token) one strided copy per index, up to 64.
type gatherGPU struct{ axis int }

func (o gatherGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) != 2 || in[0] == nil || in[1] == nil || !allF32(in[0]) || !in[1].IsContiguous() {
		return nil, nil, false
	}
	data, ix := in[0], in[1]
	ds := data.Shape()
	axis := o.axis
	if axis < 0 {
		axis += len(ds)
	}
	limit := 1024
	if axis != 0 {
		limit = 64
	}
	if axis < 0 || axis >= len(ds) || ix.Numel() == 0 || ix.Numel() > limit {
		return nil, nil, false
	}
	V := ds[axis]
	idx := make([]uint32, ix.Numel())
	for i := range idx {
		var v int64
		switch ix.DType() {
		case tensor.I64:
			v = ix.I64()[i]
		case tensor.I32:
			v = int64(ix.I32()[i])
		default:
			return nil, nil, false
		}
		if v < 0 {
			v += int64(V)
		}
		if v < 0 || v >= int64(V) {
			return nil, nil, false // the CPU op reports the error
		}
		idx[i] = uint32(v)
	}
	rd, ok := c.regions(data)
	if !ok {
		return nil, nil, false
	}
	outer, inner := 1, 1
	for _, d := range ds[:axis] {
		outer *= d
	}
	for _, d := range ds[axis+1:] {
		inner *= d
	}
	oshape := append(append(append([]int{}, ds[:axis]...), ix.Shape()...), ds[axis+1:]...)
	out := c.out(oshape...)
	ro, ok := c.regions(out)
	if !ok {
		c.release(out)
		return nil, nil, false
	}
	if out.Numel() == 0 {
		return []*tensor.Tensor{out}, func(*metal.Encoder) {}, true
	}
	if axis == 0 {
		return []*tensor.Tensor{out}, func(e *metal.Encoder) { e.GatherRowsConst(rd[0], ro[0], idx, inner) }, true
	}
	n := len(idx)
	return []*tensor.Tensor{out}, func(e *metal.Encoder) {
		for j, v := range idx {
			src := metal.Region{B: rd[0].B, Off: rd[0].Off + 4*int(v)*inner}
			dst := metal.Region{B: ro[0].B, Off: ro[0].Off + 4*j*inner}
			e.Copy2D(src, dst, outer, inner, V*inner, n*inner)
		}
	}, true
}

type concatGPU struct{ axis int }

func (o concatGPU) prepare(c *gpuCtx, st *step, in []*tensor.Tensor) ([]*tensor.Tensor, func(*metal.Encoder), bool) {
	if len(in) == 0 || !allF32(in...) {
		return nil, nil, false
	}
	for _, t := range in {
		if t == nil {
			return nil, nil, false
		}
	}
	xs := in[0].Shape()
	axis := o.axis
	if axis < 0 {
		axis += len(xs)
	}
	if axis < 0 || axis >= len(xs) {
		return nil, nil, false
	}
	oshape := append([]int{}, xs...)
	oshape[axis] = 0
	for _, t := range in {
		ts := t.Shape()
		if len(ts) != len(xs) {
			return nil, nil, false
		}
		for d := range ts {
			if d != axis && ts[d] != xs[d] {
				return nil, nil, false
			}
		}
		oshape[axis] += ts[axis]
	}
	outer, inner := 1, 1
	for _, d := range xs[:axis] {
		outer *= d
	}
	for _, d := range xs[axis+1:] {
		inner *= d
	}
	rs, ok := c.regions(in...)
	if !ok {
		return nil, nil, false
	}
	out := c.out(oshape...)
	ro, ok := c.regions(out)
	if !ok {
		c.release(out)
		return nil, nil, false
	}
	width := oshape[axis] * inner
	cols := make([]int, len(in))
	for i, t := range in {
		cols[i] = t.Shape()[axis] * inner
	}
	return []*tensor.Tensor{out}, func(e *metal.Encoder) {
		off := 0
		for i, r := range rs {
			if cols[i] > 0 && outer > 0 {
				e.Copy2D(r, metal.Region{B: ro[0].B, Off: ro[0].Off + 4*off}, outer, cols[i], cols[i], width)
			}
			off += cols[i]
		}
	}, true
}
