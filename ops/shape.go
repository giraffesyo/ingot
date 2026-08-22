package ops

import (
	"fmt"
	"github.com/giraffesyo/ingot/kernels/par"

	"github.com/giraffesyo/ingot/tensor"
)

// ---- helpers ----

func normAxis(axis, rank int) (int, error) {
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return 0, fmt.Errorf("axis %d out of range for rank %d", axis, rank)
	}
	return axis, nil
}

// copyView copies t's bytes into a new tensor of the same dtype and `shape`.
func copyView(ctx *Ctx, t *tensor.Tensor, shape tensor.Shape) *tensor.Tensor {
	out := ctx.New(t.DType(), shape...)
	copy(out.Bytes(), t.Bytes())
	return out
}

// ---- Reshape ----

type reshapeOp struct {
	n         NodeInfo
	allowZero bool
	attrShape []int64 // opset < 5
}

func (o *reshapeOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 1 || in[0] == nil {
		return nil, o.n.Errorf("missing input")
	}
	x := in[0]
	var want []int64
	if o.attrShape != nil {
		want = o.attrShape
	} else {
		if len(in) < 2 || in[1] == nil {
			return nil, o.n.Errorf("missing shape input")
		}
		want = in[1].I64()
	}
	xs := x.Shape()
	shape := make(tensor.Shape, len(want))
	neg := -1
	known := 1
	for i, d := range want {
		switch {
		case d == 0 && !o.allowZero:
			if i >= len(xs) {
				return nil, o.n.Errorf("0 dim at %d beyond input rank", i)
			}
			shape[i] = xs[i]
		case d == -1:
			if neg >= 0 {
				return nil, o.n.Errorf("multiple -1 dims")
			}
			neg = i
			continue
		default:
			shape[i] = int(d)
		}
		known *= shape[i]
	}
	if neg >= 0 {
		if known == 0 {
			shape[neg] = 0
		} else {
			shape[neg] = x.Numel() / known
		}
	}
	if shape.Numel() != x.Numel() {
		return nil, o.n.Errorf("cannot reshape %v to %v", xs, shape)
	}
	return []*tensor.Tensor{copyView(ctx, x, shape)}, nil
}

// ---- Flatten ----

type flattenOp struct {
	n    NodeInfo
	axis int
}

func (o *flattenOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	xs := x.Shape()
	axis := o.axis
	if axis < 0 {
		axis += len(xs)
	}
	if axis < 0 || axis > len(xs) {
		return nil, o.n.Errorf("axis %d out of range", o.axis)
	}
	a, b := 1, 1
	for _, d := range xs[:axis] {
		a *= d
	}
	for _, d := range xs[axis:] {
		b *= d
	}
	return []*tensor.Tensor{copyView(ctx, x, tensor.Shape{a, b})}, nil
}

// ---- Squeeze / Unsqueeze ----

type squeezeOp struct {
	n        NodeInfo
	attrAxes []int64 // opset < 13
	unsq     bool
}

func (o *squeezeOp) axes(in []*tensor.Tensor) []int64 {
	if o.attrAxes != nil {
		return o.attrAxes
	}
	if len(in) > 1 && in[1] != nil {
		return in[1].I64()
	}
	return nil
}

func (o *squeezeOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	xs := x.Shape()
	axes := o.axes(in)
	var shape tensor.Shape
	if o.unsq {
		rank := len(xs) + len(axes)
		set := map[int]bool{}
		for _, a := range axes {
			ax := int(a)
			if ax < 0 {
				ax += rank
			}
			if ax < 0 || ax >= rank || set[ax] {
				return nil, o.n.Errorf("bad axes %v", axes)
			}
			set[ax] = true
		}
		shape = make(tensor.Shape, 0, rank)
		src := 0
		for i := 0; i < rank; i++ {
			if set[i] {
				shape = append(shape, 1)
			} else {
				shape = append(shape, xs[src])
				src++
			}
		}
	} else {
		set := map[int]bool{}
		for _, a := range axes {
			ax := int(a)
			if ax < 0 {
				ax += len(xs)
			}
			if ax < 0 || ax >= len(xs) || xs[ax] != 1 {
				return nil, o.n.Errorf("cannot squeeze axis %d of %v", a, xs)
			}
			set[ax] = true
		}
		shape = make(tensor.Shape, 0, len(xs))
		for i, d := range xs {
			if set[i] || (axes == nil && d == 1) {
				continue
			}
			shape = append(shape, d)
		}
	}
	return []*tensor.Tensor{copyView(ctx, x, shape)}, nil
}

// ---- Transpose ----

type transposeOp struct {
	n    NodeInfo
	perm []int64
}

func (o *transposeOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	xs := x.Shape()
	r := len(xs)
	perm := make([]int, r)
	if o.perm == nil {
		for i := range perm {
			perm[i] = r - 1 - i
		}
	} else {
		if len(o.perm) != r {
			return nil, o.n.Errorf("perm %v rank mismatch %v", o.perm, xs)
		}
		for i, p := range o.perm {
			perm[i] = int(p)
		}
	}
	oshape := make(tensor.Shape, r)
	for i, p := range perm {
		oshape[i] = xs[p]
	}
	out := ctx.New(x.DType(), oshape...)
	transposeBytes(x, out, perm)
	return []*tensor.Tensor{out}, nil
}

// transposeBytes performs a generic N-d permutation copy for any dtype.
func transposeBytes(x, out *tensor.Tensor, perm []int) {
	xs := x.Shape()
	os := out.Shape()
	r := len(xs)
	if r == 0 {
		copy(out.Bytes(), x.Bytes())
		return
	}
	xst := xs.Strides()
	// stride in source for each output dim
	sst := make([]int, r)
	for i, p := range perm {
		sst[i] = xst[p]
	}
	n := out.Numel()
	esz := x.DType().Size()
	src, dst := x.Bytes(), out.Bytes()
	idx := make([]int, r)
	off := 0
	inner := os[r-1]
	is := sst[r-1]
	switch {
	case esz == 4:
		s4 := x.F32()
		d4 := out.F32()
		for oi := 0; oi < n; oi += inner {
			for j := 0; j < inner; j++ {
				d4[oi+j] = s4[off+j*is]
			}
			off = advance(idx, os, sst, r-2, off)
		}
	default:
		for oi := 0; oi < n; oi += inner {
			for j := 0; j < inner; j++ {
				copy(dst[(oi+j)*esz:(oi+j+1)*esz], src[(off+j*is)*esz:(off+j*is+1)*esz])
			}
			off = advance(idx, os, sst, r-2, off)
		}
	}
}

// advance increments an odometer over dims [0, top] and returns the new
// source offset.
func advance(idx, shape, strides []int, top, off int) int {
	for d := top; d >= 0; d-- {
		idx[d]++
		off += strides[d]
		if idx[d] < shape[d] {
			return off
		}
		off -= strides[d] * shape[d]
		idx[d] = 0
	}
	return off
}

// ---- Concat ----

type concatOp struct {
	n    NodeInfo
	axis int
}

func (o *concatOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) == 0 {
		return nil, o.n.Errorf("no inputs")
	}
	first := in[0]
	xs := first.Shape()
	axis, err := normAxis(o.axis, len(xs))
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	oshape := xs.Clone()
	oshape[axis] = 0
	for _, t := range in {
		if t == nil {
			return nil, o.n.Errorf("nil input")
		}
		ts := t.Shape()
		if len(ts) != len(xs) || t.DType() != first.DType() {
			return nil, o.n.Errorf("rank/dtype mismatch %v vs %v", ts, xs)
		}
		for d := range ts {
			if d != axis && ts[d] != xs[d] {
				return nil, o.n.Errorf("shape mismatch %v vs %v", ts, xs)
			}
		}
		oshape[axis] += ts[axis]
	}
	out := ctx.New(first.DType(), oshape...)
	esz := first.DType().Size()
	outer := 1
	for _, d := range xs[:axis] {
		outer *= d
	}
	inner := 1
	for _, d := range xs[axis+1:] {
		inner *= d
	}
	dst := out.Bytes()
	rowOut := oshape[axis] * inner * esz
	// Each (outer index, input) pair is one contiguous copy; split large
	// copies into pieces so the memcpy bandwidth of several cores is used.
	const piece = 256 << 10
	offs := make([]int, len(in)+1)
	for i, t := range in {
		offs[i+1] = offs[i] + t.Dim(axis)*inner*esz
	}
	maxChunk := 0
	for _, t := range in {
		maxChunk = max(maxChunk, t.Dim(axis)*inner*esz)
	}
	pieces := max(1, (maxChunk+piece-1)/piece)
	par.For(outer*len(in)*pieces, 1, func(t, _ int) {
		pc := t % pieces
		t /= pieces
		ii := t % len(in)
		oi := t / len(in)
		chunk := offs[ii+1] - offs[ii]
		lo := pc * piece
		if lo >= chunk {
			return
		}
		hi := min(lo+piece, chunk)
		pos := oi*rowOut + offs[ii]
		copy(dst[pos+lo:pos+hi], in[ii].Bytes()[oi*chunk+lo:oi*chunk+hi])
	})
	return []*tensor.Tensor{out}, nil
}

// ---- Gather ----

type gatherOp struct {
	n    NodeInfo
	axis int
}

func (o *gatherOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) != 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("need data and indices")
	}
	x, ind := in[0], in[1]
	xs := x.Shape()
	axis, err := normAxis(o.axis, len(xs))
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	var idx []int64
	switch ind.DType() {
	case tensor.I64:
		idx = ind.I64()
	case tensor.I32:
		for _, v := range ind.I32() {
			idx = append(idx, int64(v))
		}
	default:
		return nil, o.n.Errorf("indices must be int32/int64")
	}
	is := ind.Shape()
	oshape := make(tensor.Shape, 0, len(xs)+len(is)-1)
	oshape = append(oshape, xs[:axis]...)
	oshape = append(oshape, is...)
	oshape = append(oshape, xs[axis+1:]...)
	out := ctx.New(x.DType(), oshape...)
	esz := x.DType().Size()
	outer := 1
	for _, d := range xs[:axis] {
		outer *= d
	}
	inner := 1
	for _, d := range xs[axis+1:] {
		inner *= d
	}
	A := xs[axis]
	src, dst := x.Bytes(), out.Bytes()
	chunk := inner * esz
	pos := 0
	for oi := 0; oi < outer; oi++ {
		for _, i := range idx {
			if i < 0 {
				i += int64(A)
			}
			if i < 0 || int(i) >= A {
				return nil, o.n.Errorf("index %d out of range [0,%d)", i, A)
			}
			s := (oi*A + int(i)) * chunk
			copy(dst[pos:pos+chunk], src[s:s+chunk])
			pos += chunk
		}
	}
	return []*tensor.Tensor{out}, nil
}

// ---- Shape ----

type shapeOp struct {
	n          NodeInfo
	start, end int64
	hasEnd     bool
}

func (o *shapeOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	xs := in[0].Shape()
	r := int64(len(xs))
	s, e := o.start, r
	if o.hasEnd {
		e = o.end
	}
	if s < 0 {
		s += r
	}
	if e < 0 {
		e += r
	}
	s = max(0, min(s, r))
	e = max(s, min(e, r))
	out := ctx.New(tensor.I64, int(e-s))
	of := out.I64()
	for i := range of {
		of[i] = int64(xs[int(s)+i])
	}
	return []*tensor.Tensor{out}, nil
}

// ---- Constant ----

type constantOp struct {
	n NodeInfo
	t *tensor.Tensor
}

func (o *constantOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	return []*tensor.Tensor{o.t}, nil
}

func buildConstant(n NodeInfo) (Op, error) {
	a := n.Attrs
	switch {
	case a.Tensor("value") != nil:
		return &constantOp{n, a.Tensor("value")}, nil
	case a.Has("value_float"):
		return &constantOp{n, tensor.Scalar(a.Float("value_float", 0))}, nil
	case a.Has("value_floats"):
		v := a.Floats("value_floats", nil)
		return &constantOp{n, tensor.FromF32(append([]float32(nil), v...), len(v))}, nil
	case a.Has("value_int"):
		return &constantOp{n, tensor.FromI64([]int64{a.Int("value_int", 0)})}, nil
	case a.Has("value_ints"):
		v := a.Ints("value_ints", nil)
		return &constantOp{n, tensor.FromI64(append([]int64(nil), v...), len(v))}, nil
	}
	return nil, n.Errorf("unsupported Constant attribute set")
}

// ---- Slice (opset >= 10: inputs) ----

type sliceOp struct {
	n NodeInfo
	// opset < 10 attrs
	attrStarts, attrEnds, attrAxes []int64
}

func (o *sliceOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	xs := x.Shape()
	r := len(xs)
	var starts, ends, axes, steps []int64
	if o.attrStarts != nil {
		starts, ends, axes = o.attrStarts, o.attrEnds, o.attrAxes
	} else {
		if len(in) < 3 || in[1] == nil || in[2] == nil {
			return nil, o.n.Errorf("need starts and ends")
		}
		starts, ends = asI64(in[1]), asI64(in[2])
		if len(in) > 3 && in[3] != nil {
			axes = asI64(in[3])
		}
		if len(in) > 4 && in[4] != nil {
			steps = asI64(in[4])
		}
	}
	if axes == nil {
		axes = make([]int64, len(starts))
		for i := range axes {
			axes[i] = int64(i)
		}
	}
	if steps == nil {
		steps = make([]int64, len(starts))
		for i := range steps {
			steps[i] = 1
		}
	}
	// Per-dim start/step/count.
	st := make([]int, r)
	sp := make([]int, r)
	cnt := xs.Clone()
	for i := range sp {
		sp[i] = 1
	}
	for i, a := range axes {
		ax, err := normAxis(int(a), r)
		if err != nil {
			return nil, o.n.Errorf("%v", err)
		}
		dim := int64(xs[ax])
		s, e, k := starts[i], ends[i], steps[i]
		if k == 0 {
			return nil, o.n.Errorf("step 0")
		}
		if s < 0 {
			s += dim
		}
		if e < 0 {
			e += dim
		}
		if k > 0 {
			s = max(0, min(s, dim))
			e = max(0, min(e, dim))
			c := int64(0)
			if e > s {
				c = (e - s + k - 1) / k
			}
			st[ax], sp[ax], cnt[ax] = int(s), int(k), int(c)
		} else {
			s = max(-1, min(s, dim-1))
			e = max(-1, min(e, dim-1))
			c := int64(0)
			if s > e {
				c = (s - e - k - 1) / (-k)
			}
			st[ax], sp[ax], cnt[ax] = int(s), int(k), int(c)
		}
	}
	out := ctx.New(x.DType(), cnt...)
	if out.Numel() == 0 {
		return []*tensor.Tensor{out}, nil
	}
	esz := x.DType().Size()
	xst := xs.Strides()
	src, dst := x.Bytes(), out.Bytes()
	idx := make([]int, r)
	base := 0
	for d := 0; d < r; d++ {
		base += st[d] * xst[d]
	}
	strides := make([]int, r)
	for d := 0; d < r; d++ {
		strides[d] = sp[d] * xst[d]
	}
	n := out.Numel()
	inner := cnt[r-1]
	is := strides[r-1]
	off := base
	for oi := 0; oi < n; oi += inner {
		if is == 1 {
			copy(dst[oi*esz:(oi+inner)*esz], src[off*esz:(off+inner)*esz])
		} else {
			for j := 0; j < inner; j++ {
				copy(dst[(oi+j)*esz:(oi+j+1)*esz], src[(off+j*is)*esz:(off+j*is+1)*esz])
			}
		}
		off = advance(idx, cnt, strides, r-2, off)
	}
	return []*tensor.Tensor{out}, nil
}

func asI64(t *tensor.Tensor) []int64 {
	switch t.DType() {
	case tensor.I64:
		return t.I64()
	case tensor.I32:
		out := make([]int64, t.Numel())
		for i, v := range t.I32() {
			out[i] = int64(v)
		}
		return out
	}
	panic(fmt.Sprintf("ops: expected integer tensor, got %s", t.DType()))
}

// ---- Cast ----

type castOp struct {
	n  NodeInfo
	to tensor.DType
}

// onnxDType maps ONNX TensorProto.DataType values to tensor.DType.
func onnxDType(v int64) (tensor.DType, bool) {
	switch v {
	case 1:
		return tensor.F32, true
	case 2:
		return tensor.U8, true
	case 3:
		return tensor.I8, true
	case 6:
		return tensor.I32, true
	case 7:
		return tensor.I64, true
	case 9:
		return tensor.Bool, true
	case 10, 11, 16: // float16, double, bfloat16 → computed as f32
		return tensor.F32, true
	}
	return 0, false
}

func (o *castOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	out, err := castTo(ctx, x, o.to)
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	return []*tensor.Tensor{out}, nil
}

func castTo(ctx *Ctx, x *tensor.Tensor, to tensor.DType) (*tensor.Tensor, error) {
	if x.DType() == to {
		return x.Clone(), nil
	}
	out := ctx.New(to, x.Shape()...)
	n := x.Numel()
	// Read as float64 or int64 then write.
	get := func(i int) float64 {
		switch x.DType() {
		case tensor.F32:
			return float64(x.F32()[i])
		case tensor.I64:
			return float64(x.I64()[i])
		case tensor.I32:
			return float64(x.I32()[i])
		case tensor.U8:
			return float64(x.U8()[i])
		case tensor.I8:
			return float64(x.I8()[i])
		case tensor.Bool:
			if x.Bool()[i] {
				return 1
			}
			return 0
		}
		return 0
	}
	if !(x.DType() == tensor.F32 || x.DType() == tensor.I64 || x.DType() == tensor.I32 || x.DType() == tensor.U8 || x.DType() == tensor.I8 || x.DType() == tensor.Bool) {
		return nil, fmt.Errorf("cast from %s unsupported", x.DType())
	}
	switch to {
	case tensor.F32:
		d := out.F32()
		for i := 0; i < n; i++ {
			d[i] = float32(get(i))
		}
	case tensor.I64:
		d := out.I64()
		for i := 0; i < n; i++ {
			d[i] = int64(get(i))
		}
	case tensor.I32:
		d := out.I32()
		for i := 0; i < n; i++ {
			d[i] = int32(get(i))
		}
	case tensor.U8:
		d := out.U8()
		for i := 0; i < n; i++ {
			d[i] = uint8(get(i))
		}
	case tensor.I8:
		d := out.I8()
		for i := 0; i < n; i++ {
			d[i] = int8(get(i))
		}
	case tensor.Bool:
		d := out.Bool()
		for i := 0; i < n; i++ {
			d[i] = get(i) != 0
		}
	default:
		return nil, fmt.Errorf("cast to %s unsupported", to)
	}
	return out, nil
}

// ---- Expand ----

type expandOp struct{ n NodeInfo }

func (o *expandOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	want := intsToShape(asI64(in[1]))
	os, err := broadcastShape(x.Shape(), want)
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	out, err := broadcastTo(ctx, x, os)
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	return []*tensor.Tensor{out}, nil
}

// broadcastTo materialises x broadcast to shape os (any dtype).
func broadcastTo(ctx *Ctx, x *tensor.Tensor, os tensor.Shape) (*tensor.Tensor, error) {
	out := ctx.New(x.DType(), os...)
	if x.Shape().Equal(os) {
		copy(out.Bytes(), x.Bytes())
		return out, nil
	}
	st := broadcastStrides(x.Shape(), os)
	esz := x.DType().Size()
	src, dst := x.Bytes(), out.Bytes()
	r := len(os)
	n := out.Numel()
	if r == 0 {
		copy(dst, src)
		return out, nil
	}
	idx := make([]int, r)
	inner := os[r-1]
	is := st[r-1]
	off := 0
	for oi := 0; oi < n; oi += inner {
		if is == 1 {
			copy(dst[oi*esz:(oi+inner)*esz], src[off*esz:(off+inner)*esz])
		} else { // is == 0
			for j := 0; j < inner; j++ {
				copy(dst[(oi+j)*esz:(oi+j+1)*esz], src[off*esz:(off+1)*esz])
			}
		}
		off = advance(idx, os, st, r-2, off)
	}
	return out, nil
}

// ---- ConstantOfShape ----

type constantOfShapeOp struct {
	n   NodeInfo
	val *tensor.Tensor // 1-element, dtype gives output dtype
}

func (o *constantOfShapeOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	shape := intsToShape(asI64(in[0]))
	out := ctx.New(o.val.DType(), shape...)
	esz := o.val.DType().Size()
	v := o.val.Bytes()
	dst := out.Bytes()
	for i := 0; i < out.Numel(); i++ {
		copy(dst[i*esz:(i+1)*esz], v)
	}
	return []*tensor.Tensor{out}, nil
}

// ---- Where ----

type whereOp struct{ n NodeInfo }

func (o *whereOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	c, a, b := in[0], in[1], in[2]
	if c.DType() != tensor.Bool {
		return nil, o.n.Errorf("condition must be bool")
	}
	if a.DType() != b.DType() {
		return nil, o.n.Errorf("X/Y dtype mismatch")
	}
	s1, err := broadcastShape(c.Shape(), a.Shape())
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	os, err := broadcastShape(s1, b.Shape())
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	cb, err := broadcastTo(ctx, c, os)
	if err != nil {
		return nil, err
	}
	ab, err := broadcastTo(ctx, a, os)
	if err != nil {
		return nil, err
	}
	bb, err := broadcastTo(ctx, b, os)
	if err != nil {
		return nil, err
	}
	out := ctx.New(a.DType(), os...)
	esz := a.DType().Size()
	cond := cb.Bool()
	as, bs, ds := ab.Bytes(), bb.Bytes(), out.Bytes()
	for i, t := range cond {
		if t {
			copy(ds[i*esz:(i+1)*esz], as[i*esz:(i+1)*esz])
		} else {
			copy(ds[i*esz:(i+1)*esz], bs[i*esz:(i+1)*esz])
		}
	}
	return []*tensor.Tensor{out}, nil
}

// ---- comparison / logical ----

type compareOp struct {
	n  NodeInfo
	fn func(x, y float64) bool
}

func (o *compareOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	a, b := in[0], in[1]
	os, err := broadcastShape(a.Shape(), b.Shape())
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	af, err := castTo(ctx, a, tensor.F32)
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	bf, err := castTo(ctx, b, tensor.F32)
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	// Note: int64 compared via float32 loses precision beyond 2^24; acceptable
	// for shape arithmetic. TODO: native int64 path.
	ab, _ := broadcastTo(ctx, af, os)
	bb, _ := broadcastTo(ctx, bf, os)
	out := ctx.New(tensor.Bool, os...)
	x, y, d := ab.F32(), bb.F32(), out.Bool()
	for i := range d {
		d[i] = o.fn(float64(x[i]), float64(y[i]))
	}
	return []*tensor.Tensor{out}, nil
}

type notOp struct{ n NodeInfo }

func (o *notOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	out := ctx.New(tensor.Bool, x.Shape()...)
	s, d := x.Bool(), out.Bool()
	for i := range d {
		d[i] = !s[i]
	}
	return []*tensor.Tensor{out}, nil
}

// ---- Range ----

type rangeOp struct{ n NodeInfo }

func (o *rangeOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	s, l, d := in[0], in[1], in[2]
	switch s.DType() {
	case tensor.I64:
		a, b, c := s.I64()[0], l.I64()[0], d.I64()[0]
		n := max(0, int((b-a+c-1)/c))
		if c < 0 {
			n = max(0, int((a-b-c-1)/(-c)))
		}
		out := ctx.New(tensor.I64, n)
		of := out.I64()
		for i := range of {
			of[i] = a + int64(i)*c
		}
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		a, b, c := s.F32()[0], l.F32()[0], d.F32()[0]
		n := max(0, int(ceilDiv32(b-a, c)))
		out := ctx.New(tensor.F32, n)
		of := out.F32()
		for i := range of {
			of[i] = a + float32(i)*c
		}
		return []*tensor.Tensor{out}, nil
	}
	return nil, o.n.Errorf("unsupported dtype %s", s.DType())
}

func ceilDiv32(a, b float32) float32 {
	q := a / b
	if float32(int(q)) == q {
		return q
	}
	return float32(int(q) + 1)
}

// ---- Split ----

type splitOp struct {
	n         NodeInfo
	axis      int
	attrSplit []int64
	numOut    int
}

func (o *splitOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	xs := x.Shape()
	axis, err := normAxis(o.axis, len(xs))
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	var split []int64
	switch {
	case o.attrSplit != nil:
		split = o.attrSplit
	case len(in) > 1 && in[1] != nil:
		split = asI64(in[1])
	default:
		k := o.numOut
		if k == 0 {
			k = o.n.NumOut
		}
		d := xs[axis]
		each := (d + k - 1) / k
		for rem := d; rem > 0; rem -= each {
			split = append(split, int64(min(each, rem)))
		}
	}
	outer := 1
	for _, d := range xs[:axis] {
		outer *= d
	}
	inner := 1
	for _, d := range xs[axis+1:] {
		inner *= d
	}
	esz := x.DType().Size()
	src := x.Bytes()
	rowIn := xs[axis] * inner * esz
	outs := make([]*tensor.Tensor, len(split))
	off := 0
	for i, sz := range split {
		sh := xs.Clone()
		sh[axis] = int(sz)
		t := ctx.New(x.DType(), sh...)
		dst := t.Bytes()
		chunk := int(sz) * inner * esz
		for r := 0; r < outer; r++ {
			copy(dst[r*chunk:(r+1)*chunk], src[r*rowIn+off:r*rowIn+off+chunk])
		}
		off += chunk
		outs[i] = t
	}
	return outs, nil
}

// ---- Tile ----

type tileOp struct{ n NodeInfo }

func (o *tileOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	reps := asI64(in[1])
	xs := x.Shape()
	if len(reps) != len(xs) {
		return nil, o.n.Errorf("repeats rank mismatch")
	}
	os := xs.Clone()
	for i, r := range reps {
		os[i] *= int(r)
	}
	out := ctx.New(x.DType(), os...)
	esz := x.DType().Size()
	src, dst := x.Bytes(), out.Bytes()
	r := len(os)
	xst := xs.Strides()
	idx := make([]int, r)
	n := out.Numel()
	for i := 0; i < n; i++ {
		off := 0
		for d := 0; d < r; d++ {
			off += (idx[d] % xs[d]) * xst[d]
		}
		copy(dst[i*esz:(i+1)*esz], src[off*esz:(off+1)*esz])
		for d := r - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < os[d] {
				break
			}
			idx[d] = 0
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	Register("", "Reshape", 5, func(n NodeInfo) (Op, error) {
		return &reshapeOp{n: n, allowZero: n.Attrs.Int("allowzero", 0) == 1}, nil
	})
	Register("", "Reshape", 1, func(n NodeInfo) (Op, error) {
		return &reshapeOp{n: n, attrShape: n.Attrs.Ints("shape", nil)}, nil
	})
	Register("", "Flatten", 1, func(n NodeInfo) (Op, error) {
		return &flattenOp{n: n, axis: int(n.Attrs.Int("axis", 1))}, nil
	})
	Register("", "Squeeze", 13, func(n NodeInfo) (Op, error) { return &squeezeOp{n: n}, nil })
	Register("", "Squeeze", 1, func(n NodeInfo) (Op, error) {
		return &squeezeOp{n: n, attrAxes: n.Attrs.Ints("axes", nil)}, nil
	})
	Register("", "Unsqueeze", 13, func(n NodeInfo) (Op, error) { return &squeezeOp{n: n, unsq: true}, nil })
	Register("", "Unsqueeze", 1, func(n NodeInfo) (Op, error) {
		return &squeezeOp{n: n, unsq: true, attrAxes: n.Attrs.Ints("axes", nil)}, nil
	})
	Register("", "Transpose", 1, func(n NodeInfo) (Op, error) {
		return &transposeOp{n: n, perm: n.Attrs.Ints("perm", nil)}, nil
	})
	Register("", "Concat", 4, func(n NodeInfo) (Op, error) {
		if !n.Attrs.Has("axis") {
			return nil, n.Errorf("axis required")
		}
		return &concatOp{n: n, axis: int(n.Attrs.Int("axis", 0))}, nil
	})
	Register("", "Gather", 1, func(n NodeInfo) (Op, error) {
		return &gatherOp{n: n, axis: int(n.Attrs.Int("axis", 0))}, nil
	})
	Register("", "Shape", 1, func(n NodeInfo) (Op, error) {
		return &shapeOp{n: n, start: n.Attrs.Int("start", 0), end: n.Attrs.Int("end", 0), hasEnd: n.Attrs.Has("end")}, nil
	})
	Register("", "Constant", 1, buildConstant)
	Register("", "Slice", 10, func(n NodeInfo) (Op, error) { return &sliceOp{n: n}, nil })
	Register("", "Slice", 1, func(n NodeInfo) (Op, error) {
		return &sliceOp{n: n, attrStarts: n.Attrs.Ints("starts", nil), attrEnds: n.Attrs.Ints("ends", nil), attrAxes: n.Attrs.Ints("axes", nil)}, nil
	})
	Register("", "Cast", 6, func(n NodeInfo) (Op, error) {
		to, ok := onnxDType(n.Attrs.Int("to", 0))
		if !ok {
			return nil, n.Errorf("unsupported target type %d", n.Attrs.Int("to", 0))
		}
		return &castOp{n: n, to: to}, nil
	})
	Register("", "Expand", 8, func(n NodeInfo) (Op, error) { return &expandOp{n}, nil })
	Register("", "ConstantOfShape", 9, func(n NodeInfo) (Op, error) {
		v := n.Attrs.Tensor("value")
		if v == nil {
			v = tensor.Scalar(0)
		}
		return &constantOfShapeOp{n: n, val: v}, nil
	})
	Register("", "Where", 9, func(n NodeInfo) (Op, error) { return &whereOp{n}, nil })
	cmp := func(name string, since int, fn func(x, y float64) bool) {
		Register("", name, since, func(n NodeInfo) (Op, error) { return &compareOp{n, fn}, nil })
	}
	cmp("Equal", 7, func(x, y float64) bool { return x == y })
	cmp("Greater", 7, func(x, y float64) bool { return x > y })
	cmp("Less", 7, func(x, y float64) bool { return x < y })
	cmp("GreaterOrEqual", 12, func(x, y float64) bool { return x >= y })
	cmp("LessOrEqual", 12, func(x, y float64) bool { return x <= y })
	cmp("And", 7, func(x, y float64) bool { return x != 0 && y != 0 })
	cmp("Or", 7, func(x, y float64) bool { return x != 0 || y != 0 })
	cmp("Xor", 7, func(x, y float64) bool { return (x != 0) != (y != 0) })
	Register("", "Not", 1, func(n NodeInfo) (Op, error) { return &notOp{n}, nil })
	Register("", "Range", 11, func(n NodeInfo) (Op, error) { return &rangeOp{n}, nil })
	Register("", "Split", 13, func(n NodeInfo) (Op, error) {
		return &splitOp{n: n, axis: int(n.Attrs.Int("axis", 0)), numOut: int(n.Attrs.Int("num_outputs", 0))}, nil
	})
	Register("", "Split", 2, func(n NodeInfo) (Op, error) {
		return &splitOp{n: n, axis: int(n.Attrs.Int("axis", 0)), attrSplit: n.Attrs.Ints("split", nil)}, nil
	})
	Register("", "Tile", 6, func(n NodeInfo) (Op, error) { return &tileOp{n}, nil })
}
