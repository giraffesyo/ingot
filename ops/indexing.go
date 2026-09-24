package ops

import (
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/tensor"
)

// Index-driven gathers and scatters (GatherND, ScatterND, GatherElements,
// ScatterElements), OneHot and Mod. Data movement is dtype-generic (byte
// copies); scatter reductions compute on f32, i64 and i32.

// wrapIndex resolves a possibly negative index against dim.
func wrapIndex(v int64, dim int) (int, error) {
	if v < 0 {
		v += int64(dim)
	}
	if v < 0 || v >= int64(dim) {
		return 0, fmt.Errorf("index %d out of range for dim %d", v, dim)
	}
	return int(v), nil
}

func numel(s []int) int {
	n := 1
	for _, d := range s {
		n *= d
	}
	return n
}

// ---- GatherND ----

type gatherNDOp struct {
	n         NodeInfo
	batchDims int
}

func (o *gatherNDOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("GatherND: need data and indices")
	}
	data, idxT := in[0], in[1]
	ds, is := data.Shape(), idxT.Shape()
	b := o.batchDims
	if len(is) < 1 || b < 0 || b >= len(is) || b > len(ds) {
		return nil, o.n.Errorf("GatherND: data %v, indices %v, batch_dims %d", ds, is, b)
	}
	L := is[len(is)-1]
	if b+L > len(ds) {
		return nil, o.n.Errorf("GatherND: index depth %d + batch_dims %d > data rank %d", L, b, len(ds))
	}
	for d := 0; d < b; d++ {
		if ds[d] != is[d] {
			return nil, o.n.Errorf("GatherND: batch dim %d: data %d vs indices %d", d, ds[d], is[d])
		}
	}
	oshape := append(append([]int{}, is[:len(is)-1]...), ds[b+L:]...)
	out := ctx.NewUninit(data.DType(), oshape...)
	idx := asI64(idxT)
	es := data.DType().Size()
	slice := numel(ds[b+L:]) * es
	batches := numel(ds[:b])
	perBatchData := numel(ds[b:]) * es
	tuples := numel(is[b : len(is)-1]) // index tuples per batch
	src, dst := data.Bytes(), out.Bytes()
	// Strides (in elements) of the indexed dims b..b+L-1.
	st := make([]int, L)
	acc := numel(ds[b+L:])
	for d := L - 1; d >= 0; d-- {
		st[d] = acc
		acc *= ds[b+d]
	}
	for bb := 0; bb < batches; bb++ {
		for t := 0; t < tuples; t++ {
			tup := idx[(bb*tuples+t)*L : (bb*tuples+t+1)*L]
			off := 0
			for d, v := range tup {
				i, err := wrapIndex(v, ds[b+d])
				if err != nil {
					return nil, o.n.Errorf("GatherND: %v", err)
				}
				off += i * st[d]
			}
			o := bb*perBatchData + off*es
			copy(dst[(bb*tuples+t)*slice:], src[o:o+slice])
		}
	}
	return ctx.Out(out), nil
}

// ---- scatter reductions ----

type scatterRed int

const (
	redNone scatterRed = iota
	redAdd
	redMul
	redMax
	redMin
)

func parseReduction(n NodeInfo) (scatterRed, error) {
	switch r := n.Attrs.String("reduction", "none"); r {
	case "none":
		return redNone, nil
	case "add":
		return redAdd, nil
	case "mul":
		return redMul, nil
	case "max":
		return redMax, nil
	case "min":
		return redMin, nil
	default:
		return 0, n.Errorf("unsupported reduction %q", r)
	}
}

// combine applies red to out[i] with u[j] for the dtype.
func combine(dt tensor.DType, red scatterRed, out, upd *tensor.Tensor, i, j, n int) error {
	switch dt {
	case tensor.F32:
		o, u := out.F32()[i:i+n], upd.F32()[j:j+n]
		for k := range o {
			o[k] = redF32(red, o[k], u[k])
		}
	case tensor.I64:
		o, u := out.I64()[i:i+n], upd.I64()[j:j+n]
		for k := range o {
			o[k] = redInt(red, o[k], u[k])
		}
	case tensor.I32:
		o, u := out.I32()[i:i+n], upd.I32()[j:j+n]
		for k := range o {
			o[k] = int32(redInt(red, int64(o[k]), int64(u[k])))
		}
	default:
		return fmt.Errorf("reduction on %s not supported", dt)
	}
	return nil
}

func redF32(r scatterRed, a, b float32) float32 {
	switch r {
	case redAdd:
		return a + b
	case redMul:
		return a * b
	case redMax:
		return max(a, b)
	case redMin:
		return min(a, b)
	}
	return b
}

func redInt(r scatterRed, a, b int64) int64 {
	switch r {
	case redAdd:
		return a + b
	case redMul:
		return a * b
	case redMax:
		return max(a, b)
	case redMin:
		return min(a, b)
	}
	return b
}

// ---- ScatterND ----

type scatterNDOp struct {
	n   NodeInfo
	red scatterRed
}

func (o *scatterNDOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 3 || in[0] == nil || in[1] == nil || in[2] == nil {
		return nil, o.n.Errorf("ScatterND: need data, indices, updates")
	}
	data, idxT, upd := in[0], in[1], in[2]
	ds, is := data.Shape(), idxT.Shape()
	if len(is) < 1 || upd.DType() != data.DType() {
		return nil, o.n.Errorf("ScatterND: data %v %s, indices %v, updates %s", ds, data.DType(), is, upd.DType())
	}
	L := is[len(is)-1]
	if L > len(ds) {
		return nil, o.n.Errorf("ScatterND: index depth %d > data rank %d", L, len(ds))
	}
	want := append(append([]int{}, is[:len(is)-1]...), ds[L:]...)
	if !upd.Shape().Equal(want) {
		return nil, o.n.Errorf("ScatterND: updates %v, want %v", upd.Shape(), want)
	}
	out := ctx.NewUninit(data.DType(), ds...)
	copy(out.Bytes(), data.Bytes())
	idx := asI64(idxT)
	es := data.DType().Size()
	n := numel(ds[L:])
	st := make([]int, L)
	acc := n
	for d := L - 1; d >= 0; d-- {
		st[d] = acc
		acc *= ds[d]
	}
	tuples := numel(is[:len(is)-1])
	ob, ub := out.Bytes(), upd.Bytes()
	for t := 0; t < tuples; t++ {
		off := 0
		for d, v := range idx[t*L : (t+1)*L] {
			i, err := wrapIndex(v, ds[d])
			if err != nil {
				return nil, o.n.Errorf("ScatterND: %v", err)
			}
			off += i * st[d]
		}
		if o.red == redNone {
			copy(ob[off*es:(off+n)*es], ub[t*n*es:(t+1)*n*es])
			continue
		}
		if err := combine(data.DType(), o.red, out, upd, off, t*n, n); err != nil {
			return nil, o.n.Errorf("ScatterND: %v", err)
		}
	}
	return ctx.Out(out), nil
}

// ---- GatherElements / ScatterElements ----

// elemAxis resolves axis and returns (outer, dim, inner) of shape s around it.
func elemAxis(axis int, s []int) (int, int, int, error) {
	a := axis
	if a < 0 {
		a += len(s)
	}
	if a < 0 || a >= len(s) {
		return 0, 0, 0, fmt.Errorf("axis %d out of range for rank %d", axis, len(s))
	}
	return a, numel(s[:a]), numel(s[a+1:]), nil
}

type gatherElementsOp struct {
	n    NodeInfo
	axis int
}

func (o *gatherElementsOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("GatherElements: need data and indices")
	}
	data, idxT := in[0], in[1]
	ds, is := data.Shape(), idxT.Shape()
	if len(ds) != len(is) {
		return nil, o.n.Errorf("GatherElements: rank mismatch %v vs %v", ds, is)
	}
	a, _, _, err := elemAxis(o.axis, ds)
	if err != nil {
		return nil, o.n.Errorf("GatherElements: %v", err)
	}
	out := ctx.NewUninit(data.DType(), is...)
	idx := asI64(idxT)
	es := data.DType().Size()
	src, dst := data.Bytes(), out.Bytes()
	dstr := tensor.Shape(ds).Strides()
	coord := make([]int, len(is))
	for k := range idx {
		off := 0
		for d, c := range coord {
			if d == a {
				i, err := wrapIndex(idx[k], ds[a])
				if err != nil {
					return nil, o.n.Errorf("GatherElements: %v", err)
				}
				c = i
			}
			off += c * dstr[d]
		}
		copy(dst[k*es:(k+1)*es], src[off*es:(off+1)*es])
		incCoord(coord, is)
	}
	return ctx.Out(out), nil
}

// incCoord advances a row-major coordinate over shape s.
func incCoord(c []int, s []int) {
	for d := len(c) - 1; d >= 0; d-- {
		c[d]++
		if c[d] < s[d] {
			return
		}
		c[d] = 0
	}
}

type scatterElementsOp struct {
	n    NodeInfo
	axis int
	red  scatterRed
}

func (o *scatterElementsOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 3 || in[0] == nil || in[1] == nil || in[2] == nil {
		return nil, o.n.Errorf("ScatterElements: need data, indices, updates")
	}
	data, idxT, upd := in[0], in[1], in[2]
	ds, is := data.Shape(), idxT.Shape()
	if len(ds) != len(is) || !upd.Shape().Equal(is) || upd.DType() != data.DType() {
		return nil, o.n.Errorf("ScatterElements: data %v, indices %v, updates %v", ds, is, upd.Shape())
	}
	a, _, _, err := elemAxis(o.axis, ds)
	if err != nil {
		return nil, o.n.Errorf("ScatterElements: %v", err)
	}
	out := ctx.NewUninit(data.DType(), ds...)
	copy(out.Bytes(), data.Bytes())
	idx := asI64(idxT)
	es := data.DType().Size()
	ob, ub := out.Bytes(), upd.Bytes()
	dstr := tensor.Shape(ds).Strides()
	coord := make([]int, len(is))
	for k := range idx {
		off := 0
		for d, c := range coord {
			if d == a {
				i, err := wrapIndex(idx[k], ds[a])
				if err != nil {
					return nil, o.n.Errorf("ScatterElements: %v", err)
				}
				c = i
			}
			off += c * dstr[d]
		}
		if o.red == redNone {
			copy(ob[off*es:(off+1)*es], ub[k*es:(k+1)*es])
		} else if err := combine(data.DType(), o.red, out, upd, off, k, 1); err != nil {
			return nil, o.n.Errorf("ScatterElements: %v", err)
		}
		incCoord(coord, is)
	}
	return ctx.Out(out), nil
}

// ---- OneHot ----

type oneHotOp struct {
	n    NodeInfo
	axis int
}

func (o *oneHotOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 3 || in[0] == nil || in[1] == nil || in[2] == nil {
		return nil, o.n.Errorf("OneHot: need indices, depth, values")
	}
	idxT, depthT, vals := in[0], in[1], in[2]
	if depthT.Numel() != 1 || vals.Numel() != 2 {
		return nil, o.n.Errorf("OneHot: depth %v, values %v", depthT.Shape(), vals.Shape())
	}
	var depth int
	switch depthT.DType() {
	case tensor.F32:
		depth = int(depthT.F32()[0])
	default:
		depth = int(asI64(depthT)[0])
	}
	if depth <= 0 {
		return nil, o.n.Errorf("OneHot: depth %d", depth)
	}
	is := idxT.Shape()
	r := len(is) + 1
	a := o.axis
	if a < 0 {
		a += r
	}
	if a < 0 || a >= r {
		return nil, o.n.Errorf("OneHot: axis %d out of range for output rank %d", o.axis, r)
	}
	oshape := append(append(append([]int{}, is[:a]...), depth), is[a:]...)
	out := ctx.NewUninit(vals.DType(), oshape...)
	es := vals.DType().Size()
	vb, ob := vals.Bytes(), out.Bytes()
	off, on := vb[:es], vb[es:2*es]
	for i := 0; i < out.Numel(); i++ {
		copy(ob[i*es:(i+1)*es], off)
	}
	var idx []float64
	switch idxT.DType() {
	case tensor.F32:
		for _, v := range idxT.F32() {
			idx = append(idx, float64(v))
		}
	default:
		for _, v := range asI64(idxT) {
			idx = append(idx, float64(v))
		}
	}
	outer, inner := numel(is[:a]), numel(is[a:])
	for p := 0; p < outer; p++ {
		for q := 0; q < inner; q++ {
			v := int64(math.Floor(idx[p*inner+q]))
			if v < 0 {
				v += int64(depth)
			}
			if v < 0 || v >= int64(depth) {
				continue // out of range: all off (per spec)
			}
			e := (p*depth+int(v))*inner + q
			copy(ob[e*es:(e+1)*es], on)
		}
	}
	return ctx.Out(out), nil
}

// ---- Mod ----

type modOp struct {
	n    NodeInfo
	fmod bool
}

func (o *modOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) != 2 || in[0] == nil || in[1] == nil || in[0].DType() != in[1].DType() {
		return nil, o.n.Errorf("Mod: need two inputs of one dtype")
	}
	a, b := in[0], in[1]
	os, err := broadcastShape(a.Shape(), b.Shape())
	if err != nil {
		return nil, o.n.Errorf("Mod: %v", err)
	}
	ast, bst := broadcastStrides(a.Shape(), os), broadcastStrides(b.Shape(), os)
	out := ctx.NewUninit(a.DType(), os...)
	coord := make([]int, len(os))
	n := out.Numel()
	at := func() (int, int) {
		oa, ob := 0, 0
		for d, c := range coord {
			oa += c * ast[d]
			ob += c * bst[d]
		}
		return oa, ob
	}
	switch a.DType() {
	case tensor.F32:
		af, bf, of := a.F32(), b.F32(), out.F32()
		for k := 0; k < n; k++ {
			i, j := at()
			of[k] = float32(modF(float64(af[i]), float64(bf[j]), o.fmod))
			incCoord(coord, os)
		}
	case tensor.I64, tensor.I32:
		ai, bi := asI64(a), asI64(b)
		res := make([]int64, n)
		for k := 0; k < n; k++ {
			i, j := at()
			if bi[j] == 0 {
				return nil, o.n.Errorf("Mod: integer division by zero")
			}
			res[k] = modI(ai[i], bi[j], o.fmod)
			incCoord(coord, os)
		}
		if a.DType() == tensor.I64 {
			copy(out.I64(), res)
		} else {
			for k, v := range res {
				out.I32()[k] = int32(v)
			}
		}
	default:
		return nil, o.n.Errorf("Mod: unsupported dtype %s", a.DType())
	}
	return ctx.Out(out), nil
}

// modI: fmod keeps the dividend's sign (C), else the divisor's (Python).
func modI(x, y int64, fmod bool) int64 {
	r := x % y
	if !fmod && r != 0 && (r < 0) != (y < 0) {
		r += y
	}
	return r
}

func modF(x, y float64, fmod bool) float64 {
	r := math.Mod(x, y)
	if !fmod && r != 0 && (r < 0) != (y < 0) {
		r += y
	}
	return r
}

func init() {
	Register("", "GatherND", 11, func(n NodeInfo) (Op, error) {
		return &gatherNDOp{n: n, batchDims: int(n.Attrs.Int("batch_dims", 0))}, nil
	})
	Register("", "ScatterND", 11, func(n NodeInfo) (Op, error) {
		r, err := parseReduction(n)
		return &scatterNDOp{n: n, red: r}, err
	})
	Register("", "GatherElements", 11, func(n NodeInfo) (Op, error) {
		return &gatherElementsOp{n: n, axis: int(n.Attrs.Int("axis", 0))}, nil
	})
	Register("", "ScatterElements", 11, func(n NodeInfo) (Op, error) {
		r, err := parseReduction(n)
		return &scatterElementsOp{n: n, axis: int(n.Attrs.Int("axis", 0)), red: r}, err
	})
	Register("", "OneHot", 9, func(n NodeInfo) (Op, error) { return &oneHotOp{n: n, axis: int(n.Attrs.Int("axis", -1))}, nil })
	Register("", "Mod", 10, func(n NodeInfo) (Op, error) { return &modOp{n: n, fmod: n.Attrs.Int("fmod", 0) != 0}, nil })
}
