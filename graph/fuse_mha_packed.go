package graph

import (
	"github.com/giraffesyo/ingot/ops"
	"github.com/giraffesyo/ingot/tensor"
)

// fuseMHAPacked rewrites the dynamo exporter's attention block around an
// already-fused ingot.SDPA node into one ingot.MHA (layout 1) that reads the
// qkv Linear's output in place:
//
//	view [B,T,3,H,dh] → Transpose(2,0,3,1,4) → Slice(axis0, i)+Squeeze  (i = 0,1,2)
//	  q ──────────────────────────────────────────────┐
//	  k → Reshape([-1,T,dh]) → Transpose(0,2,1) → Reshape([B,H,dh,T]) ┤ SDPA(stride_out) → out [B,T,H,dh]
//	  v ──────────────────────────────────────────────┘
//
// The 5-D permute materialises 3·B·T·H·dh floats and the key chain another
// B·H·T·dh, per layer, purely to hand the GEMMs contiguous operands; the MHA
// op reads all three strided from the view (row stride 3·H·dh) and writes
// [B,T,H,dh] directly. The key chain is verified symbolically (symShape)
// because the exporter builds the Reshape targets from Shape/Slice/Concat
// plumbing at runtime; that plumbing dies with the chain.
func fuseMHAPacked(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	changed := false
	for _, host := range g.Nodes {
		if dead[host] || host.OpType != "SDPA" || host.Domain != ingotDomain || len(host.Inputs) != 3 {
			continue
		}
		if host.Attrs.Int("stride_out", 0) != 1 || host.Attrs.Int("a_layout", 0) != 0 ||
			host.Attrs.Int("b_layout", 0) != 0 || host.Attrs.Int("v_layout", 0) != 0 {
			continue
		}
		qsq, qsl, x5q, s0, ok0 := unpackDyn(g, host.Inputs[0])
		vsq, vsl, x5v, s2, ok2 := unpackDyn(g, host.Inputs[2])
		if !ok0 || !ok2 || s0 != 0 || s2 != 2 || x5q != x5v {
			continue
		}
		if g.soleConsumer(qsq.Outputs[0]) != host || g.soleConsumer(vsq.Outputs[0]) != host {
			continue
		}
		// Key: Reshape → Transpose(0,2,1) → Reshape, then the unpack.
		r2 := host.Inputs[1].Producer
		if r2 == nil || dead[r2] || r2.OpType != "Reshape" || g.soleConsumer(r2.Outputs[0]) != host || len(r2.Inputs) < 2 {
			continue
		}
		tr := r2.Inputs[0].Producer
		if tr == nil || dead[tr] || tr.OpType != "Transpose" || !permIs(tr, 0, 2, 1) || g.soleConsumer(tr.Outputs[0]) != r2 {
			continue
		}
		r1 := tr.Inputs[0].Producer
		if r1 == nil || dead[r1] || r1.OpType != "Reshape" || g.soleConsumer(r1.Outputs[0]) != tr || len(r1.Inputs) < 2 {
			continue
		}
		k4 := r1.Inputs[0]
		if !k4.HasShape || len(k4.Shape) != 4 {
			continue
		}
		sh1, ok := symShape(r1.Inputs[1], 0)
		if !ok || len(sh1) != 3 || sh1[0].src != nil || sh1[0].lit != -1 || !sh1[1].is(k4, 2) || !sh1[2].is(k4, 3) {
			continue
		}
		sh2, ok := symShape(r2.Inputs[1], 0)
		if !ok || len(sh2) != 4 || !sh2[0].is(k4, 0) || !sh2[1].is(k4, 1) || !sh2[2].is(k4, 3) || !sh2[3].is(k4, 2) {
			continue
		}
		ksq, ksl, x5k, s1, ok1 := unpackDyn(g, k4)
		if !ok1 || s1 != 1 || x5k != x5q {
			continue
		}
		// k4 feeds r1 and, at most, shape plumbing that sinks into r1/r2.
		plumb := map[*Node]bool{}
		if !shapeSinks(g, k4, map[*Node]bool{r1: true, r2: true}, plumb) {
			continue
		}
		// The permute: Transpose(2,0,3,1,4) of a rank-5 view with 3 at dim 2,
		// consumed only by the three slices.
		perm := x5q.Producer
		if perm == nil || dead[perm] || perm.OpType != "Transpose" || !permIs(perm, 2, 0, 3, 1, 4) || len(perm.Inputs) != 1 {
			continue
		}
		view := perm.Inputs[0]
		if view == nil || !view.HasShape || len(view.Shape) != 5 || (view.Shape[2] != 3 && view.Shape[2] != -1) {
			continue
		}
		if len(x5q.Consumers) != 3 || g.isOutput(x5q) {
			continue
		}
		seen := map[*Node]bool{qsl: true, ksl: true, vsl: true}
		if !seen[x5q.Consumers[0]] || !seen[x5q.Consumers[1]] || !seen[x5q.Consumers[2]] {
			continue
		}
		drop := []*Node{perm, qsl, ksl, vsl, qsq, ksq, vsq, r1, tr, r2}
		for n := range plumb {
			drop = append(drop, n)
		}
		// New input attached before the drops (see fuse-sdpa for why).
		view.Consumers = append(view.Consumers, host)
		oldIns := host.Inputs
		for _, n := range drop {
			dead[n] = true
			g.dropNode(n)
		}
		for _, v := range oldIns {
			if v != nil {
				removeConsumer(v, host)
			}
		}
		host.OpType = "MHA"
		host.Attrs = ops.Attrs{
			"scale":  host.Attrs["scale"],
			"layout": {Kind: ops.KindInt, I: 1},
		}
		host.Inputs = []*Value{view}
		stats["fuse-mha-packed"]++
		changed = true
	}
	g.compact(dead)
	return changed
}

// unpackDyn matches v = Squeeze(axes=[0])(Slice(x, start, start+1, axis 0))
// with the squeeze axes given either as an attribute (opset ≤ 12) or as a
// constant input.
func unpackDyn(g *Graph, v *Value) (sq, sl *Node, x *Value, start int64, ok bool) {
	sq = v.Producer
	if sq == nil || sq.OpType != "Squeeze" || sq.Domain != "" {
		return nil, nil, nil, 0, false
	}
	ax := sq.Attrs.Ints("axes", nil)
	if len(sq.Inputs) > 1 && sq.Inputs[1] != nil {
		c := sq.Inputs[1].Const
		if c == nil || c.DType() != tensor.I64 {
			return nil, nil, nil, 0, false
		}
		ax = c.I64()
	}
	if len(ax) != 1 || ax[0] != 0 {
		return nil, nil, nil, 0, false
	}
	sl = sq.Inputs[0].Producer
	if sl == nil || g.soleConsumer(sl.Outputs[0]) != sq {
		return nil, nil, nil, 0, false
	}
	st, ok := sliceStart(sl)
	if !ok {
		return nil, nil, nil, 0, false
	}
	return sq, sl, sl.Inputs[0], st, true
}

// shapeSinks checks that every consumer of v other than the sinks is a
// shape-plumbing node (Shape/Slice/Concat/Gather/Squeeze/Unsqueeze/Cast)
// whose outputs, transitively, feed only sinks or more plumbing. The
// plumbing nodes are collected into plumb; they hold no data once the sinks
// go, and a Shape node over a value that is about to disappear must go too.
func shapeSinks(g *Graph, v *Value, sinks map[*Node]bool, plumb map[*Node]bool) bool {
	if g.isOutput(v) {
		return false
	}
	for _, c := range v.Consumers {
		if sinks[c] || plumb[c] {
			continue
		}
		switch c.OpType {
		case "Shape", "Slice", "Concat", "Gather", "Squeeze", "Unsqueeze", "Cast":
		default:
			return false
		}
		if c.Domain != "" {
			return false
		}
		plumb[c] = true
		for _, o := range c.Outputs {
			if o != nil && !shapeSinks(g, o, sinks, plumb) {
				return false
			}
		}
	}
	return true
}

// symDim is one element of a symbolically evaluated shape tensor: either a
// literal, or "dimension axis of value src".
type symDim struct {
	lit  int64
	src  *Value
	axis int
}

// is reports whether d denotes dimension axis of v — symbolically, or as a
// literal equal to v's statically known extent.
func (d symDim) is(v *Value, axis int) bool {
	if d.src != nil {
		return d.src == v && d.axis == axis
	}
	return v.HasShape && axis < len(v.Shape) && v.Shape[axis] > 0 && d.lit == int64(v.Shape[axis])
}

// symShape evaluates a 1-D int64 shape tensor symbolically through the
// exporter's plumbing: constants, Shape(x) (with start/end), Slice with
// constant bounds on axis 0, Concat on axis 0, Gather of a scalar index and
// Unsqueeze of that scalar. Anything else fails.
func symShape(v *Value, depth int) ([]symDim, bool) {
	if v == nil || depth > 12 {
		return nil, false
	}
	if c := v.Const; c != nil {
		if c.DType() != tensor.I64 || c.Shape().Rank() > 1 {
			return nil, false
		}
		out := make([]symDim, c.Numel())
		for i, x := range c.I64() {
			out[i] = symDim{lit: x}
		}
		return out, true
	}
	p := v.Producer
	if p == nil || p.Domain != "" {
		return nil, false
	}
	switch p.OpType {
	case "Shape":
		x := p.Inputs[0]
		if x == nil || !x.HasShape {
			return nil, false
		}
		r := len(x.Shape)
		st := int(p.Attrs.Int("start", 0))
		en := int(p.Attrs.Int("end", int64(r)))
		if st < 0 {
			st += r
		}
		if en < 0 {
			en += r
		}
		st, en = max(0, min(st, r)), max(0, min(en, r))
		var out []symDim
		for i := st; i < en; i++ {
			out = append(out, symDim{src: x, axis: i})
		}
		return out, true
	case "Slice":
		base, ok := symShape(p.Inputs[0], depth+1)
		if !ok || len(p.Inputs) < 3 {
			return nil, false
		}
		scalar := func(i int, def int64) (int64, bool) {
			if i >= len(p.Inputs) || p.Inputs[i] == nil {
				return def, true
			}
			c := p.Inputs[i].Const
			if c == nil || c.DType() != tensor.I64 || c.Numel() != 1 {
				return 0, false
			}
			return c.I64()[0], true
		}
		st, ok1 := scalar(1, 0)
		en, ok2 := scalar(2, int64(len(base)))
		ax, ok3 := scalar(3, 0)
		sp, ok4 := scalar(4, 1)
		if !ok1 || !ok2 || !ok3 || !ok4 || ax != 0 || sp != 1 {
			return nil, false
		}
		n := int64(len(base))
		if st < 0 {
			st += n
		}
		if en < 0 {
			en += n
		}
		st, en = max(0, min(st, n)), max(0, min(en, n))
		if st > en {
			st = en
		}
		return base[st:en], true
	case "Concat":
		if p.Attrs.Int("axis", 0) != 0 {
			return nil, false
		}
		var out []symDim
		for _, in := range p.Inputs {
			part, ok := symShape(in, depth+1)
			if !ok {
				return nil, false
			}
			out = append(out, part...)
		}
		return out, true
	case "Gather":
		base, ok := symShape(p.Inputs[0], depth+1)
		if !ok || len(p.Inputs) < 2 || p.Inputs[1] == nil || p.Attrs.Int("axis", 0) != 0 {
			return nil, false
		}
		c := p.Inputs[1].Const
		if c == nil || c.DType() != tensor.I64 || c.Numel() != 1 {
			return nil, false
		}
		i := c.I64()[0]
		if i < 0 {
			i += int64(len(base))
		}
		if i < 0 || i >= int64(len(base)) {
			return nil, false
		}
		return base[i : i+1], true
	case "Unsqueeze", "Squeeze":
		return symShape(p.Inputs[0], depth+1) // scalar ↔ [1]: same single element
	case "Cast":
		if p.Attrs.Int("to", 0) != int64(onnxInt64) {
			return nil, false
		}
		return symShape(p.Inputs[0], depth+1)
	}
	return nil, false
}

// onnxInt64 is TensorProto.INT64.
const onnxInt64 = 7
