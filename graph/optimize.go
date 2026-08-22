package graph

import (
	"math"

	"github.com/giraffesyo/ingot/ops"
	"github.com/giraffesyo/ingot/tensor"
)

// Optimize applies semantics-preserving rewrites to g in place and returns a
// count of applications per pass. Compile runs it; CompileRaw does not.
//
// Passes (iterated to a fixed point):
//
//   - fuse-hardswish:   Add(x,3) → Clip(0,6) → Mul(x,·) → Div(·,6)  ⇒  ingot.HardSwish(x)
//     (the opset<14 decomposition emitted by Paddle2ONNX and torch).
//   - fuse-silu:        Mul(x, Sigmoid(x))  ⇒  ingot.SiLU(x)  (torch SiLU/Swish export).
//   - fuse-gelu:        x·0.5·(1+Erf(x/√2)) in any association  ⇒  ingot.Gelu(x).
//   - fold-conv-affine: Conv/ConvTranspose → {Mul,Add,Sub by scalar or per-channel
//     const | BatchNormalization}  ⇒  folded into W and bias (exact in f64).
//   - fuse-conv-act:    Conv/ConvTranspose → {Relu,HardSwish,HardSigmoid,Sigmoid,
//     Clip,LeakyRelu}  ⇒  activation runs in the conv epilogue (ingot_act attrs).
//   - fold-post-affine: Conv(act) → {Mul,Add,Sub,Div by scalar const}  ⇒  post
//     scale/shift in the conv epilogue.
//
// Every rewrite keeps the consumer's output Value (so graph outputs and names
// survive) and only fires when intermediate values have a single consumer.
func Optimize(g *Graph) map[string]int {
	stats := map[string]int{}
	for changed := true; changed; {
		changed = false
		changed = fuseHardSwish(g, stats) || changed
		changed = fuseSiLU(g, stats) || changed
		changed = fuseGelu(g, stats) || changed
		changed = foldConvAffine(g, stats) || changed
		changed = fuseConvAct(g, stats) || changed
		changed = foldPostAffine(g, stats) || changed
	}
	renumber(g)
	return stats
}

// ingotDomain is the operator domain for runtime-internal fused ops.
const ingotDomain = "ingot"

// ---- helpers ----

func scalarConst(v *Value) (float32, bool) {
	if v == nil || v.Const == nil || v.Const.DType() != tensor.F32 || v.Const.Numel() != 1 {
		return 0, false
	}
	return v.Const.F32()[0], true
}

// channelConst returns a per-channel vector of length M if v is an f32
// constant that broadcasts over NCHW channels: a scalar, or shape [..1, M, 1, 1].
func channelConst(v *Value, M int) ([]float32, bool) {
	if v == nil || v.Const == nil || v.Const.DType() != tensor.F32 {
		return nil, false
	}
	f := v.Const.F32()
	if len(f) == 1 {
		out := make([]float32, M)
		for i := range out {
			out[i] = f[0]
		}
		return out, true
	}
	sh := v.Const.Shape()
	if len(f) != M || len(sh) < 3 || sh[len(sh)-1] != 1 || sh[len(sh)-2] != 1 || sh[len(sh)-3] != M {
		return nil, false
	}
	for _, d := range sh[:len(sh)-3] {
		if d != 1 {
			return nil, false
		}
	}
	return append([]float32(nil), f...), true
}

func (g *Graph) isOutput(v *Value) bool {
	for _, o := range g.Outputs {
		if o == v {
			return true
		}
	}
	return false
}

// soleConsumer returns the only consumer of v, or nil if v is a graph output
// or has any other number of consumers.
func (g *Graph) soleConsumer(v *Value) *Node {
	if v == nil || len(v.Consumers) != 1 || g.isOutput(v) {
		return nil
	}
	return v.Consumers[0]
}

func removeConsumer(v *Value, n *Node) {
	for i, c := range v.Consumers {
		if c == n {
			v.Consumers = append(v.Consumers[:i], v.Consumers[i+1:]...)
			return
		}
	}
}

// dropNode detaches n from its inputs and deletes its outputs from the value
// table. Constants left without consumers are deleted too.
func (g *Graph) dropNode(n *Node) {
	for _, v := range n.Inputs {
		if v == nil {
			continue
		}
		removeConsumer(v, n)
		if v.Const != nil && len(v.Consumers) == 0 && !g.isOutput(v) {
			delete(g.Values, v.Name)
		}
	}
	for _, v := range n.Outputs {
		if v != nil && v.Producer == n {
			v.Producer = nil
			delete(g.Values, v.Name)
		}
	}
}

// absorb makes node n (producer of mid) take over consumer u's output: n now
// produces u.Outputs[0]; mid and u disappear.
func (g *Graph) absorb(n *Node, mid *Value, u *Node, dead map[*Node]bool) {
	out := u.Outputs[0]
	removeConsumer(mid, u)
	g.dropNode(u) // removes u's other inputs' consumer links, deletes out from Values — re-add below
	g.Values[out.Name] = out
	out.Producer = n
	for i, v := range n.Outputs {
		if v == mid {
			n.Outputs[i] = out
		}
	}
	delete(g.Values, mid.Name)
	dead[u] = true
}

func (g *Graph) compact(dead map[*Node]bool) {
	if len(dead) == 0 {
		return
	}
	keep := g.Nodes[:0]
	for _, n := range g.Nodes {
		if !dead[n] {
			keep = append(keep, n)
		}
	}
	for i := len(keep); i < len(g.Nodes); i++ {
		g.Nodes[i] = nil
	}
	g.Nodes = keep
}

// renumber reassigns dense value ids after deletions (Compile indexes by id).
func renumber(g *Graph) {
	i := 0
	for _, v := range g.Values {
		v.id = i
		i++
	}
	for i, n := range g.Nodes {
		n.id = i
	}
}

func newConst(g *Graph, name string, t *tensor.Tensor) *Value {
	for g.Values[name] != nil {
		name += "_"
	}
	v := &Value{Name: name, Const: t, DType: t.DType(), Shape: t.Shape(), HasShape: true, id: len(g.Values)}
	g.Values[name] = v
	return v
}

// privateConst ensures n.Inputs[i] is a constant that no other node shares
// (cloning it into a new value if necessary) and returns its tensor.
func (g *Graph) privateConst(n *Node, i int) *tensor.Tensor {
	v := n.Inputs[i]
	if len(v.Consumers) == 1 && !g.isOutput(v) {
		return v.Const
	}
	nv := newConst(g, v.Name+"_"+n.Name, v.Const.Clone())
	removeConsumer(v, n)
	nv.Consumers = []*Node{n}
	n.Inputs[i] = nv
	return nv.Const
}

// convInfo describes a Conv/ConvTranspose node with constant weights.
type convInfo struct {
	n         *Node
	transpose bool
	M         int // output channels
	group     int
	w         *Value
}

func convOf(n *Node) (convInfo, bool) {
	if (n.OpType != "Conv" && n.OpType != "ConvTranspose") || n.Domain != "" {
		return convInfo{}, false
	}
	if len(n.Inputs) < 2 || n.Inputs[1] == nil || n.Inputs[1].Const == nil || n.Inputs[1].Const.DType() != tensor.F32 {
		return convInfo{}, false
	}
	if len(n.Inputs) > 2 && n.Inputs[2] != nil && n.Inputs[2].Const == nil {
		return convInfo{}, false // dynamic bias
	}
	if len(n.Outputs) != 1 || n.Outputs[0] == nil {
		return convInfo{}, false
	}
	ws := n.Inputs[1].Const.Shape()
	if len(ws) != 4 {
		return convInfo{}, false
	}
	ci := convInfo{n: n, transpose: n.OpType == "ConvTranspose", group: int(n.Attrs.Int("group", 1)), w: n.Inputs[1]}
	if ci.transpose {
		ci.M = ws[1] * ci.group
	} else {
		ci.M = ws[0]
	}
	return ci, true
}

func (ci convInfo) hasEpilogue() bool {
	return ci.n.Attrs.Has("ingot_act") || ci.n.Attrs.Has("ingot_post_scale") || ci.n.Attrs.Has("ingot_post_shift")
}

// scaleAndShift folds y ← s·y + t (per output channel) into W and bias.
func (g *Graph) scaleAndShift(ci convInfo, s, t []float32) {
	n := ci.n
	w := g.privateConst(n, 1).F32()
	ws := n.Inputs[1].Const.Shape()
	if ci.transpose {
		Cin, CoutG := ws[0], ws[1]
		CinG := Cin / ci.group
		kk := ws[2] * ws[3]
		for ic := 0; ic < Cin; ic++ {
			grp := ic / CinG
			for ocg := 0; ocg < CoutG; ocg++ {
				sc := s[grp*CoutG+ocg]
				blk := w[(ic*CoutG+ocg)*kk : (ic*CoutG+ocg+1)*kk]
				for i := range blk {
					blk[i] = float32(float64(blk[i]) * float64(sc))
				}
			}
		}
	} else {
		row := ws[1] * ws[2] * ws[3]
		for m := 0; m < ci.M; m++ {
			blk := w[m*row : (m+1)*row]
			for i := range blk {
				blk[i] = float32(float64(blk[i]) * float64(s[m]))
			}
		}
	}
	// bias
	var b []float32
	if len(n.Inputs) > 2 && n.Inputs[2] != nil {
		b = g.privateConst(n, 2).F32()
	} else {
		bv := newConst(g, n.Name+"_bias", tensor.New(tensor.F32, ci.M))
		bv.Consumers = []*Node{n}
		for len(n.Inputs) < 3 {
			n.Inputs = append(n.Inputs, nil)
		}
		n.Inputs[2] = bv
		b = bv.Const.F32()
	}
	for m := range b {
		b[m] = float32(float64(b[m])*float64(s[m]) + float64(t[m]))
	}
}

func ones(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = 1
	}
	return s
}

func setAttr(n *Node, name string, a ops.Attr) {
	if n.Attrs == nil {
		n.Attrs = ops.Attrs{}
	}
	n.Attrs[name] = a
}

func fattr(f float32) ops.Attr { return ops.Attr{Kind: ops.KindFloat, F: f} }
func sattr(s string) ops.Attr  { return ops.Attr{Kind: ops.KindString, S: s} }

// ---- passes ----

// clipBounds returns (min,max) for a Clip node when both are static.
func clipBounds(g *Graph, n *Node) (lo, hi float32, ok bool) {
	if n.OpType != "Clip" || n.Domain != "" {
		return 0, 0, false
	}
	if g.OpsetVersion("") < 11 {
		if !n.Attrs.Has("min") || !n.Attrs.Has("max") {
			return 0, 0, false
		}
		return n.Attrs.Float("min", 0), n.Attrs.Float("max", 0), true
	}
	if len(n.Inputs) < 3 {
		return 0, 0, false
	}
	lo, ok1 := scalarConst(n.Inputs[1])
	hi, ok2 := scalarConst(n.Inputs[2])
	return lo, hi, ok1 && ok2
}

// binaryWithConst matches n = op(x, c) or op(c, x) for a scalar const c and
// returns x, c, and whether the const was on the right.
func binaryWithConst(n *Node) (x *Value, c float32, constRight bool, ok bool) {
	if len(n.Inputs) != 2 || n.Inputs[0] == nil || n.Inputs[1] == nil {
		return nil, 0, false, false
	}
	if c, ok := scalarConst(n.Inputs[1]); ok && n.Inputs[0].Const == nil {
		return n.Inputs[0], c, true, true
	}
	if c, ok := scalarConst(n.Inputs[0]); ok && n.Inputs[1].Const == nil {
		return n.Inputs[1], c, false, true
	}
	return nil, 0, false, false
}

func fuseHardSwish(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	changed := false
	for _, d := range g.Nodes {
		if dead[d] || d.Domain != "" || len(d.Outputs) != 1 || d.Outputs[0] == nil {
			continue
		}
		// d: Div(m, 6) or Mul(m, 1/6)
		mv, c, right, ok := binaryWithConst(d)
		if !ok || !right {
			continue
		}
		switch {
		case d.OpType == "Div" && c == 6:
		case d.OpType == "Mul" && math.Abs(float64(c)-1.0/6) < 1e-7:
		default:
			continue
		}
		m := mv.Producer
		if m == nil || dead[m] || m.OpType != "Mul" || m.Domain != "" || g.soleConsumer(mv) != d || len(m.Inputs) != 2 {
			continue
		}
		// m: Mul(x, clipOut) either order
		var x, cv *Value
		for i := 0; i < 2; i++ {
			if p := m.Inputs[i].Producer; p != nil && p.OpType == "Clip" {
				cv, x = m.Inputs[i], m.Inputs[1-i]
			}
		}
		if cv == nil || x == nil || g.soleConsumer(cv) != m {
			continue
		}
		cl := cv.Producer
		if dead[cl] {
			continue
		}
		lo, hi, ok := clipBounds(g, cl)
		if !ok || lo != 0 || hi != 6 {
			continue
		}
		av := cl.Inputs[0]
		a := av.Producer
		if a == nil || dead[a] || a.OpType != "Add" || a.Domain != "" || g.soleConsumer(av) != cl {
			continue
		}
		ax, ac, _, ok := binaryWithConst(a)
		if !ok || ac != 3 || ax != x {
			continue
		}
		// Rewrite: hs := ingot.HardSwish(x) producing d's output.
		out := d.Outputs[0]
		hs := &Node{Name: d.Name + "_hardswish", OpType: "HardSwish", Domain: ingotDomain, Attrs: ops.Attrs{}, Inputs: []*Value{x}, Outputs: []*Value{out}}
		for _, n := range []*Node{a, cl, m, d} {
			g.dropNode(n)
			dead[n] = true
		}
		g.Values[out.Name] = out
		out.Producer = hs
		x.Consumers = append(x.Consumers, hs)
		// Place hs where d was (topological order preserved: hs depends only on x).
		for i, n := range g.Nodes {
			if n == d {
				g.Nodes[i] = hs
				break
			}
		}
		delete(dead, d) // d's slot now holds hs
		stats["fuse-hardswish"]++
		changed = true
	}
	g.compact(dead)
	if changed {
		if g.Opsets == nil {
			g.Opsets = map[string]int{}
		}
		g.Opsets[ingotDomain] = 1
	}
	return changed
}

func foldConvAffine(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	changed := false
	for _, n := range g.Nodes {
		if dead[n] {
			continue
		}
		ci, ok := convOf(n)
		if !ok || ci.hasEpilogue() {
			continue
		}
		out := n.Outputs[0]
		u := g.soleConsumer(out)
		if u == nil || dead[u] || u.Domain != "" || len(u.Outputs) < 1 || u.Outputs[0] == nil {
			continue
		}
		var s, t []float32
		switch u.OpType {
		case "Mul", "Add", "Sub":
			if len(u.Inputs) != 2 {
				continue
			}
			var cv *Value
			if u.Inputs[0] == out {
				cv = u.Inputs[1]
			} else if u.Inputs[1] == out && u.OpType != "Sub" {
				cv = u.Inputs[0]
			} else {
				continue
			}
			vec, ok := channelConst(cv, ci.M)
			if !ok {
				continue
			}
			switch u.OpType {
			case "Mul":
				s, t = vec, make([]float32, ci.M)
			case "Add":
				s, t = ones(ci.M), vec
			case "Sub":
				for i := range vec {
					vec[i] = -vec[i]
				}
				s, t = ones(ci.M), vec
			}
		case "BatchNormalization":
			if len(u.Inputs) != 5 {
				continue
			}
			usable := true
			for _, v := range u.Outputs[1:] {
				if v != nil && (len(v.Consumers) > 0 || g.isOutput(v)) {
					usable = false
				}
			}
			if !usable {
				continue
			}
			var p [4][]float32
			for i := 1; i <= 4; i++ {
				v := u.Inputs[i]
				if v == nil || v.Const == nil || v.Const.DType() != tensor.F32 || v.Const.Numel() != ci.M {
					usable = false
					break
				}
				p[i-1] = v.Const.F32()
			}
			if !usable {
				continue
			}
			eps := float64(u.Attrs.Float("epsilon", 1e-5))
			s, t = make([]float32, ci.M), make([]float32, ci.M)
			for m := 0; m < ci.M; m++ {
				sc := float64(p[0][m]) / math.Sqrt(float64(p[3][m])+eps)
				s[m] = float32(sc)
				t[m] = float32(float64(p[1][m]) - float64(p[2][m])*sc)
			}
		default:
			continue
		}
		g.scaleAndShift(ci, s, t)
		g.absorb(n, out, u, dead)
		stats["fold-conv-affine"]++
		changed = true
	}
	g.compact(dead)
	return changed
}

func fuseConvAct(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	changed := false
	for _, n := range g.Nodes {
		if dead[n] {
			continue
		}
		ci, ok := convOf(n)
		if !ok || ci.hasEpilogue() {
			continue
		}
		out := n.Outputs[0]
		u := g.soleConsumer(out)
		if u == nil || dead[u] || len(u.Outputs) != 1 || u.Outputs[0] == nil || u.Inputs[0] != out {
			continue
		}
		var act string
		var alpha, beta float32
		switch {
		case u.Domain == "" && u.OpType == "Relu":
			act = "relu"
		case u.OpType == "HardSwish" && (u.Domain == "" || u.Domain == ingotDomain):
			act = "hardswish"
		case u.OpType == "SiLU" && u.Domain == ingotDomain:
			act = "silu"
		case u.Domain == "" && u.OpType == "HardSigmoid":
			act, alpha, beta = "hardsigmoid", u.Attrs.Float("alpha", 0.2), u.Attrs.Float("beta", 0.5)
		case u.Domain == "" && u.OpType == "Sigmoid":
			act = "sigmoid"
		case u.Domain == "" && u.OpType == "LeakyRelu":
			act, alpha = "leakyrelu", u.Attrs.Float("alpha", 0.01)
		case u.Domain == "" && u.OpType == "Clip":
			lo, hi, ok := clipBounds(g, u)
			if !ok {
				continue
			}
			act, alpha, beta = "clip", lo, hi
		default:
			continue
		}
		setAttr(n, "ingot_act", sattr(act))
		setAttr(n, "ingot_act_alpha", fattr(alpha))
		setAttr(n, "ingot_act_beta", fattr(beta))
		g.absorb(n, out, u, dead)
		stats["fuse-conv-act"]++
		changed = true
	}
	g.compact(dead)
	return changed
}

func foldPostAffine(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	changed := false
	for _, n := range g.Nodes {
		if dead[n] {
			continue
		}
		_, ok := convOf(n)
		if !ok || !n.Attrs.Has("ingot_act") {
			continue
		}
		out := n.Outputs[0]
		u := g.soleConsumer(out)
		if u == nil || dead[u] || u.Domain != "" || len(u.Outputs) != 1 || u.Outputs[0] == nil {
			continue
		}
		x, c, right, ok := binaryWithConst(u)
		if !ok || x != out {
			continue
		}
		a := float64(n.Attrs.Float("ingot_post_scale", 1))
		b := float64(n.Attrs.Float("ingot_post_shift", 0))
		switch u.OpType {
		case "Mul":
			a, b = a*float64(c), b*float64(c)
		case "Add":
			b += float64(c)
		case "Sub":
			if !right {
				continue // c - x: not an affine in the supported form (would need a<0 — fine actually)
			}
			b -= float64(c)
		case "Div":
			if !right || c == 0 {
				continue
			}
			a, b = a/float64(c), b/float64(c)
		default:
			continue
		}
		setAttr(n, "ingot_post_scale", fattr(float32(a)))
		setAttr(n, "ingot_post_shift", fattr(float32(b)))
		g.absorb(n, out, u, dead)
		stats["fold-post-affine"]++
		changed = true
	}
	g.compact(dead)
	return changed
}

// fuseSiLU rewrites Mul(x, Sigmoid(x)) (either operand order) into
// ingot.SiLU(x) when the Sigmoid output has no other consumer.
func fuseSiLU(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	changed := false
	for _, m := range g.Nodes {
		if dead[m] || m.Domain != "" || m.OpType != "Mul" || len(m.Inputs) != 2 || len(m.Outputs) != 1 || m.Outputs[0] == nil {
			continue
		}
		var x, sv *Value
		for i := 0; i < 2; i++ {
			if p := m.Inputs[i].Producer; p != nil && p.OpType == "Sigmoid" && p.Domain == "" && !dead[p] && p.Inputs[0] == m.Inputs[1-i] {
				sv, x = m.Inputs[i], m.Inputs[1-i]
			}
		}
		if sv == nil || g.soleConsumer(sv) != m {
			continue
		}
		sg := sv.Producer
		out := m.Outputs[0]
		n := &Node{Name: m.Name + "_silu", OpType: "SiLU", Domain: ingotDomain, Attrs: ops.Attrs{}, Inputs: []*Value{x}, Outputs: []*Value{out}}
		g.dropNode(sg)
		g.dropNode(m)
		dead[sg] = true
		g.Values[out.Name] = out
		out.Producer = n
		x.Consumers = append(x.Consumers, n)
		for i, nn := range g.Nodes {
			if nn == m {
				g.Nodes[i] = n
				break
			}
		}
		stats["fuse-silu"]++
		changed = true
	}
	g.compact(dead)
	if changed {
		if g.Opsets == nil {
			g.Opsets = map[string]int{}
		}
		g.Opsets[ingotDomain] = 1
	}
	return changed
}

// fuseGelu rewrites the erf-GELU decomposition torch emits —
// Div(x,√2) [or Mul(x,1/√2)] → Erf → Add(·,1) → Mul with x and 0.5 in either
// order/association — into ingot.Gelu(x).
func fuseGelu(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	changed := false
	const sqrt2 = 1.4142135
	for _, e := range g.Nodes {
		if dead[e] || e.Domain != "" || e.OpType != "Erf" || len(e.Outputs) != 1 || e.Outputs[0] == nil {
			continue
		}
		d := e.Inputs[0].Producer
		if d == nil || dead[d] || g.soleConsumer(e.Inputs[0]) != e {
			continue
		}
		dx, dc, right, ok := binaryWithConst(d)
		if !ok {
			continue
		}
		switch {
		case d.OpType == "Div" && right && math.Abs(float64(dc)-sqrt2) < 1e-5:
		case d.OpType == "Mul" && math.Abs(float64(dc)-1/sqrt2) < 1e-6:
		default:
			continue
		}
		x := dx
		a := g.soleConsumer(e.Outputs[0])
		if a == nil || dead[a] || a.OpType != "Add" || a.Domain != "" {
			continue
		}
		ax, ac, _, ok := binaryWithConst(a)
		if !ok || ac != 1 || ax != e.Outputs[0] {
			continue
		}
		// After Add: two Muls combining with x and 0.5 in some order.
		m1 := g.soleConsumer(a.Outputs[0])
		if m1 == nil || dead[m1] || m1.OpType != "Mul" || m1.Domain != "" || len(m1.Inputs) != 2 {
			continue
		}
		other := m1.Inputs[0]
		if other == a.Outputs[0] {
			other = m1.Inputs[1]
		}
		var last *Node
		var halfNode *Node // the Mul by 0.5 (may be m1's other input producer, or m1's consumer)
		switch {
		case other.Const != nil:
			// m1 = 0.5*(1+erf); m2 = x*m1 (either order)
			if c, ok := scalarConst(other); !ok || c != 0.5 {
				continue
			}
			m2 := g.soleConsumer(m1.Outputs[0])
			if m2 == nil || dead[m2] || m2.OpType != "Mul" || m2.Domain != "" || len(m2.Inputs) != 2 {
				continue
			}
			if !((m2.Inputs[0] == x && m2.Inputs[1] == m1.Outputs[0]) || (m2.Inputs[1] == x && m2.Inputs[0] == m1.Outputs[0])) {
				continue
			}
			last = m2
		case other == x:
			// m1 = x*(1+erf); m2 = m1*0.5
			m2 := g.soleConsumer(m1.Outputs[0])
			if m2 == nil || dead[m2] || m2.OpType != "Mul" || m2.Domain != "" {
				continue
			}
			mx, mc, _, ok := binaryWithConst(m2)
			if !ok || mc != 0.5 || mx != m1.Outputs[0] {
				continue
			}
			last = m2
		default:
			// other = Mul(x, 0.5) (or Mul(0.5, x)), sole consumer m1
			h := other.Producer
			if h == nil || dead[h] || h.OpType != "Mul" || h.Domain != "" || g.soleConsumer(other) != m1 {
				continue
			}
			hx, hc, _, ok := binaryWithConst(h)
			if !ok || hc != 0.5 || hx != x {
				continue
			}
			halfNode = h
			last = m1
		}
		out := last.Outputs[0]
		n := &Node{Name: last.Name + "_gelu", OpType: "Gelu", Domain: ingotDomain, Attrs: ops.Attrs{}, Inputs: []*Value{x}, Outputs: []*Value{out}}
		toDrop := []*Node{d, e, a, m1}
		if halfNode != nil {
			toDrop = append(toDrop, halfNode)
		}
		if last != m1 {
			toDrop = append(toDrop, last)
		}
		for _, nn := range toDrop {
			g.dropNode(nn)
			dead[nn] = true
		}
		g.Values[out.Name] = out
		out.Producer = n
		x.Consumers = append(x.Consumers, n)
		for i, nn := range g.Nodes {
			if nn == last {
				g.Nodes[i] = n
				break
			}
		}
		delete(dead, last)
		stats["fuse-gelu"]++
		changed = true
	}
	g.compact(dead)
	if changed {
		if g.Opsets == nil {
			g.Opsets = map[string]int{}
		}
		g.Opsets[ingotDomain] = 1
	}
	return changed
}
