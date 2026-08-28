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
//   - fold-const:       any node whose inputs are all constants is evaluated at
//     load time with the same registered op the executor would run (bit-identical
//     by construction) and replaced by constant values. Sized-guarded so a
//     broadcast blow-up is never baked into the model.
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
//   - fuse-se:          GlobalAveragePool → 1x1 Conv → 1x1 Conv → Mul  ⇒
//     ingot.SE (the squeeze-excite island runs inline).
//
// Every rewrite keeps the consumer's output Value (so graph outputs and names
// survive) and only fires when intermediate values have a single consumer.
func Optimize(g *Graph) map[string]int {
	stats := map[string]int{}
	for changed := true; changed; {
		changed = false
		changed = foldConst(g, stats) || changed
		changed = fuseHardSwish(g, stats) || changed
		changed = fuseSiLU(g, stats) || changed
		changed = fuseGelu(g, stats) || changed
		changed = foldConvAffine(g, stats) || changed
		changed = fuseConvAct(g, stats) || changed
		changed = foldPostAffine(g, stats) || changed
		changed = fuseSE(g, stats) || changed
		changed = foldQDQAffine(g, stats) || changed
		changed = fuseQLUT(g, stats) || changed
		changed = fuseLayerNorm(g, stats) || changed
		changed = fuseAttention(g, stats) || changed
		changed = fuseSDPA(g, stats) || changed
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

// ---- quantized-island passes ----

// intZP reads a scalar integer zero point (u8/i8/i32 const; nil means 0).
func intZP(v *Value) (int32, tensor.DType, bool) {
	if v == nil {
		return 0, tensor.U8, true
	}
	if v.Const == nil || v.Const.Numel() != 1 {
		return 0, 0, false
	}
	switch v.Const.DType() {
	case tensor.U8:
		return int32(v.Const.U8()[0]), tensor.U8, true
	case tensor.I8:
		return int32(v.Const.I8()[0]), tensor.I8, true
	case tensor.I32:
		return v.Const.I32()[0], tensor.I32, true
	}
	return 0, 0, false
}

func zpTensor(dt tensor.DType, zp int32) *tensor.Tensor {
	t := tensor.New(dt)
	switch dt {
	case tensor.U8:
		t.U8()[0] = uint8(zp)
	case tensor.I8:
		t.I8()[0] = int8(zp)
	case tensor.I32:
		t.I32()[0] = zp
	}
	return t
}

// foldQDQAffine folds QLinearConv → DequantizeLinear → (scalar Mul/Add/Sub/Div)*
// → QuantizeLinear into the conv's own output quantization: the conv's y_scale
// becomes s2/a and its y_zero_point rne(b/s2 + z2) — the requantized result
// then encodes a·r+b at (s2, z2), exactly what the chain's consumers expect.
// One fewer rounding than the original chain; the zero point must land on a
// representable integer or the site is skipped.
func foldQDQAffine(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	changed := false
	for _, n := range g.Nodes {
		if dead[n] || n.OpType != "QLinearConv" || n.Domain != "" || len(n.Inputs) < 8 {
			continue
		}
		ys, ok := scalarConst(n.Inputs[6])
		if !ok {
			continue
		}
		yz, yzdt, ok := intZP(n.Inputs[7])
		if !ok {
			continue
		}
		dq := g.soleConsumer(n.Outputs[0])
		if dq == nil || dead[dq] || dq.OpType != "DequantizeLinear" || dq.Domain != "" {
			continue
		}
		ds, ok := scalarConst(dq.Inputs[1])
		var dz int32
		if !ok {
			continue
		}
		if len(dq.Inputs) > 2 {
			var okz bool
			dz, _, okz = intZP(dq.Inputs[2])
			if !okz {
				continue
			}
		}
		if ds != ys || dz != yz {
			continue
		}
		// Walk the scalar affine chain.
		a, b := 1.0, 0.0
		chain := []*Node{dq}
		cur := dq.Outputs[0]
		for {
			u := g.soleConsumer(cur)
			if u == nil || dead[u] || u.Domain != "" {
				chain = nil
				break
			}
			if u.OpType == "QuantizeLinear" {
				chain = append(chain, u)
				break
			}
			x, c, right, ok := binaryWithConst(u)
			if !ok || x != cur {
				chain = nil
				break
			}
			switch u.OpType {
			case "Mul":
				a, b = a*float64(c), b*float64(c)
			case "Add":
				b += float64(c)
			case "Sub":
				if !right {
					chain = nil
				} else {
					b -= float64(c)
				}
			case "Div":
				if !right || c == 0 {
					chain = nil
				} else {
					a, b = a/float64(c), b/float64(c)
				}
			default:
				chain = nil
			}
			if chain == nil {
				break
			}
			chain = append(chain, u)
			cur = u.Outputs[0]
		}
		if chain == nil || a == 0 {
			continue
		}
		q := chain[len(chain)-1]
		s2, ok := scalarConst(q.Inputs[1])
		if !ok || s2 == 0 {
			continue
		}
		var z2 int32
		z2dt := tensor.U8
		if len(q.Inputs) > 2 {
			z2, z2dt, ok = intZP(q.Inputs[2])
			if !ok {
				continue
			}
		}
		if z2dt != yzdt || q.Outputs[0].DType != n.Outputs[0].DType {
			continue
		}
		newScale := float64(s2) / a
		newZP := math.RoundToEven(b/float64(s2)) + float64(z2)
		lo, hi := 0.0, 255.0
		if z2dt == tensor.I8 {
			lo, hi = -128, 127
		}
		if newScale <= 0 || newZP < lo || newZP > hi || newZP != math.Trunc(newZP) && math.Abs(newZP-math.RoundToEven(newZP)) > 1e-3 {
			continue
		}
		// Rewire: conv writes q's output directly with adjusted (scale, zp).
		st := tensor.New(tensor.F32)
		st.F32()[0] = float32(newScale)
		sv := newConst(g, q.Outputs[0].Name+"_fold_scale", st)
		zv := newConst(g, q.Outputs[0].Name+"_fold_zp", zpTensor(z2dt, int32(math.RoundToEven(newZP))))
		removeConsumer(n.Inputs[6], n)
		removeConsumer(n.Inputs[7], n)
		n.Inputs[6], n.Inputs[7] = sv, zv
		sv.Consumers = append(sv.Consumers, n)
		zv.Consumers = append(zv.Consumers, n)
		out := q.Outputs[0]
		mid := n.Outputs[0]
		for _, cn := range chain[:len(chain)-1] {
			dead[cn] = true
		}
		removeConsumer(mid, dq)
		for _, cn := range chain {
			if cn != q {
				g.dropNode(cn)
			}
		}
		g.dropNode(q)
		g.Values[out.Name] = out
		out.Producer = n
		n.Outputs[0] = out
		delete(g.Values, mid.Name)
		dead[q] = true
		stats["fold-qdq-affine"]++
		changed = true
	}
	g.compact(dead)
	return changed
}

// channelConstAny returns the flattened values of an f32 constant whose shape
// broadcasts over NCHW channels ([..,C,1,1] or scalar).
func channelConstAny(v *Value) ([]float32, bool) {
	if v == nil || v.Const == nil || v.Const.DType() != tensor.F32 {
		return nil, false
	}
	f := v.Const.F32()
	if len(f) == 1 {
		return f, true
	}
	sh := v.Const.Shape()
	if len(sh) < 3 || sh[len(sh)-1] != 1 || sh[len(sh)-2] != 1 || sh[len(sh)-3] != len(f) {
		return nil, false
	}
	for _, d := range sh[:len(sh)-3] {
		if d != 1 {
			return nil, false
		}
	}
	return f, true
}

// bnAffine reduces a BatchNormalization with constant parameters to the
// per-channel affine y = scale·x + shift.
func bnAffine(u *Node, x *Value) (scale, shift []float32, ok bool) {
	if len(u.Inputs) < 5 || u.Inputs[0] != x {
		return nil, nil, false
	}
	var p [4][]float32
	for i := 1; i < 5; i++ {
		v := u.Inputs[i]
		if v == nil || v.Const == nil || v.Const.DType() != tensor.F32 {
			return nil, nil, false
		}
		p[i-1] = v.Const.F32()
	}
	gamma, beta, mean, vr := p[0], p[1], p[2], p[3]
	C := len(gamma)
	if len(beta) != C || len(mean) != C || len(vr) != C {
		return nil, nil, false
	}
	eps := float64(u.Attrs.Float("epsilon", 1e-5))
	scale = make([]float32, C)
	shift = make([]float32, C)
	for c := 0; c < C; c++ {
		inv := 1 / math.Sqrt(float64(vr[c])+eps)
		scale[c] = float32(float64(gamma[c]) * inv)
		shift[c] = beta[c] - scale[c]*mean[c]
	}
	return scale, shift, true
}

// fuseQLUT collapses DequantizeLinear → (scalar elementwise chain) →
// QuantizeLinear (u8 in, u8 out, per-tensor scales) into a single
// ingot.QLut node: the whole island is a pure function of one byte, so it
// is exactly representable as a 256-entry table built at optimize time.
func fuseQLUT(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	changed := false
	for _, n := range g.Nodes {
		if dead[n] || n.OpType != "DequantizeLinear" || n.Domain != "" {
			continue
		}
		if n.Inputs[0] == nil {
			continue
		}
		s1, ok := scalarConst(n.Inputs[1])
		if !ok {
			continue
		}
		// x dtype comes from the zero point (the spec ties them); intermediate
		// Value.DType is not populated at optimize time.
		if len(n.Inputs) < 3 || n.Inputs[2] == nil {
			continue
		}
		z1, z1dt, okz := intZP(n.Inputs[2])
		if !okz || z1dt != tensor.U8 {
			continue
		}
		// Collect the elementwise chain as a composed float function.
		type stage struct {
			op    string
			c     float64
			cv    []float32 // per-channel const (Mul/Add/Sub) or BN scale
			bv    []float32 // BN shift
			right bool
			lo,
			hi float64
			alpha, beta float64
		}
		var stages []stage
		chain := []*Node{}
		cur := n.Outputs[0]
		var q *Node
		for {
			u := g.soleConsumer(cur)
			if u == nil || dead[u] {
				break
			}
			if u.OpType == "QuantizeLinear" && u.Domain == "" {
				q = u
				break
			}
			st := stage{op: u.OpType}
			switch {
			case u.Domain == "" && (u.OpType == "Mul" || u.OpType == "Add" || u.OpType == "Sub" || u.OpType == "Div"):
				x, c, right, ok := binaryWithConst(u)
				if ok && x == cur {
					st.c, st.right = float64(c), right
					break
				}
				// Per-channel [..,C,1,1] constant? (Sub only with x on the left.)
				if u.OpType != "Div" && len(u.Inputs) == 2 {
					xi, ci := u.Inputs[0], u.Inputs[1]
					if xi != cur {
						xi, ci = ci, xi
					}
					if xi == cur && ci != nil && ci.Const != nil && (u.OpType != "Sub" || u.Inputs[0] == cur) {
						if cv, okc := channelConstAny(ci); okc {
							st.cv = cv
							break
						}
					}
				}
				u = nil
			case u.Domain == "" && u.OpType == "Clip":
				lo, hi, ok := clipBounds(g, u)
				if !ok || u.Inputs[0] != cur {
					u = nil
				} else {
					st.lo, st.hi = float64(lo), float64(hi)
				}
			case u.Domain == "" && u.OpType == "Relu":
			case u.Domain == "" && u.OpType == "Sigmoid":
			case u.OpType == "HardSwish" && (u.Domain == "" || u.Domain == ingotDomain):
				st.op = "HardSwish"
			case u.Domain == "" && u.OpType == "HardSigmoid":
				st.alpha = float64(u.Attrs.Float("alpha", 0.2))
				st.beta = float64(u.Attrs.Float("beta", 0.5))
			case u.Domain == "" && u.OpType == "BatchNormalization":
				sc, sh, okbn := bnAffine(u, cur)
				if !okbn {
					u = nil
				} else {
					st.cv, st.bv = sc, sh
				}
			default:
				u = nil
			}
			if u == nil || len(u.Outputs) != 1 || u.Outputs[0] == nil {
				break
			}
			stages = append(stages, st)
			chain = append(chain, u)
			cur = u.Outputs[0]
		}
		if q == nil || len(chain) == 0 {
			continue
		}
		s2, ok := scalarConst(q.Inputs[1])
		if !ok || s2 <= 0 {
			continue
		}
		if len(q.Inputs) < 3 || q.Inputs[2] == nil {
			continue // need the zp to know the output dtype
		}
		z2, z2dt, okz2 := intZP(q.Inputs[2])
		if !okz2 || z2dt != tensor.U8 {
			continue
		}
		// Build the table: one 256-entry row, or C rows when any stage is
		// per-channel (the runtime picks row c for channel c).
		C := 1
		for _, st := range stages {
			if st.cv != nil && len(st.cv) > C {
				C = len(st.cv)
			}
		}
		mismatch := false
		for _, st := range stages {
			if st.cv != nil && len(st.cv) != C && len(st.cv) != 1 {
				mismatch = true
			}
		}
		if mismatch {
			continue
		}
		eval := func(f float64, c int) float64 {
			for _, st := range stages {
				cc := st.c
				if st.cv != nil {
					cc = float64(st.cv[min(c, len(st.cv)-1)])
				}
				switch st.op {
				case "Mul":
					f *= cc
				case "Add":
					f += cc
				case "Sub":
					if st.right || st.cv != nil {
						f -= cc
					} else {
						f = cc - f
					}
				case "Div":
					if st.right {
						f /= cc
					} else if f != 0 {
						f = cc / f
					}
				case "Clip":
					f = math.Min(math.Max(f, st.lo), st.hi)
				case "Relu":
					f = math.Max(f, 0)
				case "Sigmoid":
					f = 1 / (1 + math.Exp(-f))
				case "HardSwish":
					f = f * math.Min(math.Max(f+3, 0), 6) / 6
				case "HardSigmoid":
					f = math.Min(math.Max(st.alpha*f+st.beta, 0), 1)
				case "BatchNormalization":
					f = f*cc + float64(st.bv[min(c, len(st.bv)-1)])
				}
			}
			return f
		}
		var tb *tensor.Tensor
		if C == 1 {
			tb = tensor.New(tensor.U8, 256)
		} else {
			tb = tensor.New(tensor.U8, C, 256)
		}
		lut := tb.U8()
		for c := 0; c < C; c++ {
			for v := 0; v < 256; v++ {
				f := eval(float64(s1)*float64(int32(v)-z1), c)
				qv := math.RoundToEven(f/float64(s2)) + float64(z2)
				lut[c*256+v] = uint8(math.Min(math.Max(qv, 0), 255))
			}
		}
		// Rewire: mutate the DQ node in place (its position is trivially
		// topological) into QLut(x, table) producing q's output.
		out := q.Outputs[0]
		x := n.Inputs[0]
		mid := n.Outputs[0]
		tv := newConst(g, out.Name+"_lut", tb)
		for _, v := range n.Inputs[1:] {
			if v != nil {
				removeConsumer(v, n)
				if v.Const != nil && len(v.Consumers) == 0 && !g.isOutput(v) {
					delete(g.Values, v.Name)
				}
			}
		}
		removeConsumer(mid, chain[0])
		for _, cn := range chain {
			dead[cn] = true
			g.dropNode(cn)
		}
		dead[q] = true
		g.dropNode(q)
		delete(g.Values, mid.Name)
		g.Values[out.Name] = out
		n.OpType, n.Domain = "QLut", ingotDomain
		g.Opsets[ingotDomain] = 1
		n.Attrs = nil
		n.Inputs = []*Value{x, tv}
		tv.Consumers = append(tv.Consumers, n)
		n.Outputs = []*Value{out}
		out.Producer = n
		stats["fuse-qlut"]++
		changed = true
	}
	g.compact(dead)
	return changed
}

// fuseLayerNorm rewrites the decomposed pattern
//
//	m = ReduceMean(x); d = Sub(x, m); v = ReduceMean(Pow(d, 2));
//	y = Div(d, Sqrt(Add(v, eps))); out = Add(Mul(y, gamma), beta)
//
// into one ingot.LayerNorm(x, gamma, beta) node.
func fuseLayerNorm(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	changed := false
	for _, rm1 := range g.Nodes {
		if dead[rm1] || rm1.OpType != "ReduceMean" || rm1.Domain != "" {
			continue
		}
		x := rm1.Inputs[0]
		sub := g.soleConsumer(rm1.Outputs[0])
		if sub == nil || dead[sub] || sub.OpType != "Sub" || sub.Inputs[0] != x || sub.Inputs[1] != rm1.Outputs[0] {
			continue
		}
		d := sub.Outputs[0]
		if len(d.Consumers) != 2 || g.isOutput(d) {
			continue
		}
		var pow, div *Node
		for _, c := range d.Consumers {
			switch c.OpType {
			case "Pow":
				pow = c
			case "Div":
				div = c
			}
		}
		if pow == nil || div == nil || dead[pow] || dead[div] || div.Inputs[0] != d {
			continue
		}
		if e, ok := scalarConst(pow.Inputs[1]); !ok || e != 2 {
			continue
		}
		rm2 := g.soleConsumer(pow.Outputs[0])
		if rm2 == nil || rm2.OpType != "ReduceMean" {
			continue
		}
		ax1 := rm1.Attrs.Ints("axes", nil)
		ax2 := rm2.Attrs.Ints("axes", nil)
		if len(ax1) != 1 || len(ax2) != 1 || ax1[0] != ax2[0] {
			continue
		}
		// The normalized axis must be the innermost one; a positive axis is
		// only verifiable against a known rank.
		if ax1[0] != -1 {
			if !x.HasShape || int(ax1[0]) != len(x.Shape)-1 {
				continue
			}
		}
		addE := g.soleConsumer(rm2.Outputs[0])
		if addE == nil || addE.OpType != "Add" {
			continue
		}
		eps, ok := scalarConst(addE.Inputs[1])
		if !ok {
			eps, ok = scalarConst(addE.Inputs[0])
		}
		if !ok {
			continue
		}
		sq := g.soleConsumer(addE.Outputs[0])
		if sq == nil || sq.OpType != "Sqrt" || g.soleConsumer(sq.Outputs[0]) != div || div.Inputs[1] != sq.Outputs[0] {
			continue
		}
		// Fold the trailing Mul(gamma)/Add(beta) only when both are plain
		// 1-D vectors — those provably match the normalized (innermost)
		// dim. A [C,1]-shaped gamma is a separate per-channel affine (the
		// pattern normalizes one axis, scales another): fuse the core only.
		var gamma, beta *Value
		mul := g.soleConsumer(div.Outputs[0])
		var add2 *Node
		if mul != nil && mul.OpType == "Mul" {
			for _, v := range mul.Inputs {
				if v != div.Outputs[0] && v != nil && v.Const != nil && v.Const.DType() == tensor.F32 && len(v.Const.Shape()) == 1 {
					gamma = v
				}
			}
			if gamma != nil {
				add2 = g.soleConsumer(mul.Outputs[0])
				if add2 != nil && add2.OpType == "Add" {
					for _, v := range add2.Inputs {
						if v != mul.Outputs[0] && v != nil && v.Const != nil && v.Const.DType() == tensor.F32 && len(v.Const.Shape()) == 1 {
							beta = v
						}
					}
				}
				if beta == nil {
					gamma, add2 = nil, nil
				}
			}
		}
		drop := []*Node{sub, pow, rm2, addE, sq, div}
		out := div.Outputs[0]
		ins := []*Value{x}
		if gamma != nil {
			drop = append(drop, mul, add2)
			out = add2.Outputs[0]
			ins = append(ins, gamma, beta)
			// Re-home gamma/beta onto rm1 BEFORE dropping mul/add2 — a
			// consumer-less const gets deleted from g.Values by dropNode,
			// keeping a stale id that collides after renumber.
			removeConsumer(gamma, mul)
			removeConsumer(beta, add2)
			gamma.Consumers = append(gamma.Consumers, rm1)
			beta.Consumers = append(beta.Consumers, rm1)
		}
		for _, n2 := range drop {
			dead[n2] = true
		}
		removeConsumer(x, sub)
		for _, n2 := range drop {
			g.dropNode(n2)
		}
		mid := rm1.Outputs[0]
		delete(g.Values, mid.Name)
		g.Values[out.Name] = out
		rm1.OpType, rm1.Domain = "LayerNorm", ingotDomain
		rm1.Attrs = ops.Attrs{"axis": {Kind: ops.KindInt, I: -1}, "epsilon": fattr(eps)}
		rm1.Inputs = ins
		rm1.Outputs = []*Value{out}
		out.Producer = rm1
		g.Opsets[ingotDomain] = 1
		stats["fuse-layernorm"]++
		changed = true
	}
	g.compact(dead)
	return changed
}

// sliceStart matches Slice(x, axis 0, starts s, ends s+1, steps 1) and
// returns s. Bounds come as 1-element const inputs (opset ≥10).
func sliceStart(n *Node) (int64, bool) {
	if n.OpType != "Slice" || n.Domain != "" || len(n.Inputs) < 4 {
		return 0, false
	}
	get := func(i int) (int64, bool) {
		if i >= len(n.Inputs) || n.Inputs[i] == nil || n.Inputs[i].Const == nil ||
			n.Inputs[i].Const.DType() != tensor.I64 || n.Inputs[i].Const.Numel() != 1 {
			return 0, false
		}
		return n.Inputs[i].Const.I64()[0], true
	}
	st, ok1 := get(1)
	en, ok2 := get(2)
	ax, ok3 := get(3)
	if !ok1 || !ok2 || !ok3 || ax != 0 || en != st+1 {
		return 0, false
	}
	if len(n.Inputs) > 4 && n.Inputs[4] != nil {
		if sp, ok := get(4); !ok || sp != 1 {
			return 0, false
		}
	}
	return st, true
}

func permIs(n *Node, want ...int64) bool {
	p := n.Attrs.Ints("perm", nil)
	if len(p) != len(want) {
		return false
	}
	for i := range p {
		if p[i] != want[i] {
			return false
		}
	}
	return true
}

// fuseAttention rewrites the exported multi-head-attention pattern
//
//	x[3,B,H,T,dh] → Slice{0,1,2}+Squeeze → q·scale, Kᵀ([0,1,3,2])
//	→ MatMul → Softmax(last) → MatMul(·V) → Transpose([0,2,1,3])
//
// into one ingot.MHA node producing the transposed output directly.
func fuseAttention(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	changed := false
	for _, sm := range g.Nodes {
		if dead[sm] || sm.OpType != "Softmax" || sm.Domain != "" {
			continue
		}
		if ax := sm.Attrs.Int("axis", -1); ax != -1 && ax != 3 {
			continue
		}
		mm1 := sm.Inputs[0].Producer
		if mm1 == nil || dead[mm1] || mm1.OpType != "MatMul" || g.soleConsumer(mm1.Outputs[0]) != sm {
			continue
		}
		// q side: optional scalar Mul.
		qv := mm1.Inputs[0]
		scale := float32(1)
		var mulN *Node
		if p := qv.Producer; p != nil && p.OpType == "Mul" {
			if x, c, _, ok := binaryWithConst(p); ok && g.soleConsumer(p.Outputs[0]) == mm1 {
				scale, qv, mulN = c, x, p
			}
		}
		// k side: Transpose [0,1,3,2].
		ktr := mm1.Inputs[1].Producer
		if ktr == nil || ktr.OpType != "Transpose" || !permIs(ktr, 0, 1, 3, 2) || g.soleConsumer(ktr.Outputs[0]) != mm1 {
			continue
		}
		kv := ktr.Inputs[0]
		// probs → MatMul(·V) → Transpose [0,2,1,3].
		mm2 := g.soleConsumer(sm.Outputs[0])
		if mm2 == nil || mm2.OpType != "MatMul" || mm2.Inputs[0] != sm.Outputs[0] {
			continue
		}
		vv := mm2.Inputs[1]
		otr := g.soleConsumer(mm2.Outputs[0])
		if otr == nil || otr.OpType != "Transpose" || !permIs(otr, 0, 2, 1, 3) {
			continue
		}
		// q/k/v each: Squeeze(axes=[0]) of Slice(x5, start 0/1/2).
		unpack := func(v *Value) (*Node, *Node, *Value, int64, bool) {
			sq := v.Producer
			if sq == nil || sq.OpType != "Squeeze" {
				return nil, nil, nil, 0, false
			}
			ax := sq.Attrs.Ints("axes", nil)
			if len(ax) != 1 || ax[0] != 0 {
				return nil, nil, nil, 0, false
			}
			sl := sq.Inputs[0].Producer
			if sl == nil || g.soleConsumer(sl.Outputs[0]) != sq {
				return nil, nil, nil, 0, false
			}
			st, ok := sliceStart(sl)
			if !ok {
				return nil, nil, nil, 0, false
			}
			return sq, sl, sl.Inputs[0], st, true
		}
		qsq, qsl, x5q, s0, ok0 := unpack(qv)
		ksq, ksl, x5k, s1, ok1 := unpack(kv)
		vsq, vsl, x5v, s2, ok2 := unpack(vv)
		if !ok0 || !ok1 || !ok2 || x5q != x5k || x5q != x5v || s0 != 0 || s1 != 1 || s2 != 2 {
			continue
		}
		// Consumer hygiene: q/k/v squeezes feed only this pattern.
		if g.soleConsumer(qsq.Outputs[0]) == nil || g.soleConsumer(ksq.Outputs[0]) != ktr || g.soleConsumer(vsq.Outputs[0]) != mm2 {
			continue
		}
		out := otr.Outputs[0]
		drop := []*Node{qsl, ksl, vsl, qsq, ksq, vsq, ktr, mm1, sm, mm2, otr}
		if mulN != nil {
			drop = append(drop, mulN)
		}
		// Mutate the q-slice node (earliest safe position) into the MHA node.
		host := qsl
		for _, n2 := range drop {
			dead[n2] = true
			if n2 != host {
				g.dropNode(n2)
			}
		}
		for _, v := range host.Inputs[1:] {
			if v != nil {
				removeConsumer(v, host)
				if v.Const != nil && len(v.Consumers) == 0 && !g.isOutput(v) {
					delete(g.Values, v.Name)
				}
			}
		}
		mid := host.Outputs[0]
		delete(g.Values, mid.Name)
		g.Values[out.Name] = out
		dead[host] = false
		host.OpType, host.Domain = "MHA", ingotDomain
		host.Attrs = ops.Attrs{"scale": fattr(scale)}
		host.Inputs = host.Inputs[:1]
		host.Outputs = []*Value{out}
		out.Producer = host
		g.Opsets[ingotDomain] = 1
		stats["fuse-attention"]++
		changed = true
	}
	g.compact(dead)
	return changed
}

// fuseSDPA rewrites the generic attention core
//
//	MatMul(a, b) → [Div/Mul scalar] → [Add mask-const] → Softmax(last axis)
//	→ MatMul(probs, v) [→ Transpose 0,2,1,3]
//
// (with optional scalar Muls peeled off either MatMul input — torch splits
// the 1/√d as a factor on each side) into one ingot.SDPA node. The B input
// is whatever tensor the graph fed the first MatMul: no claim is made about
// how it was produced — the win is folding the scale into the GEMM, the
// softmax running on the hot score tile, and the [B,H,T,Tk] intermediates
// never touching memory.
// transProducer returns v's producer when it is a sole-consumed Transpose
// with exactly this perm (a peelable head-split permute), else nil.
func transProducer(g *Graph, v *Value, dead map[*Node]bool, perm ...int64) *Node {
	p := v.Producer
	if p == nil || dead[p] || p.OpType != "Transpose" || p.Domain != "" ||
		p.Inputs[0] == nil || g.soleConsumer(v) == nil || !permIs(p, perm...) {
		return nil
	}
	return p
}

func fuseSDPA(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	changed := false
	for _, sm := range g.Nodes {
		if dead[sm] || sm.OpType != "Softmax" || sm.Domain != "" {
			continue
		}
		if ax := sm.Attrs.Int("axis", -1); ax != -1 && ax != 3 {
			continue
		}
		scale := 1.0
		cur := sm.Inputs[0]
		var mask *Value
		var mid []*Node
		// Peel [Add mask] then [Div/Mul scalar] (either order) back to the MatMul.
		for i := 0; i < 2; i++ {
			p := cur.Producer
			if p == nil || dead[p] || p.Domain != "" || g.soleConsumer(cur) == nil {
				break
			}
			if p.OpType == "Add" && mask == nil && len(p.Inputs) == 2 {
				var mv *Value
				switch {
				case p.Inputs[1] != nil && p.Inputs[1].Const != nil:
					mv = p.Inputs[1]
				case p.Inputs[0] != nil && p.Inputs[0].Const != nil:
					mv = p.Inputs[0]
				}
				if mv == nil || mv.Const.DType() != tensor.F32 {
					break
				}
				mask = mv
				mid = append(mid, p)
				if p.Inputs[0] == mv {
					cur = p.Inputs[1]
				} else {
					cur = p.Inputs[0]
				}
				continue
			}
			if x, c, right, ok := binaryWithConst(p); ok && (p.OpType == "Mul" || (p.OpType == "Div" && right && c != 0)) {
				if p.OpType == "Mul" {
					scale *= float64(c)
				} else {
					scale /= float64(c)
				}
				mid = append(mid, p)
				cur = x
				continue
			}
			break
		}
		mm1 := cur.Producer
		if mm1 == nil || dead[mm1] || mm1.OpType != "MatMul" || mm1.Domain != "" || g.soleConsumer(mm1.Outputs[0]) == nil {
			continue
		}
		// Peel a scalar Mul off each MatMul input.
		peel := func(v *Value) (*Value, *Node) {
			p := v.Producer
			if p == nil || dead[p] || p.OpType != "Mul" || g.soleConsumer(v) != mm1 {
				return v, nil
			}
			if x, c, _, ok := binaryWithConst(p); ok {
				scale *= float64(c)
				return x, p
			}
			return v, nil
		}
		av, amul := peel(mm1.Inputs[0])
		bv, bmul := peel(mm1.Inputs[1])
		mm2 := g.soleConsumer(sm.Outputs[0])
		if mm2 == nil || dead[mm2] || mm2.OpType != "MatMul" || mm2.Inputs[0] != sm.Outputs[0] {
			continue
		}
		vv := mm2.Inputs[1]
		drop := append(mid, sm, mm2)
		// Peel the head-split transposes off A, B and V: the op reads these
		// layouts strided, so the permute copies vanish. Layout codes:
		// 0 = as-declared ([B,H,T,dh] / [B,H,dh,Tk] / [B,H,Tk,dh]),
		// 1 = [B,T,H,dh] (peeled [0,2,1,3]),
		// 2 = B only: [B,H,Tk,dh] (peeled [0,1,3,2] — legacy-export Kᵀ).
		aLay, bLay, vLay := 0, 0, 0
		if t := transProducer(g, av, dead, 0, 2, 1, 3); t != nil {
			drop = append(drop, t)
			av, aLay = t.Inputs[0], 1
		}
		if t := transProducer(g, bv, dead, 0, 2, 3, 1); t != nil {
			drop = append(drop, t)
			bv, bLay = t.Inputs[0], 1
		} else if t := transProducer(g, bv, dead, 0, 1, 3, 2); t != nil {
			drop = append(drop, t)
			bv, bLay = t.Inputs[0], 2
		}
		if t := transProducer(g, vv, dead, 0, 2, 1, 3); t != nil {
			drop = append(drop, t)
			vv, vLay = t.Inputs[0], 1
		}
		out := mm2.Outputs[0]
		strideOut := 0
		var otr *Node
		if t := g.soleConsumer(out); t != nil && t.OpType == "Transpose" && permIs(t, 0, 2, 1, 3) {
			otr = t
			out = t.Outputs[0]
			strideOut = 1
		}
		if amul != nil {
			drop = append(drop, amul)
		}
		if bmul != nil {
			drop = append(drop, bmul)
		}
		if otr != nil {
			drop = append(drop, otr)
		}
		host := mm1
		ins := []*Value{av, bv, vv}
		if mask != nil {
			ins = append(ins, mask)
		}
		// Attach the new inputs BEFORE dropping the pattern nodes: a const
		// (the mask; a peeled-Mul operand) that momentarily loses its last
		// consumer is deleted from g.Values by dropNode and keeps a stale
		// value id — the executor then feeds the wrong tensor entirely.
		// (Same landmine as fuse-layernorm's gamma; see PERF.md.)
		for _, v := range ins {
			v.Consumers = append(v.Consumers, host)
		}
		oldIns := host.Inputs
		for _, n2 := range drop {
			dead[n2] = true
			g.dropNode(n2)
		}
		for _, v := range oldIns {
			if v != nil {
				removeConsumer(v, host)
			}
		}
		mid1 := host.Outputs[0]
		delete(g.Values, mid1.Name)
		g.Values[out.Name] = out
		host.OpType, host.Domain = "SDPA", ingotDomain
		host.Attrs = ops.Attrs{
			"scale":      fattr(float32(scale)),
			"stride_out": {Kind: ops.KindInt, I: int64(strideOut)},
			"a_layout":   {Kind: ops.KindInt, I: int64(aLay)},
			"b_layout":   {Kind: ops.KindInt, I: int64(bLay)},
			"v_layout":   {Kind: ops.KindInt, I: int64(vLay)},
		}
		host.Inputs = ins
		host.Outputs = []*Value{out}
		out.Producer = host
		g.Opsets[ingotDomain] = 1
		stats["fuse-sdpa"]++
		changed = true
	}
	g.compact(dead)
	return changed
}

// foldConst evaluates nodes whose inputs are all constant and replaces their
// outputs with constants, running the same registered op the executor would —
// folded values are bit-identical to what execution would have produced.
// Ops that are unregistered here, fail to build, or fail to run are left
// alone: Compile reports them later with proper context, and folding must
// never turn a working graph into a broken one.
//
// A size guard skips folds that would inflate resident memory: the folded
// outputs must fit in max(16 MiB, 8× the const input bytes). This permits
// scalar chains and weight-shaped transforms while refusing to bake a
// broadcast blow-up (e.g. Expand of a scalar to an activation shape) into
// the model.
func foldConst(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	for _, n := range g.Nodes {
		if len(n.Inputs) == 0 {
			continue // Constant is folded at load; anything else 0-input is not ours to guess
		}
		foldable, inBytes := true, 0
		ins := make([]*tensor.Tensor, len(n.Inputs))
		for i, v := range n.Inputs {
			if v == nil {
				continue
			}
			if v.Const == nil {
				foldable = false
				break
			}
			ins[i] = v.Const
			inBytes += v.Const.Numel() * v.Const.DType().Size()
		}
		if !foldable {
			continue
		}
		ver := g.OpsetVersion(n.Domain)
		b, err := ops.Lookup(n.Domain, n.OpType, ver)
		if err != nil {
			continue
		}
		op, err := b(ops.NodeInfo{
			Name: n.Name, OpType: n.OpType, Domain: n.Domain, Version: ver,
			Attrs: n.Attrs, NumIn: len(n.Inputs), NumOut: len(n.Outputs),
		})
		if err != nil {
			continue
		}
		outs, err := op.Run(&ops.Ctx{}, ins)
		if err != nil || len(outs) < len(n.Outputs) {
			continue
		}
		outBytes := 0
		for i, v := range n.Outputs {
			if v == nil {
				continue
			}
			if outs[i] == nil {
				foldable = false // op declined an output we need
				break
			}
			outBytes += outs[i].Numel() * outs[i].DType().Size()
		}
		lim := 8 * inBytes
		if lim < 16<<20 {
			// The flat cap admits attention masks built at runtime by
			// Expand+Trilu chains (4 MB at T=1024) — baking those unblocks
			// fuse-sdpa — while still refusing unbounded broadcast blow-ups.
			lim = 16 << 20
		}
		if !foldable || outBytes > lim {
			continue
		}
		for i, v := range n.Outputs {
			if v == nil {
				continue
			}
			v.Producer = nil
			if len(v.Consumers) == 0 && !g.isOutput(v) {
				delete(g.Values, v.Name)
				continue
			}
			v.Const = outs[i]
			v.DType = outs[i].DType()
			v.Shape = outs[i].Shape()
			v.HasShape = true
		}
		for _, v := range n.Inputs {
			if v == nil {
				continue
			}
			removeConsumer(v, n)
			if v.Const != nil && len(v.Consumers) == 0 && !g.isOutput(v) {
				delete(g.Values, v.Name)
			}
		}
		dead[n] = true
		stats["fold-const"]++
	}
	g.compact(dead)
	return len(dead) > 0
}

// fuseSE fuses the squeeze-excite island — GlobalAveragePool → 1x1 Conv
// (+act) → 1x1 Conv (+act) → Mul back onto the pooled input — into one
// ingot.SE node. The island is four dispatches and three tiny intermediates
// for ~2C·Cr FLOPs of real work; fused it is two parallel regions around an
// inline FC chain.
func fuseSE(g *Graph, stats map[string]int) bool {
	conv1x1 := func(n *Node) bool {
		if n.OpType != "Conv" || n.Domain != "" || int(n.Attrs.Int("group", 1)) != 1 {
			return false
		}
		if n.Attrs.String("auto_pad", "NOTSET") != "NOTSET" {
			return false
		}
		for _, a := range n.Attrs.Ints("pads", nil) {
			if a != 0 {
				return false
			}
		}
		for _, a := range n.Attrs.Ints("strides", nil) {
			if a != 1 {
				return false
			}
		}
		for _, a := range n.Attrs.Ints("dilations", nil) {
			if a != 1 {
				return false
			}
		}
		if len(n.Inputs) < 2 || n.Inputs[1] == nil || n.Inputs[1].Const == nil || n.Inputs[1].Const.DType() != tensor.F32 {
			return false
		}
		ws := n.Inputs[1].Const.Shape()
		if len(ws) != 4 || ws[2] != 1 || ws[3] != 1 {
			return false
		}
		if len(n.Inputs) > 2 && n.Inputs[2] != nil && n.Inputs[2].Const == nil {
			return false // dynamic bias
		}
		return true
	}
	// isGAP: GlobalAveragePool, or its ReduceMean(axes=[2,3], keepdims=1)
	// spelling (efficientnet-style exports). Axes may be an attr or a const
	// input (opset 18).
	isGAP := func(n *Node) bool {
		if n.Domain != "" {
			return false
		}
		if n.OpType == "GlobalAveragePool" {
			return true
		}
		if n.OpType != "ReduceMean" || n.Attrs.Int("keepdims", 1) != 1 {
			return false
		}
		axes := n.Attrs.Ints("axes", nil)
		if axes == nil && len(n.Inputs) > 1 && n.Inputs[1] != nil && n.Inputs[1].Const != nil && n.Inputs[1].Const.DType() == tensor.I64 {
			axes = n.Inputs[1].Const.I64()
		}
		if len(axes) != 2 {
			return false
		}
		a0, a1 := axes[0], axes[1]
		if a0 < 0 {
			a0 += 4 // NCHW context: the op validates rank 4 at runtime
		}
		if a1 < 0 {
			a1 += 4
		}
		return (a0 == 2 && a1 == 3) || (a0 == 3 && a1 == 2)
	}
	dead := map[*Node]bool{}
	changed := false
	for _, gap := range g.Nodes {
		if dead[gap] || !isGAP(gap) {
			continue
		}
		x := gap.Inputs[0]
		c1 := g.soleConsumer(gap.Outputs[0])
		if c1 == nil || dead[c1] || !conv1x1(c1) {
			continue
		}
		c2 := g.soleConsumer(c1.Outputs[0])
		if c2 == nil || dead[c2] || !conv1x1(c2) {
			continue
		}
		w1s := c1.Inputs[1].Const.Shape()
		w2s := c2.Inputs[1].Const.Shape()
		if w2s[1] != w1s[0] || w2s[0] != w1s[1] {
			continue // not a squeeze-expand pair back to C channels
		}
		mul := g.soleConsumer(c2.Outputs[0])
		if mul == nil || dead[mul] || mul.OpType != "Mul" || mul.Domain != "" {
			continue
		}
		if !((mul.Inputs[0] == x && mul.Inputs[1] == c2.Outputs[0]) ||
			(mul.Inputs[1] == x && mul.Inputs[0] == c2.Outputs[0])) {
			continue
		}
		var b1, b2 *Value
		if len(c1.Inputs) > 2 {
			b1 = c1.Inputs[2]
		}
		if len(c2.Inputs) > 2 {
			b2 = c2.Inputs[2]
		}
		attrs := ops.Attrs{}
		copyEpi := func(src *Node, prefix string) {
			for _, name := range []string{"act", "act_alpha", "act_beta", "post_scale", "post_shift"} {
				if a, ok := src.Attrs["ingot_"+name]; ok {
					attrs[prefix+name] = a
				}
			}
		}
		copyEpi(c1, "se1_")
		copyEpi(c2, "se2_")
		out := mul.Outputs[0]
		// Attach surviving inputs to the host BEFORE dropping the pattern
		// nodes (drop-order landmine: a momentarily consumer-less const is
		// deleted and keeps a stale value id).
		host := gap
		ins := []*Value{x, c1.Inputs[1], b1, c2.Inputs[1], b2}
		for _, v := range ins[1:] {
			if v != nil {
				v.Consumers = append(v.Consumers, host)
			}
		}
		for _, n2 := range []*Node{c1, c2, mul} {
			dead[n2] = true
			g.dropNode(n2)
		}
		// A ReduceMean host may carry an axes const input: detach it.
		for _, v := range host.Inputs[1:] {
			if v != nil {
				removeConsumer(v, host)
				if v.Const != nil && len(v.Consumers) == 0 && !g.isOutput(v) {
					delete(g.Values, v.Name)
				}
			}
		}
		mid := gap.Outputs[0]
		delete(g.Values, mid.Name)
		g.Values[out.Name] = out
		host.OpType, host.Domain = "SE", ingotDomain
		host.Attrs = attrs
		host.Inputs = ins
		host.Outputs = []*Value{out}
		out.Producer = host
		g.Opsets[ingotDomain] = 1
		stats["fuse-se"]++
		changed = true
	}
	g.compact(dead)
	return changed
}

// causalMaskChain reports whether v is a runtime-built causal mask:
// Expand(-inf scalar, [T,T]) → Trilu(upper=1, k=1). Dynamic-shape exports
// build the mask per run; the decode form drops it (causality is the append
// position) and DCE removes the chain.
func causalMaskChain(v *Value) bool {
	tr := v.Producer
	if tr == nil || tr.OpType != "Trilu" || tr.Domain != "" || tr.Attrs.Int("upper", 1) != 1 {
		return false
	}
	if len(tr.Inputs) < 2 || tr.Inputs[1] == nil || tr.Inputs[1].Const == nil {
		return false
	}
	k := tr.Inputs[1].Const
	if k.DType() != tensor.I64 || k.Numel() != 1 || k.I64()[0] != 1 {
		return false
	}
	ex := tr.Inputs[0].Producer
	if ex == nil || ex.OpType != "Expand" || ex.Domain != "" {
		return false
	}
	sc := ex.Inputs[0]
	if sc == nil || sc.Const == nil || sc.Const.DType() != tensor.F32 || sc.Const.Numel() != 1 {
		return false
	}
	f := sc.Const.F32()[0]
	return math.IsInf(float64(f), -1)
}

// fuseSDPADecode rewrites the exporter attention core into the cached
// ingot.SDPA form for autoregressive decode: the runtime-built causal mask
// is dropped (the cached op is causal by construction), cache=1 is set, and
// no transpose peeling happens (the cached op wants [1,H,T,dh] inputs).
func fuseSDPADecode(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	changed := false
	for _, sm := range g.Nodes {
		if dead[sm] || sm.OpType != "Softmax" || sm.Domain != "" {
			continue
		}
		if ax := sm.Attrs.Int("axis", -1); ax != -1 && ax != 3 {
			continue
		}
		scale := 1.0
		cur := sm.Inputs[0]
		var mid []*Node
		matched := false
		for i := 0; i < 2; i++ {
			p := cur.Producer
			if p == nil || dead[p] || p.Domain != "" || g.soleConsumer(cur) == nil {
				break
			}
			if p.OpType == "Add" && len(p.Inputs) == 2 && !matched {
				var mv *Value
				var xv *Value
				switch {
				case p.Inputs[1] != nil && causalMaskChain(p.Inputs[1]):
					mv, xv = p.Inputs[1], p.Inputs[0]
				case p.Inputs[0] != nil && causalMaskChain(p.Inputs[0]):
					mv, xv = p.Inputs[0], p.Inputs[1]
				}
				if mv == nil {
					break
				}
				matched = true
				mid = append(mid, p)
				cur = xv
				continue
			}
			if x, c, right, ok := binaryWithConst(p); ok && (p.OpType == "Mul" || (p.OpType == "Div" && right && c != 0)) {
				if p.OpType == "Mul" {
					scale *= float64(c)
				} else {
					scale /= float64(c)
				}
				mid = append(mid, p)
				cur = x
				continue
			}
			break
		}
		if !matched {
			continue
		}
		mm1 := cur.Producer
		if mm1 == nil || dead[mm1] || mm1.OpType != "MatMul" || mm1.Domain != "" || g.soleConsumer(mm1.Outputs[0]) == nil {
			continue
		}
		mm2 := g.soleConsumer(sm.Outputs[0])
		if mm2 == nil || dead[mm2] || mm2.OpType != "MatMul" || mm2.Inputs[0] != sm.Outputs[0] {
			continue
		}
		// mm1's B input is Kᵀ: peel the transpose to reach K itself. The
		// legacy export transposes [1,H,T,dh] with [0,1,3,2] (k_layout 0);
		// the dynamo export builds Kᵀ straight from the [1,T,H,dh] view
		// with [0,2,3,1] (k_layout 1 — the append reads it strided).
		kv := mm1.Inputs[1]
		kLay := 0
		var ktr *Node
		if t := transProducer(g, kv, dead, 0, 1, 3, 2); t != nil {
			ktr = t
			kv = t.Inputs[0]
		} else if t := transProducer(g, kv, dead, 0, 2, 3, 1); t != nil {
			ktr = t
			kv = t.Inputs[0]
			kLay = 1
		} else {
			continue // cached form needs K reachable behind a transpose
		}
		av := mm1.Inputs[0]
		vv := mm2.Inputs[1]
		out := mm2.Outputs[0]
		drop := append(mid, sm, mm2, ktr)
		host := mm1
		ins := []*Value{av, kv, vv}
		for _, v := range ins {
			v.Consumers = append(v.Consumers, host)
		}
		oldIns := host.Inputs
		for _, n2 := range drop {
			dead[n2] = true
			g.dropNode(n2)
		}
		for _, v := range oldIns {
			if v != nil {
				removeConsumer(v, host)
			}
		}
		mid1 := host.Outputs[0]
		delete(g.Values, mid1.Name)
		g.Values[out.Name] = out
		host.OpType, host.Domain = "SDPA", ingotDomain
		host.Attrs = ops.Attrs{
			"scale":    fattr(float32(scale)),
			"cache":    {Kind: ops.KindInt, I: 1},
			"k_layout": {Kind: ops.KindInt, I: int64(kLay)},
		}
		host.Inputs = ins
		host.Outputs = []*Value{out}
		out.Producer = host
		g.Opsets[ingotDomain] = 1
		stats["fuse-sdpa-cache"]++
		changed = true
	}
	g.compact(dead)
	return changed
}

// dce removes nodes whose outputs are entirely unconsumed and not graph
// outputs (the decode rewrite orphans the mask-construction chain).
func dce(g *Graph, stats map[string]int) {
	for {
		dead := map[*Node]bool{}
		for _, n := range g.Nodes {
			live := false
			for _, v := range n.Outputs {
				if v != nil && (len(v.Consumers) > 0 || g.isOutput(v)) {
					live = true
					break
				}
			}
			if !live {
				dead[n] = true
				g.dropNode(n)
				stats["dce"]++
			}
		}
		if len(dead) == 0 {
			return
		}
		g.compact(dead)
	}
}
