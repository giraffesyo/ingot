package graph

import "github.com/giraffesyo/ingot/ops"

// fuseAddLayerNorm folds a residual Add into the LayerNorm it feeds:
//
//	s = Add(a, b)            (a, b same shape, neither constant)
//	y = LayerNorm(s, γ, β)   (ONNX LayerNormalization or ingot.LayerNorm)
//
// becomes ingot.AddLayerNorm(a, b, γ, β) → (s, y). The sum keeps its value
// (it is the next block's residual, so other consumers simply re-point);
// the op writes it and normalises it in one pass per row. Same-shape is
// decided statically: equal ranks, equal known dims, and unknown dims only
// in the same positions (the op re-checks at runtime and errors loudly).
func fuseAddLayerNorm(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	changed := false
	for _, ln := range g.Nodes {
		if dead[ln] || len(ln.Inputs) < 1 || ln.Inputs[0] == nil {
			continue
		}
		if !(ln.OpType == "LayerNormalization" && ln.Domain == "") && !(ln.OpType == "LayerNorm" && ln.Domain == ingotDomain) {
			continue
		}
		if len(ln.Outputs) > 1 && ln.Outputs[1] != nil {
			continue // mean / inv-std outputs requested
		}
		add := ln.Inputs[0].Producer
		if add == nil || dead[add] || add.OpType != "Add" || add.Domain != "" || len(add.Inputs) != 2 {
			continue
		}
		a, b := add.Inputs[0], add.Inputs[1]
		if a == nil || b == nil || a.Const != nil || b.Const != nil || !sameShape(a, b) {
			continue
		}
		// Only rows long enough for the per-row work to amortise the extra
		// call: at D=48 (bertish) the fused form measured +7% on Zen 5, at
		// D=384 (PARSeq) −10%. The normalised size is the scale's length.
		if len(ln.Inputs) < 2 || ln.Inputs[1] == nil || ln.Inputs[1].Const == nil || ln.Inputs[1].Const.Numel() < fuseLNMinD {
			continue
		}
		s := add.Outputs[0] // keeps its identity (and graph-output status)
		// Re-point the sum's other consumers to the fused node's output 0
		// (the Value object survives; only its producer changes).
		host := ln
		ins := []*Value{a, b}
		for _, v := range ln.Inputs[1:] {
			ins = append(ins, v)
		}
		for _, v := range ins {
			if v != nil {
				v.Consumers = append(v.Consumers, host)
			}
		}
		oldIns := ln.Inputs
		removeConsumer(s, ln)
		// Detach the Add from its inputs without deleting s from the table.
		removeConsumer(a, add)
		removeConsumer(b, add)
		dead[add] = true
		for _, v := range oldIns {
			if v != nil {
				removeConsumer(v, host)
			}
		}
		s.Producer = host
		y := ln.Outputs[0]
		host.OpType, host.Domain = "AddLayerNorm", ingotDomain
		host.Attrs = ops.Attrs{
			"axis":    ln.Attrs["axis"],
			"epsilon": ln.Attrs["epsilon"],
		}
		if _, ok := host.Attrs["axis"]; !ok {
			host.Attrs["axis"] = ops.Attr{Kind: ops.KindInt, I: -1}
		}
		if _, ok := host.Attrs["epsilon"]; !ok {
			host.Attrs["epsilon"] = ops.Attr{Kind: ops.KindFloat, F: 1e-5}
		}
		host.Inputs = ins
		host.Outputs = []*Value{s, y}
		g.Opsets[ingotDomain] = 1
		stats["fuse-add-layernorm"]++
		changed = true
	}
	g.compact(dead)
	return changed
}

// fuseLNMinD is the smallest normalised size worth fusing (see above).
const fuseLNMinD = 128

// sameShape reports whether two values are statically known to have the
// same shape: equal rank, equal known dims, unknown dims in the same places.
func sameShape(a, b *Value) bool {
	if !a.HasShape || !b.HasShape || len(a.Shape) != len(b.Shape) {
		return false
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return false
		}
	}
	return true
}
