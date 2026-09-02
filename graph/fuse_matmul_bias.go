package graph

import "github.com/giraffesyo/ingot/tensor"

// fuseMatMulBias folds MatMul(x, W) → Add(·, bias) into the MatMul when W is
// a constant 2-D weight and bias is a constant of N elements broadcast over
// rows (the exporter's nn.Linear form). The bias becomes the MatMul's third
// input; matmulOp seeds the packed-B GEMM's output with it, so the bias
// costs nothing (kernels/gemm.SgemmPackedBEpi). Residual adds and GELU stay
// separate: measured at the strip width the GEMM works in (16 columns),
// applying them per strip costs as much as their own pass.
func fuseMatMulBias(g *Graph, stats map[string]int) bool {
	dead := map[*Node]bool{}
	changed := false
	for _, mm := range g.Nodes {
		if dead[mm] || mm.OpType != "MatMul" || mm.Domain != "" || len(mm.Inputs) != 2 {
			continue
		}
		w := mm.Inputs[1]
		if w == nil || w.Const == nil || w.Const.DType() != tensor.F32 || w.Const.Shape().Rank() != 2 {
			continue
		}
		n := w.Const.Shape()[1]
		add := g.soleConsumer(mm.Outputs[0])
		if add == nil || dead[add] || add.OpType != "Add" || add.Domain != "" || len(add.Inputs) != 2 {
			continue
		}
		var bias *Value
		switch {
		case add.Inputs[0] == mm.Outputs[0] && add.Inputs[1] != nil && add.Inputs[1].Const != nil:
			bias = add.Inputs[1]
		case add.Inputs[1] == mm.Outputs[0] && add.Inputs[0] != nil && add.Inputs[0].Const != nil:
			bias = add.Inputs[0]
		default:
			continue
		}
		bs := bias.Const.Shape()
		if bias.Const.DType() != tensor.F32 || bias.Const.Numel() != n || bs.Rank() == 0 || bs[bs.Rank()-1] != n {
			continue // must broadcast as a trailing [N]
		}
		if bs.Rank() > mm.Outputs[0].rankHint() && mm.Outputs[0].HasShape {
			continue // a higher-rank bias would broadcast the output up
		}
		out := add.Outputs[0]
		// Attach the bias before dropping the Add (see fuse-sdpa on why).
		bias.Consumers = append(bias.Consumers, mm)
		removeConsumer(mm.Outputs[0], add)
		g.dropNode(add)
		g.Values[out.Name] = out
		delete(g.Values, mm.Outputs[0].Name)
		mm.Inputs = append(mm.Inputs, bias)
		mm.Outputs = []*Value{out}
		out.Producer = mm
		dead[add] = true
		stats["fuse-matmul-bias"]++
		changed = true
	}
	g.compact(dead)
	return changed
}

// rankHint is the value's static rank, or a large number when unknown.
func (v *Value) rankHint() int {
	if !v.HasShape {
		return 1 << 20
	}
	return len(v.Shape)
}
