package graph

import (
	"fmt"

	"github.com/giraffesyo/ingot/tensor"
)

// SetInputShape declares a concrete shape for a graph input before Compile.
// Layout passes use shapes only as placement heuristics: a session compiled
// for one declared shape stays correct for any runtime shape (ops read the
// actual tensor shapes), it is merely tuned for the declared one. Dynamic
// models (PP-OCR det/rec) get blocked-layout regions this way.
func (g *Graph) SetInputShape(name string, dims ...int) error {
	v := g.Values[name]
	if v == nil || v.Producer != nil || v.Const != nil {
		return fmt.Errorf("graph: %q is not a graph input", name)
	}
	for _, d := range dims {
		if d <= 0 {
			return fmt.Errorf("graph: SetInputShape %q: dims must be positive, got %v", name, dims)
		}
	}
	v.Shape = append([]int(nil), dims...)
	v.HasShape = true
	return nil
}

// propagateShapes fills concrete value shapes forward from inputs and
// constants through the ops the layout passes care about (conv backbones:
// convs, pools, elementwise, fused activations, SE, concat, resize).
// Best-effort: it stops at ops it does not model and never overwrites an
// already-concrete shape; symbolic (-1) exporter annotations are replaced
// where a concrete shape is derivable. Exporters that emit no value_info at
// all (torchvision mv3) are covered the same way.
func propagateShapes(g *Graph) {
	conc := func(v *Value) []int {
		if v == nil {
			return nil
		}
		if v.Const != nil {
			return v.Const.Shape()
		}
		if !v.HasShape {
			return nil
		}
		for _, d := range v.Shape {
			if d <= 0 {
				return nil
			}
		}
		return v.Shape
	}
	set := func(v *Value, dims []int) {
		if v != nil && conc(v) == nil {
			v.Shape, v.HasShape = dims, true
		}
	}
	ints := func(a []int64) []int {
		out := make([]int, len(a))
		for i, x := range a {
			out[i] = int(x)
		}
		return out
	}
	for _, n := range g.Nodes {
		if len(n.Outputs) == 0 || n.Outputs[0] == nil || n.Outputs[0].Const != nil {
			continue
		}
		out := n.Outputs[0]
		key := n.OpType
		if n.Domain != "" && n.Domain != ingotDomain {
			continue
		}
		switch key {
		case "Conv":
			x := conc(n.Inputs[0])
			if x == nil || len(x) != 4 || len(n.Inputs) < 2 || n.Inputs[1] == nil || n.Inputs[1].Const == nil {
				continue
			}
			ws := n.Inputs[1].Const.Shape()
			if len(ws) != 4 || n.Attrs.String("auto_pad", "NOTSET") != "NOTSET" {
				continue
			}
			st := ints(n.Attrs.Ints("strides", []int64{1, 1}))
			di := ints(n.Attrs.Ints("dilations", []int64{1, 1}))
			pa := ints(n.Attrs.Ints("pads", []int64{0, 0, 0, 0}))
			if len(st) != 2 || len(di) != 2 || len(pa) != 4 {
				continue
			}
			oh := (x[2] + pa[0] + pa[2] - (di[0]*(ws[2]-1) + 1)) / st[0]
			ow := (x[3] + pa[1] + pa[3] - (di[1]*(ws[3]-1) + 1)) / st[1]
			if oh < 0 || ow < 0 {
				continue
			}
			set(out, []int{x[0], ws[0], oh + 1, ow + 1})
		case "MaxPool", "AveragePool":
			x := conc(n.Inputs[0])
			ks := ints(n.Attrs.Ints("kernel_shape", nil))
			if x == nil || len(x) != 4 || len(ks) != 2 || n.Attrs.String("auto_pad", "NOTSET") != "NOTSET" {
				continue
			}
			st := ints(n.Attrs.Ints("strides", []int64{1, 1}))
			pa := ints(n.Attrs.Ints("pads", []int64{0, 0, 0, 0}))
			di := ints(n.Attrs.Ints("dilations", []int64{1, 1}))
			if len(st) != 2 || len(pa) != 4 || len(di) != 2 {
				continue
			}
			num := func(i, k int) int {
				v := x[2+i] + pa[i] + pa[i+2] - (di[i]*(k-1) + 1)
				if n.Attrs.Int("ceil_mode", 0) != 0 {
					return (v + st[i] - 1) / st[i]
				}
				return v / st[i]
			}
			oh, ow := num(0, ks[0]), num(1, ks[1])
			if oh < 0 || ow < 0 {
				continue
			}
			set(out, []int{x[0], x[1], oh + 1, ow + 1})
		case "GlobalAveragePool":
			if x := conc(n.Inputs[0]); len(x) == 4 {
				set(out, []int{x[0], x[1], 1, 1})
			}
		case "Add", "Sub", "Mul", "Div", "Min", "Max", "PRelu":
			a, b := conc(n.Inputs[0]), conc(n.Inputs[1])
			if a == nil || b == nil {
				continue
			}
			if bs, err := broadcastDims(a, b); err == nil {
				set(out, bs)
			}
		case "Relu", "LeakyRelu", "Sigmoid", "Tanh", "Clip", "HardSigmoid", "Erf",
			"Sqrt", "Exp", "Neg", "Abs", "Identity", "Dropout", "Softmax", "Elu",
			"HardSwish", "SiLU", "Gelu", "SE", "LayerNorm", "QLut", "BatchNormalization":
			if x := conc(n.Inputs[0]); x != nil {
				set(out, append([]int(nil), x...))
			}
		case "Concat":
			ax := int(n.Attrs.Int("axis", 0))
			var dims []int
			ok := true
			for _, in := range n.Inputs {
				x := conc(in)
				if x == nil {
					ok = false
					break
				}
				if dims == nil {
					dims = append([]int(nil), x...)
					if ax < 0 {
						ax += len(dims)
					}
					continue
				}
				if len(x) != len(dims) || ax < 0 || ax >= len(dims) {
					ok = false
					break
				}
				dims[ax] += x[ax]
			}
			if ok && dims != nil {
				set(out, dims)
			}
		case "Resize":
			// scales form only (det FPN upsample): out = floor(in * scale).
			x := conc(n.Inputs[0])
			if x == nil || len(n.Inputs) < 3 || n.Inputs[2] == nil || n.Inputs[2].Const == nil {
				continue
			}
			sc := n.Inputs[2].Const
			if sc.DType() != tensor.F32 || sc.Numel() != len(x) {
				continue
			}
			sf := sc.F32()
			dims := make([]int, len(x))
			for i := range x {
				dims[i] = int(float64(x[i]) * float64(sf[i]))
			}
			ok := true
			for _, d := range dims {
				if d <= 0 {
					ok = false
				}
			}
			if ok {
				set(out, dims)
			}
		}
	}
}

// broadcastDims returns the ONNX/numpy broadcast of two concrete shapes.
func broadcastDims(a, b []int) ([]int, error) {
	n := max(len(a), len(b))
	out := make([]int, n)
	for i := 0; i < n; i++ {
		da, db := 1, 1
		if i >= n-len(a) {
			da = a[i-(n-len(a))]
		}
		if i >= n-len(b) {
			db = b[i-(n-len(b))]
		}
		switch {
		case da == db, db == 1:
			out[i] = da
		case da == 1:
			out[i] = db
		default:
			return nil, fmt.Errorf("broadcast mismatch %v vs %v", a, b)
		}
	}
	return out, nil
}
