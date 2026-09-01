package graph

import (
	"fmt"

	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/ops"
	"github.com/giraffesyo/ingot/tensor"
)

// FromONNX converts a decoded ONNX model into a Graph. Initializers become
// constant Values; Constant nodes are folded into constants. Nodes are kept
// in the model's (topological) order after validation.
func FromONNX(m *onnx.Model) (*Graph, error) {
	if m.Graph == nil {
		return nil, fmt.Errorf("graph: model has no graph")
	}
	opsets := map[string]int{}
	for _, o := range m.OpsetImport {
		d := o.Domain
		if d == "ai.onnx" {
			d = ""
		}
		opsets[d] = int(o.Version)
	}
	if _, ok := opsets[""]; !ok {
		opsets[""] = 9 // conservative default if the model omits it
	}
	return buildGraph(m.Graph, opsets, nil)
}

// buildScope chains value scopes for subgraph builds (If/Loop bodies read
// outer values by name). lookup materialises capture leaves in every
// intermediate scope, so nested subgraphs capture transitively.
type buildScope struct {
	g      *Graph
	parent *buildScope
}

func (s *buildScope) lookup(name string) *Value {
	if s == nil {
		return nil
	}
	if v, ok := s.g.Values[name]; ok {
		return v
	}
	if s.parent.lookup(name) == nil {
		return nil
	}
	v := &Value{Name: name, id: len(s.g.Values)}
	s.g.Values[name] = v
	s.g.Captures = append(s.g.Captures, name)
	s.g.Inputs = append(s.g.Inputs, v)
	return v
}

// buildGraph converts one (sub)graph. outer is nil for the top level.
func buildGraph(gp *onnx.Graph, opsets map[string]int, outer *buildScope) (*Graph, error) {
	g := &Graph{
		Name:   gp.Name,
		Values: map[string]*Value{},
		Opsets: map[string]int{},
	}
	for d, v := range opsets {
		g.Opsets[d] = v
	}
	sc := &buildScope{g: g, parent: outer}
	// Names defined inside this graph shadow the outer scope for the whole
	// subgraph: only names never defined here may be captured.
	internal := map[string]bool{}
	for _, t := range gp.Initializer {
		internal[t.Name] = true
	}
	for _, vi := range gp.Input {
		internal[vi.Name] = true
	}
	for _, pn := range gp.Nodes {
		for _, o := range pn.Output {
			internal[o] = true
		}
	}

	value := func(name string) *Value {
		if v, ok := g.Values[name]; ok {
			return v
		}
		v := &Value{Name: name, id: len(g.Values)}
		g.Values[name] = v
		return v
	}

	// Initializers → constants.
	for _, t := range gp.Initializer {
		tt, err := TensorFromONNX(t)
		if err != nil {
			return nil, fmt.Errorf("graph: initializer %q: %w", t.Name, err)
		}
		v := value(t.Name)
		v.Const = tt
		v.DType = tt.DType()
		v.Shape = tt.Shape()
		v.HasShape = true
	}
	// Inputs (skip those that are initializers — older exporters list both).
	for _, vi := range gp.Input {
		v := value(vi.Name)
		applyValueInfo(v, vi)
		if v.Const == nil {
			g.Inputs = append(g.Inputs, v)
		}
	}
	for _, vi := range gp.ValueInfo {
		applyValueInfo(value(vi.Name), vi)
	}
	for _, vi := range gp.Output {
		v := value(vi.Name)
		applyValueInfo(v, vi)
		g.Outputs = append(g.Outputs, v)
	}

	// Nodes.
	for i, pn := range gp.Nodes {
		attrs, subAttrs, err := convertAttrs(pn)
		if err != nil {
			return nil, fmt.Errorf("graph: node %d %s(%s): %w", i, pn.OpType, pn.Name, err)
		}
		n := &Node{Name: pn.Name, OpType: pn.OpType, Domain: pn.Domain, Attrs: attrs, id: i}
		if n.Domain == "ai.onnx" {
			n.Domain = ""
		}
		if n.Name == "" {
			n.Name = fmt.Sprintf("%s_%d", pn.OpType, i)
		}
		// Fold Constant nodes.
		if n.OpType == "Constant" && n.Domain == "" && len(pn.Output) == 1 {
			b, err := ops.Lookup("", "Constant", g.OpsetVersion(""))
			if err != nil {
				return nil, err
			}
			op, err := b(ops.NodeInfo{Name: n.Name, OpType: "Constant", Attrs: attrs, NumOut: 1})
			if err != nil {
				return nil, fmt.Errorf("graph: %w", err)
			}
			outs, err := op.Run(&ops.Ctx{}, nil)
			if err != nil {
				return nil, fmt.Errorf("graph: folding %s: %w", n, err)
			}
			v := value(pn.Output[0])
			v.Const = outs[0]
			v.DType = outs[0].DType()
			v.Shape = outs[0].Shape()
			v.HasShape = true
			continue
		}
		for _, name := range pn.Input {
			if name == "" {
				n.Inputs = append(n.Inputs, nil)
				continue
			}
			v := g.Values[name]
			if v == nil && !internal[name] {
				v = sc.lookup(name) // outer-scope read: becomes a capture
			}
			if v == nil {
				v = value(name)
			}
			v.Consumers = append(v.Consumers, n)
			n.Inputs = append(n.Inputs, v)
		}
		for _, name := range pn.Output {
			if name == "" {
				n.Outputs = append(n.Outputs, nil)
				continue
			}
			v := value(name)
			if v.Producer != nil {
				return nil, fmt.Errorf("graph: value %q produced by both %s and %s", name, v.Producer, n)
			}
			v.Producer = n
			n.Outputs = append(n.Outputs, v)
		}
		// Subgraph attributes (If then/else_branch, Loop body): build each
		// with this graph as the parent scope, then wire every captured name
		// as an extra data input of the node (deduplicated; Caps names them,
		// in the same order as the appended inputs).
		for _, sa := range subAttrs {
			sub, err := buildGraph(sa.G, opsets, sc)
			if err != nil {
				return nil, fmt.Errorf("graph: node %s attribute %q: %w", n, sa.Name, err)
			}
			if n.Sub == nil {
				n.Sub = map[string]*Graph{}
			}
			n.Sub[sa.Name] = sub
			for _, cap := range sub.Captures {
				already := false
				for _, c := range n.Caps {
					if c == cap {
						already = true
						break
					}
				}
				if already {
					continue
				}
				v := sc.lookup(cap)
				if v == nil {
					return nil, fmt.Errorf("graph: node %s: subgraph %q captures undefined value %q", n, sa.Name, cap)
				}
				v.Consumers = append(v.Consumers, n)
				n.Inputs = append(n.Inputs, v)
				n.Caps = append(n.Caps, cap)
			}
		}
		g.Nodes = append(g.Nodes, n)
	}

	// Validate: every non-constant, non-input value consumed must be produced
	// earlier (ONNX requires topological order; we enforce rather than sort so
	// errors point at the model).
	seen := map[*Value]bool{}
	for _, v := range g.Inputs {
		seen[v] = true
	}
	for _, n := range g.Nodes {
		for _, in := range n.Inputs {
			if in == nil || in.Const != nil {
				continue
			}
			if !seen[in] {
				if in.Producer == nil {
					return nil, fmt.Errorf("graph: %s reads %q which is never produced and is not an input", n, in.Name)
				}
				return nil, fmt.Errorf("graph: %s reads %q before it is produced by %s (graph not topologically sorted)", n, in.Name, in.Producer)
			}
		}
		for _, out := range n.Outputs {
			if out != nil {
				seen[out] = true
			}
		}
	}
	for _, v := range g.Outputs {
		if !seen[v] && v.Const == nil {
			return nil, fmt.Errorf("graph: output %q is never produced", v.Name)
		}
	}
	return g, nil
}

func applyValueInfo(v *Value, vi *onnx.ValueInfo) {
	if dt, ok := dtypeFromONNX(vi.ElemType); ok && v.DType == tensor.Invalid {
		v.DType = dt
	}
	if vi.HasShape && !v.HasShape {
		v.Shape = make([]int, len(vi.Shape))
		for i, d := range vi.Shape {
			v.Shape[i] = int(d)
		}
		v.HasShape = true
	}
}

func dtypeFromONNX(d onnx.DataType) (tensor.DType, bool) {
	switch d {
	case onnx.Float, onnx.Float16, onnx.BFloat16, onnx.Double:
		return tensor.F32, true
	case onnx.Int64:
		return tensor.I64, true
	case onnx.Int32:
		return tensor.I32, true
	case onnx.Bool:
		return tensor.Bool, true
	case onnx.Uint8:
		return tensor.U8, true
	case onnx.Int8:
		return tensor.I8, true
	}
	return tensor.Invalid, false
}

// TensorFromONNX materialises an ONNX TensorProto as a runtime tensor.
// Float16/BFloat16/Double are converted to F32; integer types narrower than
// 64 bits keep their width except uint16/int16/uint32/uint64 which widen to int64.
func TensorFromONNX(t *onnx.Tensor) (*tensor.Tensor, error) {
	if t.DataLocation == 1 {
		return nil, fmt.Errorf("external data not supported yet (%v)", t.ExternalData)
	}
	shape := make([]int, len(t.Dims))
	for i, d := range t.Dims {
		shape[i] = int(d)
	}
	switch t.DataType {
	case onnx.Float, onnx.Float16, onnx.BFloat16, onnx.Double:
		f, err := t.Float32s()
		if err != nil {
			return nil, err
		}
		return tensor.FromF32(f, shape...), nil
	case onnx.Int64, onnx.Uint16, onnx.Int16, onnx.Uint32, onnx.Uint64:
		v, err := t.Int64s()
		if err != nil {
			return nil, err
		}
		return tensor.FromI64(v, shape...), nil
	case onnx.Int32:
		v, err := t.Int64s()
		if err != nil {
			return nil, err
		}
		out := tensor.New(tensor.I32, shape...)
		d := out.I32()
		for i, x := range v {
			d[i] = int32(x)
		}
		return out, nil
	case onnx.Bool:
		v, err := t.Int64s()
		if err != nil {
			return nil, err
		}
		out := tensor.New(tensor.Bool, shape...)
		d := out.Bool()
		for i, x := range v {
			d[i] = x != 0
		}
		return out, nil
	case onnx.Uint8:
		v, err := t.Int64s()
		if err != nil {
			return nil, err
		}
		out := tensor.New(tensor.U8, shape...)
		d := out.U8()
		for i, x := range v {
			d[i] = uint8(x)
		}
		return out, nil
	case onnx.Int8:
		v, err := t.Int64s()
		if err != nil {
			return nil, err
		}
		out := tensor.New(tensor.I8, shape...)
		d := out.I8()
		for i, x := range v {
			d[i] = int8(x)
		}
		return out, nil
	}
	return nil, fmt.Errorf("unsupported tensor data type %s", t.DataType)
}

// convertAttrs converts scalar/tensor attributes; graph-typed attributes
// (If/Loop subgraphs) are returned separately for the caller to build.
func convertAttrs(pn *onnx.Node) (ops.Attrs, []*onnx.Attribute, error) {
	if len(pn.Attribute) == 0 {
		return nil, nil, nil
	}
	var subs []*onnx.Attribute
	out := make(ops.Attrs, len(pn.Attribute))
	for _, a := range pn.Attribute {
		var v ops.Attr
		switch a.Type {
		case onnx.AttrFloat:
			v = ops.Attr{Kind: ops.KindFloat, F: a.F}
		case onnx.AttrInt:
			v = ops.Attr{Kind: ops.KindInt, I: a.I}
		case onnx.AttrString:
			v = ops.Attr{Kind: ops.KindString, S: string(a.S)}
		case onnx.AttrTensor:
			t, err := TensorFromONNX(a.T)
			if err != nil {
				return nil, nil, fmt.Errorf("attribute %q: %w", a.Name, err)
			}
			v = ops.Attr{Kind: ops.KindTensor, T: t}
		case onnx.AttrFloats:
			v = ops.Attr{Kind: ops.KindFloats, Floats: a.Floats}
		case onnx.AttrInts:
			v = ops.Attr{Kind: ops.KindInts, Ints: a.Ints}
		case onnx.AttrStrings:
			ss := make([]string, len(a.Strings))
			for i, s := range a.Strings {
				ss[i] = string(s)
			}
			v = ops.Attr{Kind: ops.KindStrings, Strings: ss}
		case onnx.AttrGraph:
			subs = append(subs, a)
			continue
		case onnx.AttrGraphs:
			return nil, nil, fmt.Errorf("attribute %q: GRAPHS attributes (Scan) not supported", a.Name)
		default:
			// Older exporters omit Type; infer from populated fields.
			switch {
			case len(a.Ints) > 0:
				v = ops.Attr{Kind: ops.KindInts, Ints: a.Ints}
			case len(a.Floats) > 0:
				v = ops.Attr{Kind: ops.KindFloats, Floats: a.Floats}
			case a.T != nil:
				t, err := TensorFromONNX(a.T)
				if err != nil {
					return nil, nil, err
				}
				v = ops.Attr{Kind: ops.KindTensor, T: t}
			case a.S != nil:
				v = ops.Attr{Kind: ops.KindString, S: string(a.S)}
			default:
				return nil, nil, fmt.Errorf("attribute %q: unknown type %d", a.Name, a.Type)
			}
		}
		out[a.Name] = v
	}
	return out, subs, nil
}
