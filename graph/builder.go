package graph

import (
	"fmt"
	"sort"
	"strings"

	"github.com/giraffesyo/ingot/ops"
	"github.com/giraffesyo/ingot/tensor"
)

// Builder constructs a Graph in Go — the path for models defined over
// safetensors weights rather than exported to ONNX. The result is an
// ordinary Graph: Compile optimizes and runs it like a loaded model.
//
// Nodes are appended in call order, so the graph is topologically sorted by
// construction. Names are scoped (Scope) so profiles read like the module
// tree: "blocks.3.attn.q".
type Builder struct {
	g      *Graph
	prefix string
	names  map[string]int // shared across scopes: auto-name counters
}

// Default opsets for built graphs: the newest standard semantics the ops
// registry implements, plus the runtime's own domain.
const (
	BuilderOpset      = 21
	BuilderIngotOpset = 1
)

// NewBuilder starts an empty graph.
func NewBuilder(name string) *Builder {
	return &Builder{
		g: &Graph{Name: name, Values: map[string]*Value{},
			Opsets: map[string]int{"": BuilderOpset, "ingot": BuilderIngotOpset}},
		names: map[string]int{},
	}
}

// Scope returns a builder that prefixes node and constant names with
// name + ".". It shares the graph with b.
func (b *Builder) Scope(name string) *Builder {
	c := *b
	c.prefix = b.prefix + name + "."
	return &c
}

// fresh returns an unused value/node name under the current scope.
func (b *Builder) fresh(base string) string {
	name := b.prefix + base
	if _, taken := b.g.Values[name]; !taken && b.names[name] == 0 {
		b.names[name] = 1
		return name
	}
	for {
		b.names[name]++
		n := fmt.Sprintf("%s_%d", name, b.names[name]-1)
		if _, taken := b.g.Values[n]; !taken {
			return n
		}
	}
}

func (b *Builder) value(name string) *Value {
	v := &Value{Name: name, id: len(b.g.Values)}
	b.g.Values[name] = v
	return v
}

// Input declares a graph input; -1 marks a dynamic dim.
func (b *Builder) Input(name string, dt tensor.DType, shape ...int) *Value {
	if _, taken := b.g.Values[name]; taken {
		panic(fmt.Sprintf("graph: builder: input %q already defined", name))
	}
	v := b.value(name)
	v.DType, v.Shape, v.HasShape = dt, append([]int(nil), shape...), true
	b.g.Inputs = append(b.g.Inputs, v)
	return v
}

// Const adds a constant (a weight, a shape, an axis list).
func (b *Builder) Const(name string, t *tensor.Tensor) *Value {
	v := b.value(b.fresh(name))
	v.Const, v.DType, v.Shape, v.HasShape = t, t.DType(), t.Shape(), true
	return v
}

// Scalar adds a rank-0 f32 constant.
func (b *Builder) Scalar(x float32) *Value { return b.Const("scalar", tensor.Scalar(x)) }

// Ints adds a 1-D int64 constant (shapes, axes, slice bounds).
func (b *Builder) Ints(xs ...int64) *Value {
	return b.Const("ints", tensor.FromI64(append([]int64(nil), xs...), len(xs)))
}

// Op appends a single-output node. opType may carry a domain prefix
// ("ingot.SDPA"); nil inputs are omitted optional inputs.
func (b *Builder) Op(opType string, attrs ops.Attrs, in ...*Value) *Value {
	return b.OpN(opType, attrs, 1, in...)[0]
}

// OpN appends a node with nOut outputs.
func (b *Builder) OpN(opType string, attrs ops.Attrs, nOut int, in ...*Value) []*Value {
	domain, op := "", opType
	if i := strings.LastIndexByte(opType, '.'); i >= 0 {
		domain, op = opType[:i], opType[i+1:]
	}
	n := &Node{Name: b.fresh(strings.ToLower(op)), OpType: op, Domain: domain, Attrs: attrs, id: len(b.g.Nodes)}
	for _, v := range in {
		if v != nil {
			if b.g.Values[v.Name] != v {
				panic(fmt.Sprintf("graph: builder: %s reads value %q from another graph", n.Name, v.Name))
			}
			v.Consumers = append(v.Consumers, n)
		}
		n.Inputs = append(n.Inputs, v)
	}
	outs := make([]*Value, nOut)
	for i := range outs {
		name := n.Name
		if nOut > 1 {
			name = fmt.Sprintf("%s:%d", n.Name, i)
		}
		v := b.value(name)
		v.Producer = n
		outs[i] = v
	}
	n.Outputs = outs
	b.g.Nodes = append(b.g.Nodes, n)
	return outs
}

// Output marks v as a graph output named name (renaming it).
func (b *Builder) Output(name string, v *Value) {
	if name != v.Name {
		if _, taken := b.g.Values[name]; taken {
			panic(fmt.Sprintf("graph: builder: output name %q already used", name))
		}
		delete(b.g.Values, v.Name)
		v.Name = name
		b.g.Values[name] = v
	}
	b.g.Outputs = append(b.g.Outputs, v)
}

// Build returns the graph, failing loudly — listing every one — if a node's
// op is not in the registry at the builder's opsets.
func (b *Builder) Build() (*Graph, error) {
	if len(b.g.Outputs) == 0 {
		return nil, fmt.Errorf("graph: builder %q: no outputs", b.g.Name)
	}
	var missing []string
	seen := map[string]bool{}
	for _, n := range b.g.Nodes {
		key := n.Domain + "." + n.OpType
		if seen[key] {
			continue
		}
		seen[key] = true
		if _, err := ops.Lookup(n.Domain, n.OpType, b.g.OpsetVersion(n.Domain)); err != nil {
			missing = append(missing, fmt.Sprintf("%s (e.g. node %q)", key, n.Name))
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return nil, fmt.Errorf("graph: builder %q: unsupported ops: %s", b.g.Name, strings.Join(missing, ", "))
	}
	return b.g, nil
}

// Attr builds an attribute map from name/value pairs: int, int64, float32,
// float64, string, []int64, []int, []float32.
//
//	graph.Attr("axis", -1, "epsilon", 1e-6)
func Attr(kv ...any) ops.Attrs {
	if len(kv)%2 != 0 {
		panic("graph: Attr: odd number of arguments")
	}
	a := make(ops.Attrs, len(kv)/2)
	for i := 0; i < len(kv); i += 2 {
		name, ok := kv[i].(string)
		if !ok {
			panic(fmt.Sprintf("graph: Attr: name %v is not a string", kv[i]))
		}
		switch v := kv[i+1].(type) {
		case int:
			a[name] = ops.Attr{Kind: ops.KindInt, I: int64(v)}
		case int64:
			a[name] = ops.Attr{Kind: ops.KindInt, I: v}
		case float32:
			a[name] = ops.Attr{Kind: ops.KindFloat, F: v}
		case float64:
			a[name] = ops.Attr{Kind: ops.KindFloat, F: float32(v)}
		case string:
			a[name] = ops.Attr{Kind: ops.KindString, S: v}
		case []int64:
			a[name] = ops.Attr{Kind: ops.KindInts, Ints: v}
		case []int:
			is := make([]int64, len(v))
			for j, x := range v {
				is[j] = int64(x)
			}
			a[name] = ops.Attr{Kind: ops.KindInts, Ints: is}
		case []float32:
			a[name] = ops.Attr{Kind: ops.KindFloats, Floats: v}
		default:
			panic(fmt.Sprintf("graph: Attr: %s: unsupported value type %T", name, v))
		}
	}
	return a
}
