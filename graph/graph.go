// Package graph is the runtime's intermediate representation and executor.
//
// A Graph is a DAG of Nodes over named Values. Constants (initializers and
// folded Constant nodes) carry their tensor. Session compiles a Graph against
// the ops registry — failing loudly on any unsupported op — and runs it with
// pooled, refcounted intermediate buffers.
package graph

import (
	"fmt"

	"github.com/giraffesyo/ingot/ops"
	"github.com/giraffesyo/ingot/tensor"
)

// Value is a named tensor edge.
type Value struct {
	Name  string
	Const *tensor.Tensor // non-nil for constants
	// Static type info from the model (may be partial). Shape -1 = unknown dim.
	DType    tensor.DType
	Shape    []int
	HasShape bool

	Producer  *Node
	Consumers []*Node
	id        int
}

func (v *Value) String() string {
	if v.Const != nil {
		return fmt.Sprintf("%s=const%v", v.Name, v.Const.Shape())
	}
	return v.Name
}

// Node is an operator application.
type Node struct {
	Name    string
	OpType  string
	Domain  string
	Inputs  []*Value // nil for omitted optional inputs
	Outputs []*Value // nil for omitted optional outputs
	Attrs   ops.Attrs
	// Sub holds subgraph attributes (If then/else_branch, Loop body). Values
	// a subgraph captures from this scope are appended to Inputs after the
	// ONNX-declared ones, named in Caps (same order) — explicit data edges,
	// so the optimizer keeps captured values alive.
	Sub  map[string]*Graph
	Caps []string
	id   int
}

func (n *Node) String() string {
	return fmt.Sprintf("%s(%s)", n.OpType, n.Name)
}

// Graph is a topologically ordered DAG.
type Graph struct {
	Name    string
	Nodes   []*Node
	Inputs  []*Value // graph inputs (excluding initializers)
	Outputs []*Value
	Values  map[string]*Value
	Opsets  map[string]int // domain → version
	// Captures names outer-scope values this (sub)graph reads; each has a
	// producer-less leaf Value here that is also appended to Inputs, so a
	// sub-Session is fed captures by name like ordinary inputs.
	Captures []string
}

// OpsetVersion returns the opset version for a domain (default domain if
// unknown, 0 if absent).
func (g *Graph) OpsetVersion(domain string) int {
	if domain == "ai.onnx" {
		domain = ""
	}
	return g.Opsets[domain]
}

// Summary returns op-type counts, useful for diagnostics.
func (g *Graph) Summary() map[string]int {
	m := map[string]int{}
	for _, n := range g.Nodes {
		m[n.OpType]++
	}
	return m
}
