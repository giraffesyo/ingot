package graph

import (
	"math"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// testGraph builds graphs node-by-node for optimizer unit tests.
type testGraph struct{ g *Graph }

func newTestGraph() *testGraph {
	return &testGraph{g: &Graph{Name: "t", Values: map[string]*Value{}, Opsets: map[string]int{"": 17}}}
}

func (tg *testGraph) val(name string) *Value {
	if v, ok := tg.g.Values[name]; ok {
		return v
	}
	v := &Value{Name: name, id: len(tg.g.Values)}
	tg.g.Values[name] = v
	return v
}

func (tg *testGraph) constF32(name string, shape []int, data ...float32) *Value {
	t := tensor.New(tensor.F32, shape...)
	copy(t.F32(), data)
	v := tg.val(name)
	v.Const, v.DType, v.Shape, v.HasShape = t, tensor.F32, t.Shape(), true
	return v
}

func (tg *testGraph) constI64(name string, shape []int, data ...int64) *Value {
	t := tensor.New(tensor.I64, shape...)
	copy(t.I64(), data)
	v := tg.val(name)
	v.Const, v.DType, v.Shape, v.HasShape = t, tensor.I64, t.Shape(), true
	return v
}

func (tg *testGraph) node(op string, ins []string, outs ...string) *Node {
	n := &Node{Name: op + "_" + outs[0], OpType: op, id: len(tg.g.Nodes)}
	for _, name := range ins {
		v := tg.val(name)
		v.Consumers = append(v.Consumers, n)
		n.Inputs = append(n.Inputs, v)
	}
	for _, name := range outs {
		v := tg.val(name)
		v.Producer = n
		n.Outputs = append(n.Outputs, v)
	}
	tg.g.Nodes = append(tg.g.Nodes, n)
	return n
}

func (tg *testGraph) finish(inputs, outputs []string) *Graph {
	for _, name := range inputs {
		v := tg.val(name)
		v.DType = tensor.F32
		tg.g.Inputs = append(tg.g.Inputs, v)
	}
	for _, name := range outputs {
		tg.g.Outputs = append(tg.g.Outputs, tg.val(name))
	}
	return tg.g
}

// TestFoldConstChain folds the torch-export scale idiom Pow → Reciprocal →
// Mul down to one scalar, leaving a single dynamic Mul.
func TestFoldConstChain(t *testing.T) {
	tg := newTestGraph()
	tg.constF32("c4", nil, 4)
	tg.constF32("c5", nil, 5)
	tg.constF32("c2", nil, 2)
	tg.node("Pow", []string{"c4", "c5"}, "p")
	tg.node("Reciprocal", []string{"p"}, "r")
	tg.node("Mul", []string{"r", "c2"}, "s")
	tg.node("Mul", []string{"x", "s"}, "y")
	g := tg.finish([]string{"x"}, []string{"y"})

	stats := Optimize(g)
	if stats["fold-const"] != 3 {
		t.Fatalf("fold-const = %d, want 3 (stats %v)", stats["fold-const"], stats)
	}
	if len(g.Nodes) != 1 || g.Nodes[0].OpType != "Mul" {
		t.Fatalf("nodes = %v, want [Mul]", g.Nodes)
	}
	sc := g.Nodes[0].Inputs[1]
	if sc.Const == nil || sc.Const.Numel() != 1 {
		t.Fatalf("scale input not folded to scalar const: %v", sc)
	}
	want := float32(1/math.Pow(4, 5)) * 2
	if got := sc.Const.F32()[0]; got != want {
		t.Fatalf("folded scale = %v, want %v", got, want)
	}
	for _, name := range []string{"p", "r", "c4", "c5", "c2"} {
		if g.Values[name] != nil {
			t.Errorf("value %q should have been deleted", name)
		}
	}
}

// TestFoldConstGraphOutput folds a node whose output is a graph output; the
// session must still serve (a clone of) it.
func TestFoldConstGraphOutput(t *testing.T) {
	tg := newTestGraph()
	tg.constF32("a", []int{2}, 1, 2)
	tg.constF32("b", []int{2}, 10, 20)
	tg.node("Add", []string{"a", "b"}, "y")
	tg.node("Relu", []string{"x"}, "z")
	g := tg.finish([]string{"x"}, []string{"y", "z"})

	stats := Optimize(g)
	if stats["fold-const"] != 1 {
		t.Fatalf("fold-const = %d, want 1 (stats %v)", stats["fold-const"], stats)
	}
	yv := g.Values["y"]
	if yv == nil || yv.Const == nil || yv.Producer != nil {
		t.Fatalf("output y not a producer-less const: %+v", yv)
	}
	s, err := CompileRaw(g)
	if err != nil {
		t.Fatal(err)
	}
	x := tensor.New(tensor.F32, 2)
	res, err := s.Run(map[string]*tensor.Tensor{"x": x})
	if err != nil {
		t.Fatal(err)
	}
	got := res["y"].F32()
	if got[0] != 11 || got[1] != 22 {
		t.Fatalf("y = %v, want [11 22]", got)
	}
	if &got[0] == &yv.Const.F32()[0] {
		t.Fatal("session returned the graph const itself, not a clone")
	}
}

// TestFoldConstSizeGuard refuses to bake a broadcast blow-up (Expand of a
// scalar to 2M elements = 8 MiB out of ~28 input bytes) into the model.
func TestFoldConstSizeGuard(t *testing.T) {
	tg := newTestGraph()
	tg.constF32("s", nil, 3)
	tg.constI64("shape", []int{3}, 1, 1024, 2048)
	tg.node("Expand", []string{"s", "shape"}, "e")
	tg.node("Add", []string{"x", "e"}, "y")
	g := tg.finish([]string{"x"}, []string{"y"})

	stats := Optimize(g)
	if stats["fold-const"] != 0 {
		t.Fatalf("fold-const = %d, want 0 (size guard; stats %v)", stats["fold-const"], stats)
	}
	if len(g.Nodes) != 2 {
		t.Fatalf("nodes = %v, want Expand+Add intact", g.Nodes)
	}
}

// TestFoldConstSharedInput keeps a const alive while another node consumes it.
func TestFoldConstSharedInput(t *testing.T) {
	tg := newTestGraph()
	tg.constF32("c", nil, 7)
	tg.node("Reciprocal", []string{"c"}, "r")
	tg.node("Mul", []string{"x", "r"}, "y")
	tg.node("Add", []string{"x2", "c"}, "z") // second consumer of c
	g := tg.finish([]string{"x", "x2"}, []string{"y", "z"})

	stats := Optimize(g)
	if stats["fold-const"] != 1 {
		t.Fatalf("fold-const = %d, want 1 (stats %v)", stats["fold-const"], stats)
	}
	if g.Values["c"] == nil || g.Values["c"].Const == nil {
		t.Fatal("shared const c must survive")
	}
	if g.Values["r"] == nil || g.Values["r"].Const == nil {
		t.Fatal("r should be a folded const")
	}
}
