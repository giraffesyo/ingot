package graph

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

func randTensor(r *rand.Rand, shape ...int) *tensor.Tensor {
	t := tensor.New(tensor.F32, shape...)
	for i := range t.F32() {
		t.F32()[i] = r.Float32()*2 - 1
	}
	return t
}

// TestBuilderMLP builds Linear → LayerNorm → SiLU → RMSNorm → GeluTanh and
// checks Compile and CompileRaw against a float64 oracle.
func TestBuilderMLP(t *testing.T) {
	const M, In, Out = 5, 12, 8
	r := rand.New(rand.NewPCG(3, 4))
	w, bias := randTensor(r, Out, In), randTensor(r, Out)
	lnS, lnB, rmsW := randTensor(r, Out), randTensor(r, Out), randTensor(r, Out)
	x := randTensor(r, M, In)

	build := func() *Graph {
		b := NewBuilder("mlp")
		in := b.Input("x", tensor.F32, -1, In)
		blk := b.Scope("block")
		h := blk.Linear(in, w, bias)
		h = blk.LayerNorm(h, Out, lnS, lnB, 1e-5)
		h = blk.SiLU(h)
		h = blk.RMSNorm(h, rmsW, 1e-6)
		b.Output("y", blk.GeluTanh(h))
		g, err := b.Build()
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasPrefix(g.Nodes[0].Name, "block.") {
			t.Fatalf("scoped node name %q", g.Nodes[0].Name)
		}
		return g
	}

	want := make([]float64, M*Out)
	for i := range M {
		h := make([]float64, Out)
		for o := range Out {
			s := float64(bias.F32()[o])
			for k := range In {
				s += float64(x.F32()[i*In+k]) * float64(w.F32()[o*In+k])
			}
			h[o] = s
		}
		var mean, vs float64
		for _, v := range h {
			mean += v
		}
		mean /= Out
		for _, v := range h {
			vs += (v - mean) * (v - mean)
		}
		inv := 1 / math.Sqrt(vs/Out+1e-5)
		var ms float64
		for o := range h {
			v := (h[o]-mean)*inv*float64(lnS.F32()[o]) + float64(lnB.F32()[o])
			v = v / (1 + math.Exp(-v))
			h[o] = v
			ms += v * v
		}
		rinv := 1 / math.Sqrt(ms/Out+1e-6)
		for o := range h {
			v := h[o] * rinv * float64(rmsW.F32()[o])
			want[i*Out+o] = 0.5 * v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*v*v*v)))
		}
	}

	for _, compile := range []struct {
		name string
		fn   func(*Graph) (*Session, error)
	}{{"Compile", Compile}, {"CompileRaw", CompileRaw}} {
		s, err := compile.fn(build())
		if err != nil {
			t.Fatalf("%s: %v", compile.name, err)
		}
		out, err := s.Run(map[string]*tensor.Tensor{"x": x})
		if err != nil {
			t.Fatalf("%s: %v", compile.name, err)
		}
		got := out["y"]
		if !got.Shape().Equal([]int{M, Out}) {
			t.Fatalf("%s: shape %v", compile.name, got.Shape())
		}
		for i, v := range want {
			if d := math.Abs(float64(got.F32()[i]) - v); d > 1e-5*(1+math.Abs(v)) {
				t.Fatalf("%s: y[%d] = %g, want %g", compile.name, i, got.F32()[i], v)
			}
		}
	}
}

func TestBuilderErrors(t *testing.T) {
	b := NewBuilder("bad")
	x := b.Input("x", tensor.F32, 2)
	b.Output("y", b.Op("NoSuchOp", nil, b.Op("ingot.AlsoMissing", nil, x)))
	_, err := b.Build()
	if err == nil || !strings.Contains(err.Error(), ".NoSuchOp") || !strings.Contains(err.Error(), "ingot.AlsoMissing") {
		t.Fatalf("want both missing ops listed, got %v", err)
	}
	if _, err := NewBuilder("empty").Build(); err == nil {
		t.Fatal("no outputs: want error")
	}
	other := NewBuilder("other").Input("z", tensor.F32, 1)
	defer func() {
		if recover() == nil {
			t.Fatal("value from another graph: want panic")
		}
	}()
	b.Add(x, other)
}

// TestBuilderMultiOutput runs a 2-output Split and checks fresh naming keeps
// repeated scopes distinct.
func TestBuilderMultiOutput(t *testing.T) {
	b := NewBuilder("split")
	x := b.Input("x", tensor.F32, 4)
	parts := b.OpN("Split", Attr("axis", 0, "num_outputs", 2), 2, x)
	s1, s2 := b.Scope("s"), b.Scope("s")
	b.Output("a", s1.Mul(parts[0], s1.Scalar(2)))
	b.Output("b", s2.Mul(parts[1], s2.Scalar(3)))
	g, err := b.Build()
	if err != nil {
		t.Fatal(err)
	}
	s, err := Compile(g)
	if err != nil {
		t.Fatal(err)
	}
	out, err := s.Run(map[string]*tensor.Tensor{"x": tensor.FromF32([]float32{1, 2, 3, 4}, 4)})
	if err != nil {
		t.Fatal(err)
	}
	if a, bb := out["a"].F32(), out["b"].F32(); a[0] != 2 || a[1] != 4 || bb[0] != 9 || bb[1] != 12 {
		t.Fatalf("a=%v b=%v", a, bb)
	}
}
