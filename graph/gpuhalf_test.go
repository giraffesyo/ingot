//go:build darwin && arm64

package graph_test

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/tensor"
)

func randTensor(r *rand.Rand, scale float32, shape ...int) *tensor.Tensor {
	t := tensor.New(tensor.F32, shape...)
	for i := range t.F32() {
		t.F32()[i] = (r.Float32()*2 - 1) * scale
	}
	return t
}

// attnBlock builds a transformer block in the shape GPUBF16's
// half-precision plan targets: a modulated LayerNorm feeding a qkv
// projection (Add → Gemm), L2-normalised and rotated queries and keys and
// a sliced value feeding token-major attention (RoPE, Mul, Slice →
// Reshape → SDPA), then an MLP (Gemm → Gelu → Gemm). extra adds a second,
// f32 reader of the attention query (the value must then stay f32).
func attnBlock(r *rand.Rand, T, H, dh int, extra bool) *graph.Graph {
	D := H * dh
	b := graph.NewBuilder("attn_block")
	x := b.Input("x", tensor.F32, T, D)
	kc := b.Input("kc", tensor.F32, 1, 5, H, dh) // a short cross-attention condition
	vc := b.Input("vc", tensor.F32, 1, 5, H, dh)
	lin := func(v *graph.Value, in, out int) *graph.Value {
		return b.Linear(v, randTensor(r, float32(1/math.Sqrt(float64(in))), out, in), randTensor(r, 0.1, out))
	}
	l2 := func(v *graph.Value) *graph.Value {
		n := b.Op("ReduceL2", graph.Attr("keepdims", 1), v, b.Ints(-1))
		return b.Mul(b.Div(v, b.Op("Max", nil, n, b.Scalar(1e-12))), b.Const("gamma", randTensor(r, 4, H, dh)))
	}
	attrs := graph.Attr("scale", float32(1/math.Sqrt(float64(dh))), "a_layout", 1, "b_layout", 1, "v_layout", 1, "stride_out", 1)
	cos, sin := tensor.New(tensor.F32, T, dh/2), tensor.New(tensor.F32, T, dh/2)
	for i := range cos.F32() {
		a := float64(i) * 0.37
		cos.F32()[i], sin.F32()[i] = float32(math.Cos(a)), float32(math.Sin(a))
	}
	cv, sv := b.Const("cos", cos), b.Const("sin", sin)
	heads := func(v *graph.Value) *graph.Value { return b.Reshape(v, int64(T), int64(H), int64(dh)) }
	batch := func(v *graph.Value) *graph.Value { return b.Reshape(v, 1, int64(T), int64(H), int64(dh)) }

	y := b.Add(b.Mul(b.LayerNorm(x, D, nil, nil, 1e-6), b.Const("scale", randTensor(r, 1, 1, D))), b.Const("shift", randTensor(r, 1, 1, D)))
	qkv := lin(y, D, 3*D)
	q := l2(heads(b.Slice(qkv, 1, 0, int64(D))))
	k := l2(heads(b.Slice(qkv, 1, int64(D), int64(2*D))))
	q = batch(b.Op("ingot.RoPE", graph.Attr("layout", 0), q, cv, sv))
	k = batch(b.Op("ingot.RoPE", graph.Attr("layout", 0), k, cv, sv))
	v := batch(b.Slice(qkv, 1, int64(2*D), int64(3*D)))
	o := b.Reshape(b.Op("ingot.SDPA", attrs, q, k, v), int64(T), int64(D))
	x = b.Add(x, lin(o, D, D))

	qc := batch(l2(heads(lin(b.LayerNorm(x, D, nil, nil, 1e-6), D, D))))
	o = b.Reshape(b.Op("ingot.SDPA", attrs, qc, kc, vc), int64(T), int64(D))
	x = b.Add(x, lin(o, D, D))
	if extra {
		x = b.Add(x, b.Reshape(qc, int64(T), int64(D)))
	}
	x = b.Add(x, lin(b.GeluTanh(lin(x, D, 2*D)), 2*D, D))
	b.Output("y", x)
	g, err := b.Build()
	if err != nil {
		panic(err)
	}
	return g
}

// TestGPUHalfActivations runs the block on the CPU and under GPUBF16 and
// compares: with head width 128 the attention is the fused bf16 kernel and
// every operand it and the bf16 products read is produced in bf16; with
// another width the attention declines, so its bf16-stored operands must
// be widened back for the fallback (FlushedBy records it); with a second
// reader of an operand the value stays f32. Staged inputs are read in
// place and give the same result run after run.
func TestGPUHalfActivations(t *testing.T) {
	if !metal.Available() {
		t.Skip("no Metal device")
	}
	for _, c := range []struct {
		name      string
		T, H, dh  int
		extra     bool
		wantWiden bool
	}{
		{"flash", 70, 2, 128, false, false},
		{"flash/shared-operand", 70, 2, 128, true, false},
		{"widen", 33, 3, 64, false, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			D := c.H * c.dh
			feeds := map[string]*tensor.Tensor{}
			r := rand.New(rand.NewPCG(5, 6))
			feeds["x"] = randTensor(r, 1, c.T, D)
			feeds["kc"], feeds["vc"] = randTensor(r, 2, 1, 5, c.H, c.dh), randTensor(r, 1, 1, 5, c.H, c.dh)
			build := func() *graph.Graph { return attnBlock(rand.New(rand.NewPCG(1, 2)), c.T, c.H, c.dh, c.extra) }
			cpu, err := graph.Compile(build())
			if err != nil {
				t.Fatal(err)
			}
			want, err := cpu.Run(feeds)
			if err != nil {
				t.Fatal(err)
			}
			s, err := graph.CompileGPU(build(), graph.GPUBF16())
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			staged := map[string]*tensor.Tensor{"x": feeds["x"], "kc": s.Stage(feeds["kc"]), "vc": s.Stage(feeds["vc"])}
			var first []float32
			for run := range 3 {
				got, err := s.Run(staged)
				if err != nil {
					t.Fatal(err)
				}
				gf, wf := got["y"].F32(), want["y"].F32()
				var maxd, maxw float64
				for i := range wf {
					maxd = math.Max(maxd, math.Abs(float64(gf[i]-wf[i])))
					maxw = math.Max(maxw, math.Abs(float64(wf[i])))
				}
				// bf16 operands: ~2⁻⁸ relative per product, a few products deep.
				if maxd > 0.03*maxw || math.IsNaN(maxd) {
					t.Fatalf("run %d: max |Δ| %.3g vs CPU (max |want| %.3g)", run, maxd, maxw)
				}
				if run == 0 {
					first = slices.Clone(gf)
					t.Logf("max |Δ| %.3g of %.3g; gpu %d / cpu %d steps, flushes %v", maxd, maxw, s.GPUSteps, s.CPUSteps, s.FlushedBy)
				} else if !slices.Equal(first, gf) {
					t.Fatalf("run %d differs from run 0", run)
				}
				if widened := slices.Contains(s.FlushedBy, "widen"); widened != c.wantWiden {
					t.Fatalf("widened = %v, want %v (flushes %v)", widened, c.wantWiden, s.FlushedBy)
				}
				s.Release(got)
			}
		})
	}
}

// TestGPUStageView: an output that is a view of a staged input comes back
// as a copy — releasing it must not hand the staged buffer to the pool.
func TestGPUStageView(t *testing.T) {
	if !metal.Available() {
		t.Skip("no Metal device")
	}
	b := graph.NewBuilder("view")
	x := b.Input("x", tensor.F32, 4, 6)
	b.Output("y", b.Reshape(x, 6, 4))
	b.Output("z", b.Add(x, b.Scalar(1)))
	g, err := b.Build()
	if err != nil {
		t.Fatal(err)
	}
	s, err := graph.CompileGPU(g)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	in := randTensor(rand.New(rand.NewPCG(3, 4)), 1, 4, 6)
	st := s.Stage(in)
	for run := range 3 {
		res, err := s.Run(map[string]*tensor.Tensor{"x": st})
		if err != nil {
			t.Fatal(err)
		}
		if res["y"].SharesBuffer(st) {
			t.Fatal("output aliases the staged input")
		}
		for i, v := range in.F32() {
			if res["y"].F32()[i] != v || res["z"].F32()[i] != v+1 || st.F32()[i] != v {
				t.Fatalf("run %d: [%d] y %g z %g staged %g, want %g", run, i, res["y"].F32()[i], res["z"].F32()[i], st.F32()[i], v)
			}
		}
		s.Release(res)
	}
}
