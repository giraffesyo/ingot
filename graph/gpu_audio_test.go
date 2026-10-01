//go:build darwin && arm64

package graph_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/tensor"
)

// TestGPUAudioOps: the GPU executor's 1-D Conv (causal, dilated,
// depthwise), 1-D ConvTranspose (direct and the wide GEMM + col2im path),
// ingot.Snake and ingot.RoPE against the CPU session on one built graph —
// every node placed on the GPU, no mid-graph flushes.
func TestGPUAudioOps(t *testing.T) {
	if !metal.Available() {
		t.Skip("no Metal device")
	}
	r := rand.New(rand.NewPCG(61, 62))
	rnd := func(shape ...int) *tensor.Tensor {
		x := tensor.New(tensor.F32, shape...)
		for i := range x.F32() {
			x.F32()[i] = r.Float32()*2 - 1
		}
		return x
	}
	const C, T = 64, 37
	Tp := ((T-1)*2+4-1)*3 + 6
	// Constants made once: both sessions must see the same weights.
	W, B, DW, DB, F, S := rnd(C, C, 5), rnd(C), rnd(C, 1, 7), rnd(C), rnd(C), rnd(C)
	T1, T1B, UP, T2, T2B := rnd(C, 8, 4), rnd(8), rnd(256, 8, 1), rnd(256, 128, 6), rnd(128)
	COS, SIN := rnd(Tp, 16), rnd(Tp, 16)
	build := func() *graph.Graph {
		b := graph.NewBuilder("audio")
		x := b.Input("x", tensor.F32, 1, C, T)
		h := b.Op("Conv", graph.Attr("pads", []int{12, 0}, "dilations", []int{3}),
			x, b.Const("w", W), b.Const("b", B))
		h = b.Op("Conv", graph.Attr("pads", []int{6, 0}, "group", C), h, b.Const("dw", DW), b.Const("db", DB))
		h = b.Snake(h, F, S)
		// Narrow transposed conv (direct kernel), then a wide one (GEMM path).
		h = b.Op("ConvTranspose", graph.Attr("strides", []int{2}), h, b.Const("t1", T1), b.Const("t1b", T1B))
		h = b.Op("Conv", nil, h, b.Const("up", UP))
		h = b.Op("ConvTranspose", graph.Attr("strides", []int{3}), h, b.Const("t2", T2), b.Const("t2b", T2B))
		// RoPE over [T', heads, dh] rows.
		q := b.Reshape(b.Transpose(h, 0, 2, 1), int64(Tp), 4, 32)
		q = b.Op("ingot.RoPE", graph.Attr("layout", 1), q, b.Const("cos", COS), b.Const("sin", SIN))
		b.Output("y", q)
		b.Output("upsampled", h)
		g, err := b.Build()
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	cpu, err := graph.Compile(build())
	if err != nil {
		t.Fatal(err)
	}
	gpu, err := graph.CompileGPU(build())
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Close()
	x := rnd(1, C, T)
	want, err := cpu.Run(map[string]*tensor.Tensor{"x": x})
	if err != nil {
		t.Fatal(err)
	}
	got, err := gpu.Run(map[string]*tensor.Tensor{"x": x})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"upsampled", "y"} {
		w, g := want[name].F32(), got[name].F32()
		if !want[name].Shape().Equal(got[name].Shape()) {
			t.Fatalf("%s: shape %v vs CPU %v", name, got[name].Shape(), want[name].Shape())
		}
		var maxErr, scale float64
		for i := range w {
			maxErr = math.Max(maxErr, math.Abs(float64(g[i]-w[i])))
			scale = math.Max(scale, math.Abs(float64(w[i])))
		}
		if maxErr > 1e-4*scale {
			t.Errorf("%s: GPU vs CPU max err %.3g (scale %.3g)", name, maxErr, scale)
		}
	}
	if gpu.Flushes != 1 {
		t.Errorf("%d GPU flushes (CPU fallbacks: %v), want 1", gpu.Flushes, gpu.FlushedBy)
	}
	t.Logf("GPU steps %d, CPU steps %d", gpu.GPUSteps, gpu.CPUSteps)
}
