package graph_test

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

// TestDecodeE2E: gptish_dyn through CompileDecode (prefill + token steps)
// must match the stateless full-sequence run position for position.
func TestDecodeE2E(t *testing.T) {
	path := filepath.Join(modelDir, "gptish_dyn.onnx")
	if _, err := os.Stat(path); err != nil {
		t.Skip("gptish_dyn not present")
	}
	const d, prefill, steps = 512, 24, 8
	total := prefill + steps

	load := func() *graph.Graph {
		m, err := onnx.DecodeFile(path)
		if err != nil {
			t.Fatal(err)
		}
		g, err := graph.FromONNX(m)
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	sFull, err := graph.Compile(load())
	if err != nil {
		t.Fatal(err)
	}
	gDec := load()
	sDec, err := graph.CompileDecode(gDec)
	if err != nil {
		t.Fatal(err)
	}
	if n := gDec.Summary()["SDPA"]; n != 4 {
		t.Fatalf("decode graph has %d SDPA nodes, want 4 (summary %v)", n, gDec.Summary())
	}

	x := make([]float32, total*d)
	for i := range x {
		x[i] = float32((i*2654435761)%1000)/500 - 1
	}
	feed := func(t0, tc int) map[string]*tensor.Tensor {
		xt := tensor.New(tensor.F32, 1, tc, d)
		copy(xt.F32(), x[t0*d:(t0+tc)*d])
		return map[string]*tensor.Tensor{"x": xt}
	}

	// Stateless reference over the full sequence.
	ref, err := sFull.Run(feed(0, total))
	if err != nil {
		t.Fatal(err)
	}
	rf := ref["out"].F32() // [1, total, d]

	check := func(of []float32, t0, tc int) {
		var maxd float64
		for i := 0; i < tc*d; i++ {
			if dd := math.Abs(float64(of[i] - rf[(t0)*d+i])); dd > maxd {
				maxd = dd
			}
		}
		if maxd > 5e-4 {
			t.Fatalf("positions %d..%d: max abs err %g", t0, t0+tc, maxd)
		}
	}

	dec := sDec.NewDecode(64)
	res, err := sDec.RunDecode(dec, feed(0, prefill), prefill)
	if err != nil {
		t.Fatal(err)
	}
	check(res["out"].F32(), 0, prefill)
	for s0 := prefill; s0 < total; s0++ {
		res, err := sDec.RunDecode(dec, feed(s0, 1), 1)
		if err != nil {
			t.Fatal(err)
		}
		check(res["out"].F32(), s0, 1)
	}
}
