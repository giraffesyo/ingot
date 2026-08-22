package graph_test

import (
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

// TestSessionConcurrentRun runs one compiled Session from many goroutines at
// once (go test -race) and checks every result matches a serial baseline,
// pinning the documented guarantee that Session.Run is safe for concurrent use
// when Profile is off.
func TestSessionConcurrentRun(t *testing.T) {
	candidates := []string{
		"../testdata/ocr/det.onnx",
		"../testdata/models/mobilenet_v3_small.onnx",
		"../testdata/models/resnetish.onnx",
	}
	var path string
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			path = c
			break
		}
	}
	if path == "" {
		t.Skip("no model available")
	}
	m, err := onnx.DecodeFile(path)
	if err != nil {
		t.Fatal(err)
	}
	g, err := graph.FromONNX(m)
	if err != nil {
		t.Fatal(err)
	}
	s, err := graph.Compile(g)
	if err != nil {
		t.Fatal(err)
	}
	in := g.Inputs[0]
	shape := make([]int, len(in.Shape))
	for i, d := range in.Shape {
		if d <= 0 {
			d = 1
			if i >= len(in.Shape)-2 {
				d = 32
			}
		}
		shape[i] = d
	}
	mk := func() map[string]*tensor.Tensor {
		x := tensor.New(in.DType, shape...)
		if in.DType == tensor.F32 {
			f := x.F32()
			for i := range f {
				f[i] = float32(i%17)*0.03 - 0.2
			}
		}
		return map[string]*tensor.Tensor{in.Name: x}
	}
	outName := g.Outputs[0].Name
	base, err := s.Run(mk())
	if err != nil {
		t.Fatal(err)
	}
	want := append([]float32(nil), base[outName].F32()...)

	const G, iters = 8, 4
	var wg sync.WaitGroup
	errs := make(chan error, G)
	for w := 0; w < G; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for it := 0; it < iters; it++ {
				out, err := s.Run(mk())
				if err != nil {
					errs <- err
					return
				}
				got := out[outName].F32()
				for i := range want {
					if got[i] != want[i] {
						errs <- fmt.Errorf("concurrent result diverged at %d: %g != %g", i, got[i], want[i])
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Fatal(e)
	}
}
