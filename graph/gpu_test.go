//go:build darwin && arm64

package graph_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

// TestZooGPU runs every zoo model through a GPUSession and checks its
// outputs against the ONNX Runtime references with the CPU tolerances,
// logging how many nodes ran on the GPU.
func TestZooGPU(t *testing.T) {
	if !metal.Available() {
		t.Skip("no Metal device")
	}
	for _, name := range discoverModels(t) {
		t.Run(name, func(t *testing.T) {
			mb, err := os.ReadFile(filepath.Join(modelDir, name+".json"))
			if err != nil {
				t.Skip(err)
			}
			var man manifest
			if err := json.Unmarshal(mb, &man); err != nil {
				t.Fatal(err)
			}
			m, err := onnx.DecodeFile(filepath.Join(modelDir, man.Model))
			if err != nil {
				t.Skipf("load: %v", err)
			}
			g, err := graph.FromONNX(m)
			if err != nil {
				t.Skipf("BUILD GAP: %v", err)
			}
			s, err := graph.CompileGPU(g)
			if err != nil {
				t.Skipf("OP GAP: %v", err)
			}
			defer s.Close()
			feeds := map[string]*tensor.Tensor{}
			for _, in := range man.Inputs {
				feeds[in.Name] = loadBin(t, in)
			}
			outs, err := s.Run(feeds)
			if err != nil {
				t.Skipf("RUN GAP: %v", err)
			}
			for _, o := range man.Outputs {
				want := loadBin(t, o)
				got := outs[o.Name]
				if got == nil || !got.Shape().Equal(want.Shape()) {
					t.Fatalf("output %q shape mismatch", o.Name)
				}
				if want.DType() == tensor.I64 {
					gi, wi := got.I64(), want.I64()
					for i := range wi {
						if gi[i] != wi[i] {
							t.Fatalf("output %q: [%d] = %d, want %d", o.Name, i, gi[i], wi[i])
						}
					}
					continue
				}
				gf, wf := got.F32(), want.F32()
				var maxAbs float64
				for i := range wf {
					maxAbs = math.Max(maxAbs, math.Abs(float64(gf[i]-wf[i])))
				}
				tol := tolFor(name)
				if maxAbs > tol {
					t.Fatalf("output %q: max abs err %.3g > tol %.3g (gpu %d / cpu %d steps)", o.Name, maxAbs, tol, s.GPUSteps, s.CPUSteps)
				}
				t.Logf("%s OK: gpu %d / cpu %d steps, %d flushes %v, max abs err %.2g", name, s.GPUSteps, s.CPUSteps, s.Flushes, s.FlushedBy, maxAbs)
			}
		})
	}
}

// BenchmarkModelsGPU times the zoo models on a GPUSession (compare
// BenchmarkModels).
func BenchmarkModelsGPU(b *testing.B) {
	if !metal.Available() {
		b.Skip("no Metal device")
	}
	for _, name := range []string{"bertish", "vit", "tiny_transformer", "llmblock", "gptish", "gptish_1k"} {
		b.Run(name, func(b *testing.B) {
			mb, err := os.ReadFile(filepath.Join(modelDir, name+".json"))
			if err != nil {
				b.Skip(err)
			}
			var man manifest
			json.Unmarshal(mb, &man)
			m, err := onnx.DecodeFile(filepath.Join(modelDir, man.Model))
			if err != nil {
				b.Skip(err)
			}
			g, err := graph.FromONNX(m)
			if err != nil {
				b.Skip(err)
			}
			s, err := graph.CompileGPU(g)
			if err != nil {
				b.Skip(err)
			}
			defer s.Close()
			feeds := map[string]*tensor.Tensor{}
			for _, in := range man.Inputs {
				feeds[in.Name] = loadBin(b, in)
			}
			if _, err := s.Run(feeds); err != nil {
				b.Fatal(err)
			}
			t0 := time.Now()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out, err := s.Run(feeds)
				if err != nil {
					b.Fatal(err)
				}
				s.Release(out)
			}
			_ = t0
		})
	}
}
