//go:build darwin && arm64

package graph_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
	for _, name := range []string{"bertish", "vit", "tiny_transformer", "llmblock", "gptish", "gptish_1k", "mobilenet_v2", "mobilenet_v3_small", "efficientnet_b0", "resnetish", "parseq_nar"} {
		for _, mode := range []string{"f32", "bf16"} {
			b.Run(name+"/"+mode, func(b *testing.B) {
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
				var opts []graph.GPUOption
				if mode == "bf16" {
					opts = append(opts, graph.GPUBF16())
				}
				s, err := graph.CompileGPU(g, opts...)
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
				for b.Loop() {
					out, err := s.Run(feeds)
					if err != nil {
						b.Fatal(err)
					}
					s.Release(out)
				}
			})
		}
	}
}

// TestZooGPUBF16 runs the zoo with GPUBF16: bf16 weight products must stay
// within a relative error of the output scale (max |reference|), and
// classifiers must keep their top-1.
func TestZooGPUBF16(t *testing.T) {
	if !metal.Available() {
		t.Skip("no Metal device")
	}
	for _, name := range discoverModels(t) {
		if strings.Contains(name, "int8") {
			continue // quantized chains: not a bf16 target
		}
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
			s, err := graph.CompileGPU(g, graph.GPUBF16())
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
				if want.DType() != tensor.F32 {
					continue
				}
				gf, wf := got.F32(), want.F32()
				var maxAbs, scale float64
				for i := range wf {
					maxAbs = math.Max(maxAbs, math.Abs(float64(gf[i]-wf[i])))
					scale = math.Max(scale, math.Abs(float64(wf[i])))
				}
				rel := maxAbs / math.Max(scale, 1e-6)
				t.Logf("%s: max abs err %.2g (%.2g of scale %.3g)", name, maxAbs, rel, scale)
				if rel > 3e-2 {
					t.Fatalf("output %q: error %.3g of the output scale", o.Name, rel)
				}
				if ws := want.Shape(); len(ws) == 2 && ws[1] >= 10 { // classifier logits
					C := ws[1]
					for r := range ws[0] {
						if a, b := argmax(gf[r*C:(r+1)*C]), argmax(wf[r*C:(r+1)*C]); a != b {
							t.Fatalf("row %d: top-1 %d, reference %d", r, a, b)
						}
					}
				}
			}
		})
	}
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// TestZooGPUSelfCheck runs every zoo model with GPUSession.Check: each GPU
// node's outputs must agree with its CPU op on the same inputs (shape,
// dtype, and values to 1e-3 of the output scale).
func TestZooGPUSelfCheck(t *testing.T) {
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
			s.Check = true
			feeds := map[string]*tensor.Tensor{}
			for _, in := range man.Inputs {
				feeds[in.Name] = loadBin(t, in)
			}
			if _, err := s.Run(feeds); err != nil {
				t.Skipf("RUN GAP: %v", err)
			}
			for _, mm := range s.Mismatches {
				t.Errorf("%s %s output %d: max abs %.3g of scale %.3g %s", mm.Node.OpType, mm.Node.Name, mm.Output, mm.MaxAbs, mm.Scale, mm.Detail)
			}
		})
	}
}
