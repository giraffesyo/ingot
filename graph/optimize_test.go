package graph_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

// TestOptimizeAB runs every zoo model (and the PP-OCR models if present)
// through CompileRaw and Compile (optimised) on the same input and requires
// the outputs to agree. The optimiser must be a pure performance transform.
func TestOptimizeAB(t *testing.T) {
	type cand struct {
		name, path string
		feeds      func(tb testing.TB, g *graph.Graph) map[string]*tensor.Tensor
	}
	var cands []cand
	for _, name := range discoverModels(t) {
		name := name
		cands = append(cands, cand{name, "", func(tb testing.TB, g *graph.Graph) map[string]*tensor.Tensor {
			_, man := loadSession(tb, name) // reuse manifest parsing
			feeds := map[string]*tensor.Tensor{}
			for _, in := range man.Inputs {
				feeds[in.Name] = loadBin(tb, in)
			}
			return feeds
		}})
	}
	for _, m := range []struct {
		name  string
		shape []int
	}{{"det", []int{1, 3, 160, 160}}, {"rec", []int{1, 3, 48, 160}}} {
		p := filepath.Join("..", "testdata", "ocr", m.name+".onnx")
		if _, err := os.Stat(p); err != nil {
			continue
		}
		shape := m.shape
		cands = append(cands, cand{"ocr_" + m.name, p, func(tb testing.TB, g *graph.Graph) map[string]*tensor.Tensor {
			x := tensor.New(tensor.F32, shape...)
			f := x.F32()
			for i := range f {
				f[i] = float32((i*7919)%1000)/500 - 1
			}
			return map[string]*tensor.Tensor{g.Inputs[0].Name: x}
		}})
	}
	for _, c := range cands {
		t.Run(c.name, func(t *testing.T) {
			load := func() *graph.Graph {
				path := c.path
				if path == "" {
					mb, err := os.ReadFile(filepath.Join(modelDir, c.name+".json"))
					if err != nil {
						t.Skip(err)
					}
					var man manifest
					if err := json.Unmarshal(mb, &man); err != nil {
						t.Fatal(err)
					}
					path = filepath.Join(modelDir, man.Model)
				}
				m, err := onnx.DecodeFile(path)
				if err != nil {
					t.Skip(err)
				}
				g, err := graph.FromONNX(m)
				if err != nil {
					t.Fatal(err)
				}
				return g
			}
			graw, gopt := load(), load()
			feeds := c.feeds(t, graw)
			raw, err := graph.CompileRaw(graw)
			if err != nil {
				t.Skip(err)
			}
			nRaw := len(gopt.Nodes)
			stats := graph.Optimize(gopt)
			opt, err := graph.CompileRaw(gopt)
			if err != nil {
				t.Fatalf("compile optimised: %v", err)
			}
			t.Logf("%s: %d → %d nodes, %v", c.name, nRaw, len(gopt.Nodes), stats)
			// Pattern coverage pins: a model's known fusions must keep firing.
			if want, ok := wantFusions[c.name]; ok {
				for k, n := range want {
					if stats[k] != n {
						t.Errorf("%s: %s fired %d times, want %d", c.name, k, stats[k], n)
					}
				}
			}
			a, err := raw.Run(feeds)
			if err != nil {
				t.Fatal(err)
			}
			b, err := opt.Run(feeds)
			if err != nil {
				t.Fatal(err)
			}
			for name, ta := range a {
				tb := b[name]
				if tb == nil {
					t.Fatalf("output %q missing from optimised run", name)
				}
				if !ta.Shape().Equal(tb.Shape()) {
					t.Fatalf("output %q shape %v vs %v", name, ta.Shape(), tb.Shape())
				}
				if ta.DType() != tensor.F32 {
					continue
				}
				fa, fb := ta.F32(), tb.F32()
				var worst, scale float64
				for i := range fa {
					worst = math.Max(worst, math.Abs(float64(fa[i]-fb[i])))
					scale = math.Max(scale, math.Abs(float64(fa[i])))
				}
				tol := 1e-4 * math.Max(1, scale)
				if worst > tol {
					t.Errorf("output %q: max abs diff %g (scale %g, tol %g)", name, worst, scale, tol)
				}
			}
		})
	}
}

// wantFusions pins fusion counts on zoo models whose export form a pass was
// written for (a silent pattern miss is a perf regression, not a test
// failure, so it is asserted here).
var wantFusions = map[string]map[string]int{
	"parseq_nar":    {"fuse-sdpa": 16, "fuse-mha-packed": 12, "fuse-gelu": 15, "fuse-matmul-bias": 65, "fuse-add-layernorm": 33},
	"parseq_nar_b3": {"fuse-sdpa": 16, "fuse-mha-packed": 12, "fuse-gelu": 15, "fuse-matmul-bias": 65, "fuse-add-layernorm": 33},
	"bertish":       {"fuse-sdpa": 2, "fuse-gelu": 2, "fuse-matmul-bias": 6},
	"vit":           {"fuse-sdpa": 2, "fuse-gelu": 2, "fuse-matmul-bias": 6},
}
