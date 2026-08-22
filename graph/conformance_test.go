package graph_test

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

type manifest struct {
	Model   string `json:"model"`
	Opset   int    `json:"opset"`
	Inputs  []io   `json:"inputs"`
	Outputs []io   `json:"outputs"`
}
type io struct {
	Name  string `json:"name"`
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
	File  string `json:"file"`
}

const modelDir = "../testdata/models"

func loadBin(tb testing.TB, e io) *tensor.Tensor {
	tb.Helper()
	b, err := os.ReadFile(filepath.Join(modelDir, e.File))
	if err != nil {
		tb.Fatal(err)
	}
	switch e.DType {
	case "float32":
		f := make([]float32, len(b)/4)
		for i := range f {
			f[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		}
		return tensor.FromF32(f, e.Shape...)
	case "int64":
		v := make([]int64, len(b)/8)
		for i := range v {
			v[i] = int64(binary.LittleEndian.Uint64(b[i*8:]))
		}
		return tensor.FromI64(v, e.Shape...)
	}
	tb.Fatalf("unsupported dtype %s", e.DType)
	return nil
}

// loadSession loads <name>.json manifest + model; skips if absent.
func loadSession(tb testing.TB, name string) (*graph.Session, manifest) {
	tb.Helper()
	mb, err := os.ReadFile(filepath.Join(modelDir, name+".json"))
	if err != nil {
		tb.Skipf("no manifest for %s (run tools/export): %v", name, err)
	}
	var man manifest
	if err := json.Unmarshal(mb, &man); err != nil {
		tb.Fatal(err)
	}
	m, err := onnx.DecodeFile(filepath.Join(modelDir, man.Model))
	if err != nil {
		tb.Skipf("model load: %v", err)
	}
	g, err := graph.FromONNX(m)
	if err != nil {
		tb.Fatalf("build: %v", err)
	}
	s, err := graph.Compile(g)
	if err != nil {
		tb.Fatalf("compile: %v", err)
	}
	return s, man
}

func runConformance(t *testing.T, name string, tol float64) {
	s, man := loadSession(t, name)
	feeds := map[string]*tensor.Tensor{}
	for _, in := range man.Inputs {
		feeds[in.Name] = loadBin(t, in)
	}
	start := time.Now()
	outs, err := s.Run(feeds)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	t.Logf("%s: %d nodes, run %v", name, len(s.Graph().Nodes), time.Since(start))
	for _, o := range man.Outputs {
		want := loadBin(t, o)
		got, ok := outs[o.Name]
		if !ok {
			t.Fatalf("missing output %q", o.Name)
		}
		if !got.Shape().Equal(want.Shape()) {
			t.Fatalf("output %q shape %v, want %v", o.Name, got.Shape(), want.Shape())
		}
		gf, wf := got.F32(), want.F32()
		var maxAbs, maxRel float64
		worst := 0
		for i := range wf {
			d := math.Abs(float64(gf[i] - wf[i]))
			r := d / math.Max(math.Abs(float64(wf[i])), 1e-3)
			if d > maxAbs {
				maxAbs = d
				worst = i
			}
			maxRel = math.Max(maxRel, r)
		}
		t.Logf("output %q: max abs err %.3g (at %d: got %g want %g), max rel err %.3g", o.Name, maxAbs, worst, gf[worst], wf[worst], maxRel)
		if maxAbs > tol {
			t.Errorf("output %q: max abs err %.3g > tol %.3g", o.Name, maxAbs, tol)
		}
	}
}

func TestTinyConv(t *testing.T)         { runConformance(t, "tiny_conv", 1e-5) }
func TestTinyTransformer(t *testing.T)  { runConformance(t, "tiny_transformer", 1e-4) }
func TestMobileNetV3Small(t *testing.T) { runConformance(t, "mobilenet_v3_small", 1e-3) }

// TestOpProfile prints a per-op-type time breakdown for a model
// (go test -run TestOpProfile -v -args -model=mobilenet_v3_small).
func TestOpProfile(t *testing.T) {
	name := os.Getenv("OCR_PROFILE_MODEL")
	if name == "" {
		t.Skip("set OCR_PROFILE_MODEL=<name>")
	}
	s, man := loadSession(t, name)
	feeds := map[string]*tensor.Tensor{}
	for _, in := range man.Inputs {
		feeds[in.Name] = loadBin(t, in)
	}
	for i := 0; i < 5; i++ { // warm up
		if _, err := s.Run(feeds); err != nil {
			t.Fatal(err)
		}
	}
	s.Profile = true
	const runs = 50
	start := time.Now()
	for i := 0; i < runs; i++ {
		if _, err := s.Run(feeds); err != nil {
			t.Fatal(err)
		}
	}
	total := time.Since(start)
	t.Logf("%s: %v/run over %d runs", name, total/runs, runs)
	for _, st := range s.Stats() {
		t.Logf("  %-22s n=%3d  %8.1f µs/run  %5.1f%%", st.OpType, st.Count, float64(st.Total.Microseconds())/runs, 100*float64(st.Total)/float64(total))
	}
	if os.Getenv("OCR_PROFILE_NODES") != "" {
		for _, ns := range s.NodeStats() {
			t.Logf("    %-40s %8.1f µs", ns.Node.Name, float64(ns.Total.Microseconds())/runs)
		}
	}
}

// discoverModels returns all <name>.json manifests in the model dir.
func discoverModels(tb testing.TB) []string {
	ents, err := os.ReadDir(modelDir)
	if err != nil {
		tb.Skipf("no model dir: %v", err)
	}
	var names []string
	for _, e := range ents {
		if n, ok := strings.CutSuffix(e.Name(), ".json"); ok {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	return names
}

// tolFor returns the abs-error tolerance for a model (looser for deep stacks).
func tolFor(name string) float64 {
	switch {
	case strings.Contains(name, "mobilenet"), strings.Contains(name, "efficientnet"),
		strings.Contains(name, "resnet"):
		return 2e-3
	default:
		return 1e-3
	}
}

// TestZoo runs every discovered model and reports which compile+run and match
// ORT, and which surface unsupported ops. It never fails on unsupported ops
// (those are known gaps, documented in docs/GAPS.md) but fails on a wrong
// numeric result for a model that did run.
func TestZoo(t *testing.T) {
	for _, name := range discoverModels(t) {
		name := name
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
			s, err := graph.Compile(g)
			if err != nil {
				t.Skipf("OP GAP: %v", err)
			}
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
				gf, wf := got.F32(), want.F32()
				var maxAbs float64
				for i := range wf {
					d := math.Abs(float64(gf[i] - wf[i]))
					if d > maxAbs {
						maxAbs = d
					}
				}
				tol := tolFor(name)
				if maxAbs > tol {
					t.Fatalf("output %q: max abs err %.3g > tol %.3g", o.Name, maxAbs, tol)
				}
				t.Logf("%s OK: %d nodes, max abs err %.2g", name, len(s.Graph().Nodes), maxAbs)
			}
		})
	}
}

func BenchmarkModels(b *testing.B) {
	for _, name := range discoverModels(b) {
		b.Run(name, func(b *testing.B) {
			s, man := loadSession(b, name)
			feeds := map[string]*tensor.Tensor{}
			for _, in := range man.Inputs {
				feeds[in.Name] = loadBin(b, in)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := s.Run(feeds); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
