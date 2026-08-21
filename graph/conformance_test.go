package graph_test

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/giraffesyo/ocr/graph"
	"github.com/giraffesyo/ocr/onnx"
	"github.com/giraffesyo/ocr/tensor"
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
	raw, err := os.ReadFile(filepath.Join(modelDir, man.Model))
	if err != nil {
		tb.Skipf("model file missing: %v", err)
	}
	m, err := onnx.Decode(raw)
	if err != nil {
		tb.Fatalf("decode: %v", err)
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

func BenchmarkModels(b *testing.B) {
	for _, name := range []string{"tiny_conv", "tiny_transformer", "mobilenet_v3_small"} {
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
