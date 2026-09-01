package ocr

import (
	"fmt"
	"image"
	_ "image/png"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

// Fixed input shapes for model-level benchmarks (match tools/export/orttime.py
// so "ours vs ORT" numbers are like-for-like).
var ocrBenchShapes = []struct {
	name  string
	model string
	shape tensor.Shape
}{
	{"det_640", "det.onnx", tensor.Shape{1, 3, 640, 640}},
	{"det_960", "det.onnx", tensor.Shape{1, 3, 960, 960}},
	{"rec_320", "rec.onnx", tensor.Shape{1, 3, 48, 320}},
	{"rec_b8_320", "rec.onnx", tensor.Shape{8, 3, 48, 320}},
	{"det_int8_640", "det_int8.onnx", tensor.Shape{1, 3, 640, 640}},
	{"rec_int8_320", "rec_int8.onnx", tensor.Shape{1, 3, 48, 320}},
}

func init() {
	// Benchmark knob: OCR_PAR_SPIN_NS overrides the helper spin window.
	if v := os.Getenv("OCR_PAR_SPIN_NS"); v != "" {
		fmt.Sscan(v, &par.SpinNS)
	}
	if v := os.Getenv("OCR_PAR_WORKERS"); v != "" {
		fmt.Sscan(v, &par.MaxWorkers)
	}
}

func loadOCRSession(tb testing.TB, model string, shape ...int) (*graph.Session, string) {
	tb.Helper()
	path := "../../testdata/ocr/" + model
	if _, err := os.Stat(path); err != nil {
		tb.Skip("PP-OCR models not present")
	}
	m, err := onnx.DecodeFile(path)
	if err != nil {
		tb.Fatal(err)
	}
	g, err := graph.FromONNX(m)
	if err != nil {
		tb.Fatal(err)
	}
	if len(shape) > 0 && len(g.Inputs) == 1 {
		if err := g.SetInputShape(g.Inputs[0].Name, shape...); err != nil {
			tb.Fatal(err)
		}
	}
	s, err := graph.Compile(g)
	if err != nil {
		tb.Fatal(err)
	}
	return s, g.Inputs[0].Name
}

func BenchmarkOCRModels(b *testing.B) {
	for _, c := range ocrBenchShapes {
		b.Run(c.name, func(b *testing.B) {
			s, in := loadOCRSession(b, c.model, c.shape...)
			x := tensor.New(tensor.F32, c.shape...)
			for i := range x.F32() {
				x.F32()[i] = float32(i%97)/97 - 0.5
			}
			feeds := map[string]*tensor.Tensor{in: x}
			if _, err := s.Run(feeds); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				res, err := s.Run(feeds)
				if err != nil {
					b.Fatal(err)
				}
				s.Release(res)
			}
		})
	}
}

// TestOCRProfile prints a per-op breakdown: OCR_PROFILE=det_640 go test -run TestOCRProfile -v
func TestOCRProfile(t *testing.T) {
	want := os.Getenv("OCR_PROFILE")
	if want == "" {
		t.Skip("set OCR_PROFILE=<det_640|det_960|rec_320|rec_b8_320>")
	}
	for _, c := range ocrBenchShapes {
		if c.name != want {
			continue
		}
		s, in := loadOCRSession(t, c.model, c.shape...)
		x := tensor.New(tensor.F32, c.shape...)
		feeds := map[string]*tensor.Tensor{in: x}
		for i := 0; i < 3; i++ {
			if _, err := s.Run(feeds); err != nil {
				t.Fatal(err)
			}
		}
		s.Profile = true
		const runs = 20
		start := time.Now()
		for i := 0; i < runs; i++ {
			if _, err := s.Run(feeds); err != nil {
				t.Fatal(err)
			}
		}
		total := time.Since(start)
		t.Logf("%s: %v/run over %d runs", c.name, total/runs, runs)
		for _, st := range s.Stats() {
			t.Logf("  %-22s n=%3d  %8.1f µs/run  %5.1f%%", st.OpType, st.Count, float64(st.Total.Microseconds())/runs, 100*float64(st.Total)/float64(total))
		}
		if os.Getenv("OCR_PROFILE_NODES") != "" {
			ns := s.NodeStats()
			sort.Slice(ns, func(i, j int) bool { return ns[i].Total > ns[j].Total })
			for i, ns := range ns {
				if i >= 40 {
					break
				}
				t.Logf("    %-8s %-40s %8.1f µs", ns.Node.OpType, ns.Node.Name, float64(ns.Total.Microseconds())/runs)
			}
		}
		return
	}
	t.Fatalf("unknown OCR_PROFILE %q", want)
}

// BenchmarkPipeline runs detection + recognition end-to-end on a corpus image,
// with recognition batching on (default) and off.
func BenchmarkPipeline(b *testing.B) {
	const dir = "../../testdata/ocr"
	if _, err := os.Stat(dir + "/det.onnx"); err != nil {
		b.Skip("PP-OCR models not present")
	}
	p, err := NewPipeline(dir+"/det.onnx", dir+"/rec.onnx", dir+"/rec_dict.txt")
	if err != nil {
		b.Fatal(err)
	}
	f, err := os.Open(dir + "/corpus/img_006.png")
	if err != nil {
		b.Skip(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		b.Fatal(err)
	}
	for _, bs := range []int{1, DefaultRecBatch} {
		b.Run(fmt.Sprintf("recbatch=%d", bs), func(b *testing.B) {
			p.RecBatch = bs
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := p.Run(img); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
