package graph_test

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

func readF32(t *testing.T, path string) []float32 {
	b, err := os.ReadFile(path)
	if err != nil {
		t.Skip(err)
	}
	f := make([]float32, len(b)/4)
	for i := range f {
		f[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return f
}

func TestPPOCRDetection(t *testing.T) {
	const dir = "../testdata/ocr"
	mb, err := os.ReadFile(dir + "/det.meta.json")
	if err != nil {
		t.Skip("no det model (fetch RapidOCR models)")
	}
	var meta struct {
		InShape    []int  `json:"in_shape"`
		OutShape   []int  `json:"out_shape"`
		InputName  string `json:"input_name"`
		OutputName string `json:"output_name"`
	}
	if err := json.Unmarshal(mb, &meta); err != nil {
		t.Fatal(err)
	}
	m, err := onnx.DecodeFile(dir + "/det.onnx")
	if err != nil {
		t.Fatal(err)
	}
	g, err := graph.FromONNX(m)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	s, err := graph.Compile(g)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	in := tensor.FromF32(readF32(t, dir+"/det.in.bin"), meta.InShape...)
	outs, err := s.Run(map[string]*tensor.Tensor{meta.InputName: in})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	got := outs[meta.OutputName]
	if got == nil {
		t.Fatalf("no output %q", meta.OutputName)
	}
	want := readF32(t, dir+"/det.out.bin")
	gf := got.F32()
	var maxAbs float64
	for i := range want {
		d := math.Abs(float64(gf[i] - want[i]))
		if d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("PP-OCRv4 det: %d nodes, out %v, max abs err %.3g", len(g.Nodes), got.Shape(), maxAbs)
	if maxAbs > 1e-3 {
		t.Errorf("max abs err %.3g > 1e-3", maxAbs)
	}
}
