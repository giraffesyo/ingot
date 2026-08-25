package ocr

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

// TestInt8Parity runs the quantized PP-OCR models against ONNX Runtime
// reference outputs (tools/export/ocr_int8.py). Tolerances are int8-scaled:
// a ±1-quantum requant tie-break in a late layer moves the dequantized
// output by that layer's scale.
func TestInt8Parity(t *testing.T) {
	for _, c := range []struct {
		name  string
		shape []int
		tol   float64
	}{
		{"det_int8", []int{1, 3, 640, 640}, 0.03}, // prob map in [0,1]
		{"rec_int8", []int{1, 3, 48, 320}, 0.05},  // post-softmax probs
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := "../../testdata/ocr/"
			if _, err := os.Stat(dir + c.name + ".onnx"); err != nil {
				t.Skip("quantized models not present (tools/export/ocr_int8.py)")
			}
			m, err := onnx.DecodeFile(dir + c.name + ".onnx")
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
			in := readBinF32(t, dir+c.name+".in.bin")
			want := readBinF32(t, dir+c.name+".out.bin")
			x := tensor.New(tensor.F32, c.shape...)
			copy(x.F32(), in)
			outs, err := s.Run(map[string]*tensor.Tensor{g.Inputs[0].Name: x})
			if err != nil {
				t.Fatal(err)
			}
			got := outs[g.Outputs[0].Name].F32()
			if len(got) != len(want) {
				t.Fatalf("out len %d want %d", len(got), len(want))
			}
			var worst float64
			bad := 0
			for i := range want {
				d := math.Abs(float64(got[i] - want[i]))
				if d > worst {
					worst = d
				}
				if d > c.tol {
					bad++
				}
			}
			t.Logf("%s: max abs err %.4g, >tol %d/%d", c.name, worst, bad, len(want))
			if float64(bad) > 0.001*float64(len(want)) { // ≤0.1%% of elements may exceed (tie chains)
				t.Fatalf("%d/%d elements exceed tol %g (max %g)", bad, len(want), c.tol, worst)
			}
		})
	}
}

func readBinF32(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}
