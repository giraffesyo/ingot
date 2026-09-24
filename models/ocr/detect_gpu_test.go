package ocr

import (
	"fmt"
	"image"
	_ "image/png"
	"math"
	"math/rand/v2"
	"os"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

// detGraph loads testdata's detector and its reference input.
func detGraph(tb testing.TB) (*graph.Graph, *tensor.Tensor) {
	tb.Helper()
	if !metal.Available() {
		tb.Skip("no Metal device")
	}
	m, err := onnx.DecodeFile(detDir + "/det.onnx")
	if err != nil {
		tb.Skip(err)
	}
	g, err := graph.FromONNX(m)
	if err != nil {
		tb.Fatal(err)
	}
	f, err := os.Open(detDir + "/sample.png")
	if err != nil {
		tb.Skip(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		tb.Fatal(err)
	}
	in, _, _ := preprocessDet(img, 960)
	return g, in
}

const detDir = "../../testdata/ocr"

// TestDetectorGPU: the detector's probability map (sample.png at 960) on a
// GPUSession matches the CPU session.
func TestDetectorGPU(t *testing.T) {
	g, in := detGraph(t)
	cpu, err := graph.Compile(g)
	if err != nil {
		t.Fatal(err)
	}
	g2, _ := detGraph(t)
	gpu, err := graph.CompileGPU(g2)
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Close()
	feeds := map[string]*tensor.Tensor{g.Inputs[0].Name: in}
	want, err := cpu.Run(feeds)
	if err != nil {
		t.Fatal(err)
	}
	got, err := gpu.Run(feeds)
	if err != nil {
		t.Fatal(err)
	}
	name := g.Outputs[0].Name
	var maxAbs float64
	for i, w := range want[name].F32() {
		maxAbs = math.Max(maxAbs, math.Abs(float64(got[name].F32()[i]-w)))
	}
	t.Logf("gpu %d / cpu %d steps, %d flushes %v; max abs err %.2g", gpu.GPUSteps, gpu.CPUSteps, gpu.Flushes, gpu.FlushedBy, maxAbs)
	if maxAbs > 1e-4 {
		t.Fatalf("max abs err %g", maxAbs)
	}
}

// BenchmarkDetectorDevice times one detector forward on each device:
// sample.png (320²) and a 960² input (the detector's size limit).
func BenchmarkDetectorDevice(b *testing.B) {
	for _, size := range []int{320, 960} {
		for _, dev := range []string{"cpu", "gpu"} {
			b.Run(fmt.Sprintf("size=%d/device=%s", size, dev), func(b *testing.B) {
				g, in := detGraph(b)
				if size != 320 {
					in = tensor.New(tensor.F32, 1, 3, size, size)
					r := rand.New(rand.NewPCG(1, 1))
					for i := range in.F32() {
						in.F32()[i] = r.Float32()*2 - 1
					}
				}
				r, err := graph.CompileOn(g, dev)
				if err != nil {
					b.Fatal(err)
				}
				feeds := map[string]*tensor.Tensor{g.Inputs[0].Name: in}
				for b.Loop() {
					out, err := r.Run(feeds)
					if err != nil {
						b.Fatal(err)
					}
					r.Release(out)
				}
			})
		}
	}
}
