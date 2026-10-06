package wan

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

const refDir = "../../testdata/wan22"

// tinyDir is the tiny random-weight model tools/export/wan22_ref.py writes
// (a diffusers-layout directory, like the real snapshot).
func tinyDir(t testing.TB) string {
	d := filepath.Join(refDir, "tiny")
	if _, err := os.Stat(filepath.Join(d, "model_index.json")); err != nil {
		t.Skipf("tiny model not generated (tools/export/wan22_ref.py): %v", err)
	}
	return d
}

// realDir is the Wan2.2-TI2V-5B-Diffusers snapshot when its weights are in
// the HF cache (or WAN22_DIR), else the test is skipped.
func realDir(t testing.TB) string {
	d := os.Getenv("WAN22_DIR")
	if d == "" {
		var err error
		if d, err = FindSnapshot(Repo); err != nil {
			t.Skip(err)
		}
	}
	if _, err := os.Stat(filepath.Join(d, "transformer", "diffusion_pytorch_model.safetensors.index.json")); err != nil {
		t.Skipf("weights not downloaded: %v", err)
	}
	return d
}

func openSet(t testing.TB, dir string) *safetensors.Set {
	t.Helper()
	s, err := safetensors.OpenDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func compile(t testing.TB, g *graph.Graph, err error, device string) graph.Runner {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	r, err := graph.CompileOn(g, device)
	if err != nil {
		t.Skipf("device %s: %v", device, err)
	}
	return r
}

// refCase is a reference manifest from tools/export/wan22_ref.py.
type refCase struct {
	Meta    map[string]any `json:"meta"`
	Inputs  []refTensor    `json:"inputs"`
	Outputs []refTensor    `json:"outputs"`
}

type refTensor struct {
	Name  string `json:"name"`
	File  string `json:"file"`
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
}

// loadRef reads name.json or skips when the references were not generated.
func loadRef(t testing.TB, name string) *refCase {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(refDir, name+".json"))
	if err != nil {
		t.Skipf("reference %s not generated (tools/export/wan22_ref.py): %v", name, err)
	}
	var c refCase
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	return &c
}

// tensor loads a reference array (float32 or int64).
func (c *refCase) tensor(t testing.TB, name string) *tensor.Tensor {
	t.Helper()
	for _, r := range append(append([]refTensor{}, c.Inputs...), c.Outputs...) {
		if r.Name != name {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(refDir, r.File))
		if err != nil {
			t.Fatal(err)
		}
		switch r.DType {
		case "float32":
			out := tensor.New(tensor.F32, r.Shape...)
			for i := range out.F32() {
				out.F32()[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
			}
			return out
		case "int64":
			out := tensor.New(tensor.I64, r.Shape...)
			for i := range out.I64() {
				out.I64()[i] = int64(binary.LittleEndian.Uint64(raw[8*i:]))
			}
			return out
		}
		t.Fatalf("%s: dtype %s", name, r.DType)
	}
	t.Fatalf("reference has no tensor %q", name)
	return nil
}

func (c *refCase) metaFloat(key string) float64 { v, _ := c.Meta[key].(float64); return v }

// compare reports max |got-want| and fails beyond tol·max|want|.
func compare(t testing.TB, what string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	var maxd, maxw float64
	at := 0
	for i := range want {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		if d > maxd || math.IsNaN(d) {
			maxd, at = d, i
		}
		maxw = max(maxw, math.Abs(float64(want[i])))
	}
	rel := maxd / max(maxw, 1e-30)
	t.Logf("%s: max |Δ| %.3g (rel %.3g) at %d (got %g want %g), max |want| %.3g", what, maxd, rel, at, got[at], want[at], maxw)
	if !(rel <= tol) {
		t.Fatalf("%s: rel error %.3g > %.3g", what, rel, tol)
	}
}

// devices lists the executors the parity tests run on: the CPU always,
// the GPU (f32) where there is one.
func devices() []string {
	if metal.Available() {
		return []string{"cpu", "gpu"}
	}
	return []string{"cpu"}
}
