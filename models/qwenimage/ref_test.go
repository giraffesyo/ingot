package qwenimage

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// snapshotDir returns the local Qwen-Image-2.1 HF snapshot or skips.
func snapshotDir(t testing.TB) string {
	t.Helper()
	if d := os.Getenv("QWENIMAGE_DIR"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	snaps, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots/*"))
	if len(snaps) == 0 {
		t.Skip("Qwen-Image-2.1 snapshot not in the HF cache (set QWENIMAGE_DIR)")
	}
	return snaps[0]
}

const refDir = "../../testdata/qwenimage21"

// refCase is a reference manifest from tools/export/qwenimage21_ref.py.
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
		t.Skipf("reference %s not generated (tools/export/qwenimage21_ref.py): %v", name, err)
	}
	var c refCase
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	return &c
}

// tensor loads a reference array: float32 as F32, uint8 as U8.
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
		case "uint8":
			out := tensor.New(tensor.U8, r.Shape...)
			copy(out.U8(), raw)
			return out
		}
		t.Fatalf("%s: dtype %s", name, r.DType)
	}
	t.Fatalf("reference has no tensor %q", name)
	return nil
}

// compare reports max |got-want| and fails beyond tol (absolute).
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
	t.Logf("%s: max |Δ| %.3g at %d (got %g want %g), max |want| %.3g", what, maxd, at, got[at], want[at], maxw)
	if !(maxd <= tol) {
		t.Fatalf("%s: max |Δ| %.3g > %.3g", what, maxd, tol)
	}
}
