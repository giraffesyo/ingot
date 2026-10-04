package trellis2

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// hubSnapshot returns the local HF snapshot of repo ("org/name"), or the
// directory in the environment variable env, or skips.
func hubSnapshot(t testing.TB, env, repo string) string {
	t.Helper()
	if d := os.Getenv(env); d != "" {
		return d
	}
	hub := os.Getenv("HF_HUB_CACHE")
	if hub == "" {
		home, _ := os.UserHomeDir()
		hub = filepath.Join(home, ".cache/huggingface/hub")
	}
	snaps, _ := filepath.Glob(filepath.Join(hub, "models--"+filepath.Base(filepath.Dir(repo))+"--"+filepath.Base(repo), "snapshots", "*"))
	if len(snaps) == 0 {
		t.Skipf("%s snapshot not in the HF cache (set %s)", repo, env)
	}
	return snaps[0]
}

func trellisDir(t testing.TB) string { return hubSnapshot(t, "TRELLIS2_DIR", "microsoft/TRELLIS.2-4B") }
func dinoDir(t testing.TB) string {
	return hubSnapshot(t, "TRELLIS2_DINO_DIR", "facebook/dinov3-vitl16-pretrain-lvd1689m")
}
func structureDir(t testing.TB) string {
	return hubSnapshot(t, "TRELLIS2_STRUCTURE_DIR", "microsoft/TRELLIS-image-large")
}

func openFile(t testing.TB, path string) *safetensors.File {
	t.Helper()
	f, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	return f
}

func compile(t testing.TB, g *graph.Graph, err error) *graph.Session {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	s, err := graph.Compile(g)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// flowFeeds returns a BuildFlow graph's inputs for one evaluation: x, t and
// cond's cross-attention keys and values over the first blocks blocks.
func flowFeeds(t testing.TB, cfg FlowConfig, f *safetensors.File, blocks int, x, cond *tensor.Tensor, time float32) map[string]*tensor.Tensor {
	t.Helper()
	kv, err := FlowConditions(cfg, f, blocks, cond)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[string]*tensor.Tensor(kv[0])
	feeds["x"], feeds["t"] = x, tensor.FromF32([]float32{time}, 1)
	return feeds
}

const refDir = "../../testdata/trellis2"

// refCase is a reference manifest from tools/export/trellis2_ref.py.
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
		t.Skipf("reference %s not generated (tools/export/trellis2_ref.py): %v", name, err)
	}
	var c refCase
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	return &c
}

// tensor loads a reference array (float32, int32 or uint8).
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
		case "int32":
			out := tensor.New(tensor.I32, r.Shape...)
			for i := range out.I32() {
				out.I32()[i] = int32(binary.LittleEndian.Uint32(raw[4*i:]))
			}
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
