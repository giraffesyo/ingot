package wan

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

func TestCleanPrompt(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(refDir, "tokenizer.json"))
	if err != nil {
		t.Skipf("reference not generated (tools/export/wan22_ref.py): %v", err)
	}
	var ref struct {
		Cases []struct{ Text, Clean string } `json:"cases"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	for _, c := range ref.Cases {
		if got := CleanPrompt(c.Text); got != c.Clean {
			t.Errorf("CleanPrompt(%q)\n got %q\nwant %q", c.Text, got, c.Clean)
		}
	}
}

func testTextEncoder(t *testing.T, dir, ref string, tol float64) {
	c := loadRef(t, ref)
	cfg, err := LoadTextConfig(filepath.Join(dir, "text_encoder"))
	if err != nil {
		t.Fatal(err)
	}
	set := openSet(t, filepath.Join(dir, "text_encoder"))
	ids := c.tensor(t, "input_ids").I64()
	for _, dev := range devices() {
		t.Run(dev, func(t *testing.T) {
			te, err := NewTextEncoder(cfg, set, dev != "cpu")
			if err != nil {
				t.Fatal(err)
			}
			x, err := te.Embed(ids)
			if err != nil {
				t.Fatal(err)
			}
			g, err := te.Build(len(ids), cfg.NumLayers)
			r := compile(t, g, err, dev)
			out, err := r.Run(map[string]*tensor.Tensor{"x": x})
			if err != nil {
				t.Fatal(err)
			}
			compare(t, "hidden", out["hidden"].F32(), c.tensor(t, "hidden").F32(), tol)
		})
	}
}

func TestTextEncoderTiny(t *testing.T) { testTextEncoder(t, tinyDir(t), "umt5", 1e-5) }
func TestTextEncoderReal(t *testing.T) { testTextEncoder(t, realDir(t), "real_umt5", 1e-4) }
