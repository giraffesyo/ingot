package qwenimage

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// tokenizerRef is testdata/qwenimage21/tokenizer.json.
type tokenizerRef struct {
	Cases []struct {
		Text string  `json:"text"`
		IDs  []int64 `json:"ids"`
	} `json:"cases"`
	Template string `json:"template"`
	DropIdx  int    `json:"drop_idx"`
}

func loadTokenizerRef(t *testing.T) tokenizerRef {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(refDir, "tokenizer.json"))
	if err != nil {
		t.Skipf("tokenizer reference not generated: %v", err)
	}
	var r tokenizerRef
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

// TestTextEncoderParity runs the full 36-layer language model over the
// reference token ids and compares the conditioning hidden states.
func TestTextEncoderParity(t *testing.T) {
	fullModel(t)
	dir := filepath.Join(snapshotDir(t), "text_encoder")
	ref := loadRef(t, "pipeline")
	tref := loadTokenizerRef(t)
	cfg, err := LoadTextConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	te, err := NewTextEncoder(cfg, set)
	if err != nil {
		t.Fatal(err)
	}
	ids := ref.tensor(t, "input_ids").I64()
	g, err := te.Build(len(ids), tref.DropIdx, cfg.NumHiddenLayers)
	if err != nil {
		t.Fatal(err)
	}
	s, err := graph.Compile(g)
	if err != nil {
		t.Fatal(err)
	}
	x, err := te.Embed(ids)
	if err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	out, err := s.Run(map[string]*tensor.Tensor{"x": x})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("text encoder, %d tokens: %v, sys %.1f GB", len(ids), time.Since(t0), rssGB())
	want := ref.tensor(t, "prompt_embeds")
	if !out["hidden"].Shape().Equal(want.Shape()) {
		t.Fatalf("shape %v, want %v", out["hidden"].Shape(), want.Shape())
	}
	var maxw float64
	for _, v := range want.F32() {
		maxw = max(maxw, float64(abs32(v)))
	}
	compare(t, "prompt_embeds", out["hidden"].F32(), want.F32(), 2e-4*maxw)
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
