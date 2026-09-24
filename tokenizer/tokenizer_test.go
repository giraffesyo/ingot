package tokenizer

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestSplit pins the pre-tokenizer's regex semantics on cases whose
// expected pieces follow from the pattern by hand.
func TestSplit(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"hello world", []string{"hello", " world"}},
		{"I'm don't WE'LL", []string{"I", "'m", " don", "'t", " WE", "'LL"}},
		{"abc123", []string{"abc", "1", "2", "3"}},
		{"a  b", []string{"a", " ", " b"}},             // \s+(?!\S) keeps the last space for the word
		{"end   ", []string{"end", "   "}},             // trailing run at end of text
		{"x\n\ny", []string{"x", "\n\n", "y"}},         // \s*[\r\n]+
		{"x \n y", []string{"x", " \n", " y"}},         // whitespace up to the last newline
		{"hi!!\n\nok", []string{"hi", "!!\n\n", "ok"}}, // punctuation absorbs trailing newlines
		{"a ...b", []string{"a", " ...", "b"}},
		{"$5.00", []string{"$", "5", ".", "0", "0"}},
		{"東京 café", []string{"東京", " café"}},
		{"🦊fox", []string{"🦊fox"}}, // one non-letter may lead a letter run
	} {
		if got := Split(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("Split(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestQwenReference encodes the reference strings (tools/export/
// qwenimage21_ref.py, from the transformers tokenizer) with the Qwen-Image
// checkpoint's tokenizer.json and compares ids exactly.
func TestQwenReference(t *testing.T) {
	home, _ := os.UserHomeDir()
	paths, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots/*/processor/tokenizer.json"))
	raw, err := os.ReadFile("../testdata/qwenimage21/tokenizer.json")
	if len(paths) == 0 || err != nil {
		t.Skip("Qwen-Image-2.1 tokenizer or reference ids not present")
	}
	tok, err := Load(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Cases []struct {
			Text string  `json:"text"`
			IDs  []int64 `json:"ids"`
		} `json:"cases"`
		Template string `json:"template"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	for _, c := range ref.Cases {
		got, err := tok.Encode(c.Text)
		if err != nil {
			t.Fatalf("%q: %v", c.Text, err)
		}
		if !reflect.DeepEqual(got, c.IDs) {
			t.Errorf("Encode(%q)\n got %v\nwant %v", c.Text, got, c.IDs)
		}
	}
	// The full chat-templated prompt against the pipeline's input_ids.
	var pm struct {
		Inputs []struct {
			Name, File string
		} `json:"inputs"`
	}
	if raw, err := os.ReadFile("../testdata/qwenimage21/pipeline.json"); err == nil && json.Unmarshal(raw, &pm) == nil {
		for _, in := range pm.Inputs {
			if in.Name != "input_ids" {
				continue
			}
			b, err := os.ReadFile(filepath.Join("../testdata/qwenimage21", in.File))
			if err != nil {
				t.Fatal(err)
			}
			want := make([]int64, len(b)/8)
			for i := range want {
				want[i] = int64(binary.LittleEndian.Uint64(b[8*i:]))
			}
			got, err := tok.Encode(ref.Template)
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Errorf("template\n got %v (%v)\nwant %v", got, err, want)
			}
		}
	}
}
