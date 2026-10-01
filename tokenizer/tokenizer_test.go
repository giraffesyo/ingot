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

// TestSplitMistral pins the Mistral-style split on hand-derived cases.
func TestSplitMistral(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"hello world", []string{"hello", " world"}},
		{"I'm don't", []string{"I", "'m", " don", "'t"}}, // no contraction rule: ' leads the letter run
		{"HelloWorld", []string{"Hello", "World"}},       // case boundary splits
		{"HTTPServer", []string{"HTTPServer"}},           // Up* backtracks only as far as a lower letter needs
		{"ABC def", []string{"ABC", " def"}},             // all-caps via [Up]+[Low]*
		{"abc123", []string{"abc", "1", "2", "3"}},
		{"a  b", []string{"a", " ", " b"}},
		{"x\n\ny", []string{"x", "\n\n", "y"}},
		{"a+/b", []string{"a", "+/", "b"}},
		{"hi!/\n/x", []string{"hi", "!/\n/", "x"}}, // [\r\n/]* after punctuation
		{"東京 café", []string{"東京", " café"}},
		{"$5.00", []string{"$", "5", ".", "0", "0"}},
	} {
		if got := SplitMistral(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("SplitMistral(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestQwen3TTSReference encodes tools/export/qwen3tts_ref.py's strings two
// ways — the backend tokenizer.json the reference processor ran, and the
// checkpoint's vocab.json + merges.txt through LoadBPE — and compares ids
// exactly.
func TestQwen3TTSReference(t *testing.T) {
	home, _ := os.UserHomeDir()
	snaps, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/*"))
	raw, err := os.ReadFile("../testdata/qwen3tts/tokenizer.json")
	if len(snaps) == 0 || err != nil {
		t.Skip("Qwen3-TTS snapshot or reference ids not present")
	}
	var ref struct {
		Cases []struct {
			Text string  `json:"text"`
			IDs  []int64 `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	backend, err := Load("../testdata/qwen3tts/backend_tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	add, err := AddedTokensFromConfig(filepath.Join(snaps[0], "tokenizer_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	bpe, err := LoadBPE(filepath.Join(snaps[0], "vocab.json"), filepath.Join(snaps[0], "merges.txt"), add, PreMistral)
	if err != nil {
		t.Fatal(err)
	}
	for name, tok := range map[string]*Tokenizer{"backend": backend, "bpe": bpe} {
		for _, c := range ref.Cases {
			got, err := tok.Encode(c.Text)
			if err != nil {
				t.Fatalf("%s %q: %v", name, c.Text, err)
			}
			if !reflect.DeepEqual(got, c.IDs) {
				t.Errorf("%s: Encode(%q)\n got %v\nwant %v", name, c.Text, got, c.IDs)
			}
		}
	}
}
