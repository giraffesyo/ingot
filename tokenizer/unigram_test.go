package tokenizer

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
)

// toyUnigram is a hand-built T5-style tokenizer.json: the scores make the
// best segmentation differ from greedy longest-match.
const toyUnigram = `{
 "added_tokens": [
  {"id": 0, "content": "<pad>", "special": true, "normalized": false, "lstrip": false, "rstrip": false, "single_word": false},
  {"id": 1, "content": "</s>", "special": true, "normalized": false, "lstrip": false, "rstrip": false, "single_word": false},
  {"id": 10, "content": "<x>", "special": true, "normalized": false, "lstrip": false, "rstrip": false, "single_word": false}
 ],
 "normalizer": {"type": "Sequence", "normalizers": [{"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}]},
 "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
 "post_processor": {"type": "TemplateProcessing",
  "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
  "special_tokens": {"</s>": {"id": "</s>", "ids": [1], "tokens": ["</s>"]}}},
 "model": {"type": "Unigram", "unk_id": 2, "byte_fallback": false, "vocab": [
  ["<pad>", 0.0], ["</s>", 0.0], ["<unk>", 0.0],
  ["▁", -2.0], ["▁ab", -1.0], ["c", -3.0], ["▁a", -1.5], ["bc", -1.0], ["▁abc", -5.0], ["d", -4.0], ["<x>", 0.0]
 ]}
}`

func TestUnigramToy(t *testing.T) {
	u, err := parseUnigram([]byte(toyUnigram))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		in   string
		want []int64
	}{
		// ▁abc: ▁a+bc = -2.5 beats ▁ab+c = -4 and ▁abc = -5.
		{"abc", []int64{6, 7, 1}},
		{"  abc   d ", []int64{6, 7, 3, 9, 3, 1}},
		// Unknown characters: min score -5 minus 10 each, a run fused to one <unk>.
		{"zzc", []int64{3, 2, 5, 1}},
		{"abc<x>d", []int64{6, 7, 10, 3, 9, 1}},
		{"</s>", []int64{1, 1}},
	} {
		if got := u.Encode(c.in, true); !slices.Equal(got, c.want) {
			t.Errorf("Encode(%q) = %v, want %v", c.in, got, c.want)
		}
	}
	if got := u.Encode("abc", false); !slices.Equal(got, []int64{6, 7}) {
		t.Errorf("without special tokens: %v", got)
	}
}

func TestUnigramRefuses(t *testing.T) {
	var j map[string]any
	if err := json.Unmarshal([]byte(toyUnigram), &j); err != nil {
		t.Fatal(err)
	}
	j["pre_tokenizer"] = map[string]any{"type": "Whitespace"}
	raw, _ := json.Marshal(j)
	if _, err := parseUnigram(raw); err == nil {
		t.Fatal("Whitespace pre-tokenizer accepted")
	}
}

// TestUnigramUMT5 checks umT5's tokenizer.json against transformers'
// T5TokenizerFast ids (tools/export/wan22_ref.py writes the cases).
func TestUnigramUMT5(t *testing.T) {
	raw, err := os.ReadFile("../testdata/wan22/tokenizer.json")
	if err != nil {
		t.Skipf("reference not generated (tools/export/wan22_ref.py): %v", err)
	}
	u, err := LoadUnigram("../testdata/wan22/tiny/tokenizer/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Cases []struct {
			Text   string  `json:"text"`
			Clean  string  `json:"clean"`
			IDs    []int64 `json:"ids"`
			RawIDs []int64 `json:"raw_ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	for _, c := range ref.Cases {
		if got := u.Encode(c.Clean, true); !slices.Equal(got, c.IDs) {
			t.Errorf("%q: got %v\nwant %v", c.Clean, got, c.IDs)
		}
		if got := u.Encode(c.Text, true); !slices.Equal(got, c.RawIDs) {
			t.Errorf("raw %q: got %v\nwant %v", c.Text, got, c.RawIDs)
		}
	}
}
