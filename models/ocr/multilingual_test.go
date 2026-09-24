package ocr

import (
	"encoding/json"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"testing"
)

const v5Dir = "../../testdata/ocr/v5"

var v5Dicts = map[string]string{
	"ch": "ppocrv5_dict.txt", "korean": "ppocrv5_korean_dict.txt", "latin": "ppocrv5_latin_dict.txt",
	"cyrillic": "ppocrv5_cyrillic_dict.txt", "arabic": "ppocrv5_arabic_dict.txt",
}

// TestMultilingual reads rendered lines in seven languages (Chinese,
// Japanese, Korean, Russian, French, German, Spanish, Arabic) with the
// PP-OCRv5 recognizer for each script: character accuracy per language
// against the truth (Arabic is recognised in visual order and reversed for
// comparison by the recognizer itself: RTL detected from the dictionary).
func TestMultilingual(t *testing.T) {
	b, err := os.ReadFile("../../testdata/ocr/multilingual/truth.json")
	if err != nil {
		t.Skip(err)
	}
	var lines []struct{ Image, Lang, Model, Text string }
	if err := json.Unmarshal(b, &lines); err != nil {
		t.Fatal(err)
	}
	recs := map[string]*Recognizer{}
	edits, chars := map[string]int{}, map[string]int{}
	for _, l := range lines {
		r := recs[l.Model]
		if r == nil {
			mp := filepath.Join(v5Dir, l.Model+"_PP-OCRv5_rec_mobile.onnx")
			if _, err := os.Stat(mp); err != nil {
				t.Skip("PP-OCRv5 models not present")
			}
			if r, err = NewRecognizerOn(mp, filepath.Join(v5Dir, v5Dicts[l.Model]), "cpu"); err != nil {
				t.Fatal(err)
			}
			recs[l.Model] = r
		}
		f, err := os.Open(filepath.Join("../../testdata/ocr/multilingual", l.Image))
		if err != nil {
			t.Fatal(err)
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		bb := img.Bounds()
		box := Box{Pts: [4]Point{{0, 0}, {float64(bb.Dx()), 0}, {float64(bb.Dx()), float64(bb.Dy())}, {0, float64(bb.Dy())}}}
		texts, _, err := r.RecognizeBatch(img, []Box{box})
		if err != nil {
			t.Fatal(err)
		}
		got := texts[0]
		if (l.Lang == "ar") != r.RTL {
			t.Fatalf("%s: RTL = %v", l.Model, r.RTL)
		}
		e := levenshtein(normText(l.Text), normText(got))
		edits[l.Lang] += e
		chars[l.Lang] += len([]rune(l.Text))
		if e > 0 {
			t.Logf("%s %s: %q, want %q", l.Lang, l.Image, got, l.Text)
		}
	}
	for lang, n := range chars {
		acc := 1 - float64(edits[lang])/float64(n)
		t.Log(fmt.Sprintf("%s: char accuracy %.3f (%d edits / %d)", lang, acc, edits[lang], n))
		if acc < 0.9 {
			t.Errorf("%s: char accuracy %.3f", lang, acc)
		}
	}
}

// TestLogicalOrder: RTL reversal keeps Latin/digit/space runs whole, as
// PaddleOCR's pred_reverse does (an all-Latin line is one run, unchanged).
func TestLogicalOrder(t *testing.T) {
	for in, want := range map[string]string{
		"cba":         "cba",
		"ج ب أ":       "أ ب ج",
		"ب 2024 أ":    "أ 2024 ب",
		"ب ingot 1.2": " ingot 1.2ب",
	} {
		if got := logicalOrder(in); got != want {
			t.Errorf("logicalOrder(%q) = %q, want %q", in, got, want)
		}
	}
}
