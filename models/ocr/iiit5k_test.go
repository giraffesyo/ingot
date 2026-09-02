package ocr

import (
	"encoding/json"
	"fmt"
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unicode"
)

// TestIIIT5K scores word recognisers on the IIIT5K-Word test split (3000
// scene-text crops; tools/export/iiit5k.py fetches it). Protocol is the
// usual one for the benchmark: case-insensitive, alphanumeric characters
// only. Each crop is one whole-image box. OCR_IIIT5K_LIMIT=n runs a prefix.
func TestIIIT5K(t *testing.T) {
	const dir = "../../testdata/iiit5k"
	mb, err := os.ReadFile(dir + "/manifest.json")
	if err != nil {
		t.Skip("no IIIT5K (run tools/export/iiit5k.py)")
	}
	var words []struct {
		Image string `json:"image"`
		Text  string `json:"text"`
	}
	if err := json.Unmarshal(mb, &words); err != nil {
		t.Fatal(err)
	}
	if v := os.Getenv("OCR_IIIT5K_LIMIT"); v != "" {
		var n int
		fmt.Sscan(v, &n)
		if n > 0 && n < len(words) {
			words = words[:n]
		}
	}
	norm := func(s string) string {
		var sb strings.Builder
		for _, r := range s {
			if r < 128 && (unicode.IsLetter(r) || unicode.IsDigit(r)) {
				sb.WriteRune(unicode.ToLower(r))
			}
		}
		return sb.String()
	}
	type rec struct {
		name string
		r    BoxRecognizer
		min  float64 // accuracy gate
	}
	var recs []rec
	if pr, err := NewParseq(parseqModel, parseqCharset); err == nil {
		recs = append(recs, rec{"parseq", pr, 0.90})
	} else {
		t.Logf("parseq unavailable: %v", err)
	}
	if r, err := NewRecognizer(corpusDir+"/rec.onnx", corpusDir+"/rec_dict.txt"); err == nil {
		recs = append(recs, rec{"ppocr-rec", r, 0})
	} else {
		t.Logf("PP-OCR rec unavailable: %v", err)
	}
	if len(recs) == 0 {
		t.Skip("no recognizer models present")
	}
	imgs := make([]image.Image, len(words))
	for i, w := range words {
		f, err := os.Open(filepath.Join(dir, w.Image))
		if err != nil {
			t.Fatal(err)
		}
		imgs[i], _, err = image.Decode(f)
		f.Close()
		if err != nil {
			t.Fatalf("%s: %v", w.Image, err)
		}
	}
	for _, rc := range recs {
		correct := 0
		var elapsed time.Duration
		for i, w := range words {
			b := imgs[i].Bounds()
			W, H := float64(b.Dx()), float64(b.Dy())
			box := Box{Pts: [4]Point{{0, 0}, {W, 0}, {W, H}, {0, H}}}
			t0 := time.Now()
			texts, _, err := rc.r.RecognizeBatch(imgs[i], []Box{box})
			elapsed += time.Since(t0)
			if err != nil {
				t.Fatalf("%s %s: %v", rc.name, w.Image, err)
			}
			if norm(texts[0]) == norm(w.Text) {
				correct++
			} else if os.Getenv("OCR_DUMP") != "" {
				t.Logf("%s MISS %s gt=%q pred=%q", rc.name, w.Image, w.Text, texts[0])
			}
		}
		acc := float64(correct) / float64(len(words))
		t.Logf("IIIT5K %-10s accuracy %.4f (%d/%d)  %.2f ms/word", rc.name, acc, correct, len(words), float64(elapsed.Microseconds())/1e3/float64(len(words)))
		if acc < rc.min {
			t.Errorf("%s accuracy %.4f < %.2f", rc.name, acc, rc.min)
		}
	}
}
