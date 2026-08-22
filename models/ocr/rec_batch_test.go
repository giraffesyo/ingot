package ocr

import (
	"image"
	_ "image/png"
	"os"
	"testing"
)

// TestRecognizeBatchConsistency: a batch of identical crops must reproduce
// single-box recognition exactly (no N>1 numerical drift in the ops), and
// padding a crop to a much wider batch is logged (it does change results —
// which is why Pipeline bounds batches by RecPadRatio).
func TestRecognizeBatchConsistency(t *testing.T) {
	const dir = "../../testdata/ocr"
	if _, err := os.Stat(dir + "/det.onnx"); err != nil {
		t.Skip("PP-OCR models not present")
	}
	p, err := NewPipeline(dir+"/det.onnx", dir+"/rec.onnx", dir+"/rec_dict.txt")
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(dir + "/corpus/img_006.png")
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	boxes, err := p.Det.Detect(img)
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range boxes {
		s1, c1, _ := p.Rec.Recognize(img, b)
		ts, cs, err := p.Rec.RecognizeBatch(img, []Box{b, b, b})
		if err != nil {
			t.Fatal(err)
		}
		for k := range ts {
			if ts[k] != s1 || cs[k] != c1 {
				t.Errorf("dup batch differs: single %q %.6f vs batch[%d] %q %.6f (W=%d)", s1, c1, k, ts[k], cs[k], cropWidth(b, 48))
			}
		}
	}
	// padding effect: each box alone vs in a batch with the widest box
	wmax := 0
	var wide Box
	for _, b := range boxes {
		if w := cropWidth(b, 48); w > wmax {
			wmax, wide = w, b
		}
	}
	for _, b := range boxes {
		s1, _, _ := p.Rec.Recognize(img, b)
		ts, _, _ := p.Rec.RecognizeBatch(img, []Box{b, wide})
		if ts[0] != s1 {
			t.Logf("padding changes result: W=%d→%d: %q vs %q", cropWidth(b, 48), wmax, s1, ts[0])
		}
	}
}
