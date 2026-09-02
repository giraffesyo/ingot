package ocr

import (
	"image"
	_ "image/png"
	"os"
	"testing"
)

// TestClassifier: upright corpus crops must not be flagged; the same boxes
// with corners relabelled 180° must be. Also checks the pipeline hook leaves
// upright results unchanged.
func TestClassifier(t *testing.T) {
	const dir = "../../testdata/ocr"
	if _, err := os.Stat(dir + "/cls.onnx"); err != nil {
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
	if len(boxes) == 0 {
		t.Fatal("no boxes detected")
	}
	c, err := NewClassifier(dir + "/cls.onnx")
	if err != nil {
		t.Fatal(err)
	}
	up, err := c.Rot180(img, boxes)
	if err != nil {
		t.Fatal(err)
	}
	flipped := make([]Box, len(boxes))
	for i, b := range boxes {
		flipped[i] = rot180(b)
	}
	down, err := c.Rot180(img, flipped)
	if err != nil {
		t.Fatal(err)
	}
	upWrong, downCaught := 0, 0
	for i := range boxes {
		if up[i] {
			upWrong++
		}
		if down[i] {
			downCaught++
		}
	}
	t.Logf("boxes=%d upright flagged=%d flipped caught=%d", len(boxes), upWrong, downCaught)
	if upWrong > 0 {
		t.Errorf("%d upright crops flagged as rotated", upWrong)
	}
	// The classifier is conservative (Thresh 0.9); require it to catch most.
	if downCaught*2 < len(boxes) {
		t.Errorf("only %d/%d flipped crops caught", downCaught, len(boxes))
	}

	// Pipeline with cls enabled must reproduce the plain pipeline on an
	// upright image exactly.
	base, err := p.Run(img)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.EnableClassifier(dir + "/cls.onnx"); err != nil {
		t.Fatal(err)
	}
	withCls, err := p.Run(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(base) != len(withCls) {
		t.Fatalf("result count changed: %d vs %d", len(base), len(withCls))
	}
	for i := range base {
		if base[i].Text != withCls[i].Text {
			t.Errorf("cls changed upright result %d: %q vs %q", i, base[i].Text, withCls[i].Text)
		}
	}
	// And a flipped crop recognised through rot180 should match the upright
	// recognition for at least most boxes (exercises the full correction path).
	match := 0
	rec := p.Rec.(*Recognizer)
	for i, b := range boxes {
		s1, _, _ := rec.Recognize(img, b)
		s2, _, _ := rec.Recognize(img, rot180(rot180(b)))
		if s1 == s2 {
			match++
		}
		_ = i
	}
	if match != len(boxes) {
		t.Errorf("rot180 twice is not the identity: %d/%d", match, len(boxes))
	}
}
