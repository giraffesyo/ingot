package ocr

import (
	"image"
	_ "image/png"
	"os"
	"testing"
)

// TestPipelineEndToEnd runs the full OCR pipeline if the PP-OCR models are
// present (skips otherwise), asserting exact text on the synthetic sample.
func TestPipelineEndToEnd(t *testing.T) {
	const dir = "../../testdata/ocr"
	if _, err := os.Stat(dir + "/det.onnx"); err != nil {
		t.Skip("PP-OCR models not present")
	}
	p, err := NewPipeline(dir+"/det.onnx", dir+"/rec.onnx", dir+"/rec_dict.txt")
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(dir + "/sample.png")
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	res, err := p.Run(img)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	for _, r := range res {
		t.Logf("box det=%.2f rec=%.2f %q", r.Box.Score, r.Conf, r.Text)
		got[r.Text] = true
	}
	for _, want := range []string{"Hello World", "OCR 12345", "pure golang", "DBNet test"} {
		if !got[want] {
			t.Errorf("missing expected line %q", want)
		}
	}
}
