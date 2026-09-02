package ocr

import (
	"image"
	_ "image/png"
	"os"
	"testing"
	"time"

	"github.com/giraffesyo/ingot/tensor"
)

func TestPipelineStages(t *testing.T) {
	if os.Getenv("OCR_STAGES") == "" {
		t.Skip()
	}
	const dir = "../../testdata/ocr"
	p, err := NewPipeline(dir+"/det.onnx", dir+"/rec.onnx", dir+"/rec_dict.txt")
	if err != nil {
		t.Skip(err)
	}
	f, _ := os.Open(dir + "/corpus/img_006.png")
	defer f.Close()
	img, _, _ := image.Decode(f)
	t.Logf("image %v type %T", img.Bounds().Size(), img)
	for i := 0; i < 3; i++ {
		p.Run(img)
	}
	const n = 10
	var tPre, tDet, tPost, tRec time.Duration
	var nbox int
	for i := 0; i < n; i++ {
		t0 := time.Now()
		in, sx, sy := preprocessDet(img, p.Det.limit)
		t1 := time.Now()
		outs, _ := p.Det.sess.Run(map[string]*tensor.Tensor{p.Det.inName: in})
		t2 := time.Now()
		out := outs[p.Det.outName]
		os := out.Shape()
		boxes := boxesFromProb(out.F32(), os[2], os[3], sx, sy, p.Det)
		t3 := time.Now()
		p.RecognizeBoxes(img, boxes)
		t4 := time.Now()
		tPre += t1.Sub(t0)
		tDet += t2.Sub(t1)
		tPost += t3.Sub(t2)
		tRec += t4.Sub(t3)
		nbox = len(boxes)
	}
	t.Logf("boxes=%d  preprocess %v  det-forward %v  db-postproc %v  crop+rec %v", nbox, tPre/n, tDet/n, tPost/n, tRec/n)
	in, sx, sy := preprocessDet(img, p.Det.limit)
	outs, _ := p.Det.sess.Run(map[string]*tensor.Tensor{p.Det.inName: in})
	out := outs[p.Det.outName]
	os := out.Shape()
	boxes := boxesFromProb(out.F32(), os[2], os[3], sx, sy, p.Det)
	rec := p.Rec.(*Recognizer)
	for _, b := range boxes {
		w := cropWidth(b, 48)
		t0 := time.Now()
		x := tensor.New(tensor.F32, 1, 3, 48, w)
		cropInto(x.F32(), img, b, 48, w, w, cropUpMax)
		t1 := time.Now()
		rec.sess.Run(map[string]*tensor.Tensor{rec.inName: x})
		t2 := time.Now()
		rec.sess.Run(map[string]*tensor.Tensor{rec.inName: x})
		t3 := time.Now()
		t.Logf("  box W=%d crop %v  rec-forward %v (2nd %v)", w, t1.Sub(t0), t2.Sub(t1), t3.Sub(t2))
	}
}
