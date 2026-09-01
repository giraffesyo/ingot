package ocr

import (
	"fmt"
	"image"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

// Classifier is the PP-OCR text-direction classifier: for each box it decides
// whether the crop is upside-down (rotated 180°). Rotation is applied by
// relabelling the box corners — no pixel work.
type Classifier struct {
	sess    *graph.Session
	inName  string
	outName string
	// Thresh: rotate only when P(180°) is at least this (PP-OCR default 0.9 —
	// the classifier is deliberately conservative, a wrong flip is worse than
	// a missed one).
	Thresh float64
}

// Classifier input geometry (PP-OCR cls_image_shape).
const (
	clsH = 48
	clsW = 192
)

// NewClassifier loads a PP-OCR direction classifier from an ONNX file.
func NewClassifier(path string) (*Classifier, error) {
	m, err := onnx.DecodeFile(path)
	if err != nil {
		return nil, err
	}
	g, err := graph.FromONNX(m)
	if err != nil {
		return nil, err
	}
	s, err := graph.Compile(g)
	if err != nil {
		return nil, err
	}
	if len(g.Inputs) != 1 || len(g.Outputs) != 1 {
		return nil, fmt.Errorf("classifier: expected 1 input/1 output, got %d/%d", len(g.Inputs), len(g.Outputs))
	}
	return &Classifier{sess: s, inName: g.Inputs[0].Name, outName: g.Outputs[0].Name, Thresh: 0.9}, nil
}

// Rot180 reports, per box, whether the crop should be rotated 180° before
// recognition. All boxes run in one forward pass.
func (c *Classifier) Rot180(img image.Image, boxes []Box) ([]bool, error) {
	if len(boxes) == 0 {
		return nil, nil
	}
	x := tensor.New(tensor.F32, len(boxes), 3, clsH, clsW)
	xf := x.F32()
	for i, b := range boxes {
		w := min(cropWidth(b, clsH), clsW)
		// Padding stays 0 = normalised mid-grey, as in recognition batching.
		cropInto(xf[i*3*clsH*clsW:], img, b, clsH, w, clsW)
	}
	outs, err := c.sess.Run(map[string]*tensor.Tensor{c.inName: x})
	if err != nil {
		return nil, err
	}
	out := outs[c.outName]
	of := out.F32()
	if out.Numel() != 2*len(boxes) {
		return nil, fmt.Errorf("classifier: output %v for %d boxes", out.Shape(), len(boxes))
	}
	rot := make([]bool, len(boxes))
	for i := range boxes {
		rot[i] = float64(of[i*2+1]) >= c.Thresh
	}
	c.sess.Release(outs)
	return rot, nil
}

// rot180 returns the box with its corners relabelled so the sampled crop is
// rotated 180° (top-left swaps with bottom-right, top-right with bottom-left).
func rot180(b Box) Box {
	p := b.Pts
	b.Pts = [4]Point{p[2], p[3], p[0], p[1]}
	return b
}
