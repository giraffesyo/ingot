package ocr

import (
	"fmt"
	"image"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

// Detector runs a DBNet-style text detector and returns text boxes.
type Detector struct {
	sess      *graph.Session
	inName    string
	outName   string
	limit     int
	BoxThresh float64 // min mean-probability inside a box to keep it
	BinThresh float64 // probability threshold for binarisation
	Unclip    float64 // polygon expansion ratio
	MinSize   int     // minimum box side (pixels, original scale)
}

// NewDetector loads a detection model from an ONNX file.
func NewDetector(path string) (*Detector, error) {
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
		return nil, fmt.Errorf("detector: expected 1 input/1 output, got %d/%d", len(g.Inputs), len(g.Outputs))
	}
	return &Detector{
		sess: s, inName: g.Inputs[0].Name, outName: g.Outputs[0].Name,
		limit: 960, BoxThresh: 0.6, BinThresh: 0.3, Unclip: 1.5, MinSize: 3,
	}, nil
}

// ProbMap runs the model and returns the [H,W] probability map plus the scale
// factors (resized/original) for mapping back to original image coordinates.
func (d *Detector) ProbMap(img image.Image) (prob []float32, H, W int, sx, sy float64, err error) {
	in, sx, sy := preprocessDet(img, d.limit)
	outs, err := d.sess.Run(map[string]*tensor.Tensor{d.inName: in})
	if err != nil {
		return nil, 0, 0, 0, 0, err
	}
	out := outs[d.outName]
	os := out.Shape()
	if os.Rank() != 4 || os[1] != 1 {
		return nil, 0, 0, 0, 0, fmt.Errorf("detector: unexpected output shape %v", os)
	}
	return out.F32(), os[2], os[3], sx, sy, nil
}

// Detect returns text boxes in original-image coordinates.
func (d *Detector) Detect(img image.Image) ([]Box, error) {
	prob, H, W, sx, sy, err := d.ProbMap(img)
	if err != nil {
		return nil, err
	}
	return boxesFromProb(prob, H, W, sx, sy, d), nil
}
