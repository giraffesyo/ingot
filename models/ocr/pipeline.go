package ocr

import (
	"image"
	"sort"
)

// Pipeline bundles detection and recognition into an end-to-end OCR.
type Pipeline struct {
	Det *Detector
	Rec *Recognizer
}

// Result is one recognised text line.
type Result struct {
	Box  Box
	Text string
	Conf float64
}

// NewPipeline loads detection + recognition models and the char dictionary.
func NewPipeline(detPath, recPath, dictPath string) (*Pipeline, error) {
	d, err := NewDetector(detPath)
	if err != nil {
		return nil, err
	}
	r, err := NewRecognizer(recPath, dictPath)
	if err != nil {
		return nil, err
	}
	return &Pipeline{Det: d, Rec: r}, nil
}

// Run detects and recognises all text in img, returned in reading order
// (top-to-bottom, then left-to-right).
func (p *Pipeline) Run(img image.Image) ([]Result, error) {
	boxes, err := p.Det.Detect(img)
	if err != nil {
		return nil, err
	}
	sort.Slice(boxes, func(i, j int) bool {
		yi := (boxes[i].Pts[0].Y + boxes[i].Pts[1].Y) / 2
		yj := (boxes[j].Pts[0].Y + boxes[j].Pts[1].Y) / 2
		if yi-yj > 10 || yj-yi > 10 {
			return yi < yj
		}
		return boxes[i].Pts[0].X < boxes[j].Pts[0].X
	})
	res := make([]Result, 0, len(boxes))
	for _, b := range boxes {
		text, conf, err := p.Rec.Recognize(img, b)
		if err != nil {
			return nil, err
		}
		res = append(res, Result{Box: b, Text: text, Conf: conf})
	}
	return res, nil
}
