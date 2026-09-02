package ocr

import (
	"image"
	"sort"
)

// BoxRecognizer recognises the text inside image crops. Recognizer (PP-OCR
// CTC line recogniser) and Parseq (scene-text word recogniser) implement it.
type BoxRecognizer interface {
	// RecognizeBatch recognises boxes in one forward pass, returning text and
	// confidence per box in order.
	RecognizeBatch(img image.Image, boxes []Box) ([]string, []float64, error)
	// CropWidth is the model-input width the box's crop takes. Pipeline groups
	// boxes of similar width into a batch to bound padding; fixed-size models
	// return a constant.
	CropWidth(box Box) int
}

// Pipeline bundles detection and recognition into an end-to-end OCR.
type Pipeline struct {
	Det *Detector
	Rec BoxRecognizer
	// Cls, when set (EnableClassifier), runs the PP-OCR direction classifier
	// on every detected box and flips upside-down crops before recognition.
	Cls *Classifier
	// RecBatch is the maximum number of boxes recognised per forward pass
	// (boxes are grouped by width so padding stays small). ≤1 disables batching.
	RecBatch int
	// RecPadRatio bounds the padding inside a batch: a batch only spans boxes
	// whose crop width is ≤ RecPadRatio × the narrowest box in it. Heavy
	// right-padding measurably hurts SVTR (it attends over the grey region);
	// 1 means only equal-width boxes share a batch.
	RecPadRatio float64
}

// Defaults used by NewPipeline.
const (
	DefaultRecBatch    = 8
	DefaultRecPadRatio = 1.25
)

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
	return &Pipeline{Det: d, Rec: r, RecBatch: DefaultRecBatch, RecPadRatio: DefaultRecPadRatio}, nil
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
	if p.Cls != nil {
		rots, err := p.Cls.Rot180(img, boxes)
		if err != nil {
			return nil, err
		}
		for i, r := range rots {
			if r {
				boxes[i] = rot180(boxes[i])
			}
		}
	}
	texts, confs, err := p.RecognizeBoxes(img, boxes)
	if err != nil {
		return nil, err
	}
	res := make([]Result, len(boxes))
	for i, b := range boxes {
		res[i] = Result{Box: b, Text: texts[i], Conf: confs[i]}
	}
	return res, nil
}

// RecognizeBoxes recognises boxes (in the given order) using width-grouped
// batches of at most RecBatch crops whose widths lie within RecPadRatio of
// the narrowest in the batch.
func (p *Pipeline) RecognizeBoxes(img image.Image, boxes []Box) ([]string, []float64, error) {
	texts := make([]string, len(boxes))
	confs := make([]float64, len(boxes))
	order := make([]int, len(boxes))
	for i := range order {
		order[i] = i
	}
	widths := make([]int, len(boxes))
	for i, b := range boxes {
		widths[i] = p.Rec.CropWidth(b)
	}
	sort.SliceStable(order, func(a, b int) bool {
		return widths[order[a]] < widths[order[b]]
	})
	bs := max(1, p.RecBatch)
	ratio := p.RecPadRatio
	if ratio < 1 {
		ratio = 1
	}
	for lo := 0; lo < len(order); {
		w0 := float64(widths[order[lo]])
		hi := lo + 1
		for hi < len(order) && hi-lo < bs && float64(widths[order[hi]]) <= w0*ratio {
			hi++
		}
		batch := make([]Box, 0, hi-lo)
		for _, i := range order[lo:hi] {
			batch = append(batch, boxes[i])
		}
		ts, cs, err := p.Rec.RecognizeBatch(img, batch)
		if err != nil {
			return nil, nil, err
		}
		for k, i := range order[lo:hi] {
			texts[i], confs[i] = ts[k], cs[k]
		}
		lo = hi
	}
	return texts, confs, nil
}

// EnableClassifier loads the direction classifier and enables 180° crop
// correction in Run.
func (p *Pipeline) EnableClassifier(path string) error {
	c, err := NewClassifier(path)
	if err != nil {
		return err
	}
	p.Cls = c
	return nil
}
