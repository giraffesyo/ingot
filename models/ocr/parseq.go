package ocr

import (
	"bufio"
	"fmt"
	"image"
	"math"
	"os"
	"strings"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

// Parseq is the PARSeq scene-text recogniser (baudm/parseq: ViT encoder +
// permutation-trained transformer decoder), exported non-autoregressively —
// decode_ar=False with refinement iterations, see tools/export/parseq.py.
//
// Every crop is resized to 32×128 without preserving aspect ratio (the model's
// training transform) and normalised to [-1, 1]. Output logits are [B, T, 1+C]
// per decode step over [EOS, charset...]; decoding is greedy and stops at the
// first EOS. The training charset is 94 printable ASCII characters — no space
// and no CJK — so PARSeq recognises words, not lines; the PP-OCR Recognizer
// remains the line recogniser.
type Parseq struct {
	sess    *graph.Session
	inName  string
	outName string
	charset []rune // output class i+1 -> charset[i]; class 0 is EOS
}

// PARSeq input geometry (img_size in the model config).
const (
	parseqH = 32
	parseqW = 128
)

// NewParseq loads a PARSeq ONNX export and its charset file (the training
// charset as one line, in output-class order).
func NewParseq(modelPath, charsetPath string) (*Parseq, error) {
	m, err := onnx.DecodeFile(modelPath)
	if err != nil {
		return nil, err
	}
	g, err := graph.FromONNX(m)
	if err != nil {
		return nil, err
	}
	if len(g.Inputs) != 1 || len(g.Outputs) != 1 {
		return nil, fmt.Errorf("parseq: expected 1 input/1 output, got %d/%d", len(g.Inputs), len(g.Outputs))
	}
	s, err := graph.Compile(g)
	if err != nil {
		return nil, err
	}
	cs, err := loadCharset(charsetPath)
	if err != nil {
		return nil, err
	}
	return &Parseq{sess: s, inName: g.Inputs[0].Name, outName: g.Outputs[0].Name, charset: cs}, nil
}

// loadCharset reads a single-line charset file. Only the line terminator is
// stripped, so a charset containing a space keeps it.
func loadCharset(path string) ([]rune, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	line, err := r.ReadString('\n')
	if err != nil && line == "" {
		return nil, fmt.Errorf("parseq: read charset %s: %w", path, err)
	}
	cs := []rune(strings.TrimRight(line, "\r\n"))
	if len(cs) == 0 {
		return nil, fmt.Errorf("parseq: empty charset %s", path)
	}
	return cs, nil
}

// CropWidth is constant: PARSeq resizes every crop to 128 wide, so any boxes
// may share a batch (Pipeline's width grouping degenerates to plain chunking).
func (p *Parseq) CropWidth(Box) int { return parseqW }

// Recognize runs one box.
func (p *Parseq) Recognize(img image.Image, box Box) (string, float64, error) {
	texts, confs, err := p.RecognizeBatch(img, []Box{box})
	if err != nil {
		return "", 0, err
	}
	return texts[0], confs[0], nil
}

// RecognizeBatch recognises several boxes in one forward pass and returns the
// decoded text and confidence per box. Confidence is the mean of the per-step
// max probabilities over the emitted characters and the terminating EOS
// (PARSeq's own convention is their product; the mean matches Recognizer so
// the two are interchangeable behind Pipeline).
func (p *Parseq) RecognizeBatch(img image.Image, boxes []Box) ([]string, []float64, error) {
	if len(boxes) == 0 {
		return nil, nil, nil
	}
	const plane = 3 * parseqH * parseqW
	in := tensor.New(tensor.F32, len(boxes), 3, parseqH, parseqW)
	f := in.F32()
	for i, b := range boxes {
		cropInto(f[i*plane:(i+1)*plane], img, b, parseqH, parseqW, parseqW)
	}
	outs, err := p.sess.Run(map[string]*tensor.Tensor{p.inName: in})
	if err != nil {
		return nil, nil, err
	}
	out := outs[p.outName]
	os := out.Shape()
	if os.Rank() != 3 || os[0] != len(boxes) || os[2] != len(p.charset)+1 {
		return nil, nil, fmt.Errorf("parseq: unexpected output %v for batch %d, charset %d", os, len(boxes), len(p.charset))
	}
	T, C := os[1], os[2]
	of := out.F32()
	texts := make([]string, len(boxes))
	confs := make([]float64, len(boxes))
	for i := range boxes {
		texts[i], confs[i] = p.decode(of[i*T*C:(i+1)*T*C], T, C)
	}
	p.sess.Release(outs)
	return texts, confs, nil
}

// decode greedily decodes [T,C] logits: per step the argmax class; class 0 is
// EOS and ends the string, class c ≥ 1 is charset[c-1]. Probabilities are the
// softmax of the step's logits.
func (p *Parseq) decode(logits []float32, T, C int) (string, float64) {
	var sb strings.Builder
	var conf float64
	n := 0
	for t := range T {
		row := logits[t*C : (t+1)*C]
		best, m := 0, row[0]
		for c := 1; c < C; c++ {
			if row[c] > m {
				best, m = c, row[c]
			}
		}
		var den float64
		for _, v := range row {
			den += math.Exp(float64(v - m))
		}
		conf += 1 / den
		n++
		if best == 0 {
			break
		}
		sb.WriteRune(p.charset[best-1])
	}
	return sb.String(), conf / float64(n)
}
