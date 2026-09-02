package ocr

import (
	"bufio"
	"fmt"
	"image"
	"math"
	"os"
	"runtime"
	"strings"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

// Recognizer runs a CRNN/SVTR-style text recogniser with CTC decoding.
type Recognizer struct {
	sess    *graph.Session
	inName  string
	outName string
	dict    []string // index i -> character; index 0 is CTC blank
	height  int      // model input height (48 for PP-OCRv4)
	// BeamWidth > 1 selects CTC prefix beam search over greedy decoding
	// (ctcbeam.go). The posterior is near-one-hot on clean text, so the
	// default stays greedy; see docs/PERF.md for the corpus A/B.
	BeamWidth int
}

// NewRecognizer loads a recognition model and its character dictionary. The
// dictionary file has one character per line; CTC blank is prepended at index 0
// and a space appended, matching PP-OCR's label layout.
func NewRecognizer(modelPath, dictPath string) (*Recognizer, error) {
	m, err := onnx.DecodeFile(modelPath)
	if err != nil {
		return nil, err
	}
	g, err := graph.FromONNX(m)
	if err != nil {
		return nil, err
	}
	if len(g.Inputs) == 1 && runtime.GOARCH == "amd64" {
		// Typical line-crop extent (48-high, ~320 wide): steers layout
		// placement only; batched and wider runs stay correct. amd64 only —
		// blocked rec measured rec_b8 −15% on Zen 5 but rec_320 +11% on
		// Apple silicon (the blocked-conv advantage is an x86 story; NEON
		// pipeline GEMM wins there).
		_ = g.SetInputShape(g.Inputs[0].Name, 1, 3, 48, 320)
	}
	s, err := graph.Compile(g)
	if err != nil {
		return nil, err
	}
	dict, err := loadDict(dictPath)
	if err != nil {
		return nil, err
	}
	return &Recognizer{sess: s, inName: g.Inputs[0].Name, outName: g.Outputs[0].Name, dict: dict, height: 48}, nil
}

func loadDict(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	chars := []string{"<blank>"} // CTC blank at index 0
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		chars = append(chars, strings.TrimRight(sc.Text(), "\r\n"))
	}
	chars = append(chars, " ") // PP-OCR appends a space class
	return chars, sc.Err()
}

// CropWidth is the model-input width of the box's crop at the model height
// (aspect-preserving; see cropWidth).
func (r *Recognizer) CropWidth(box Box) int { return cropWidth(box, r.height) }

// Recognize crops the box from img, runs recognition, and returns the decoded
// text and mean confidence.
func (r *Recognizer) Recognize(img image.Image, box Box) (string, float64, error) {
	texts, confs, err := r.RecognizeBatch(img, []Box{box})
	if err != nil {
		return "", 0, err
	}
	return texts[0], confs[0], nil
}

// RecognizeBatch recognises several boxes in one forward pass. Each crop is
// resized to the model height at its own aspect ratio; narrower crops are
// right-padded (with the normalised mid-grey value 0, as PP-OCR does) to the
// widest crop in the batch. Callers get the best throughput by grouping boxes
// of similar width (see Pipeline).
func (r *Recognizer) RecognizeBatch(img image.Image, boxes []Box) ([]string, []float64, error) {
	if len(boxes) == 0 {
		return nil, nil, nil
	}
	H := r.height
	widths := make([]int, len(boxes))
	Wmax := 0
	for i, b := range boxes {
		widths[i] = cropWidth(b, H)
		Wmax = max(Wmax, widths[i])
	}
	in := tensor.New(tensor.F32, len(boxes), 3, H, Wmax) // zeroed: padding is 0
	f := in.F32()
	for i, b := range boxes {
		cropInto(f[i*3*H*Wmax:(i+1)*3*H*Wmax], img, b, H, widths[i], Wmax)
	}
	outs, err := r.sess.Run(map[string]*tensor.Tensor{r.inName: in})
	if err != nil {
		return nil, nil, err
	}
	out := outs[r.outName]
	os := out.Shape()
	if os.Rank() != 3 || os[0] != len(boxes) {
		return nil, nil, fmt.Errorf("recognizer: unexpected output %v for batch %d", os, len(boxes))
	}
	T, C := os[1], os[2]
	of := out.F32()
	texts := make([]string, len(boxes))
	confs := make([]float64, len(boxes))
	for i := range boxes {
		if r.BeamWidth > 1 {
			texts[i], confs[i] = r.beamDecode(of[i*T*C:(i+1)*T*C], T, C)
			continue
		}
		t, c, err := r.ctcDecode(of[i*T*C:(i+1)*T*C], T, C)
		if err != nil {
			return nil, nil, err
		}
		texts[i], confs[i] = t, c
	}
	r.sess.Release(outs) // decoded to strings above; tensors no longer referenced
	return texts, confs, nil
}

// ctcDecode performs greedy CTC decoding over a [1,T,C] probability tensor.
func (r *Recognizer) ctcDecode(p []float32, T, C int) (string, float64, error) {
	var sb strings.Builder
	prev := -1
	var conf float64
	var n int
	for t := 0; t < T; t++ {
		row := p[t*C : (t+1)*C]
		best, bestP := 0, row[0]
		for c := 1; c < C; c++ {
			if row[c] > bestP {
				best, bestP = c, row[c]
			}
		}
		if best != 0 && best != prev {
			if best < len(r.dict) {
				sb.WriteString(r.dict[best])
			}
			conf += float64(bestP)
			n++
		}
		prev = best
	}
	if n > 0 {
		conf /= float64(n)
	}
	return sb.String(), conf, nil
}

// cropWidth is the model-input width for a box at height H: its aspect ratio
// (using the longer of the two edges in each direction), at least 4.
func cropWidth(box Box, H int) int {
	p := box.Pts
	cw := math.Max(dist(p[0], p[1]), dist(p[3], p[2]))
	ch := math.Max(dist(p[0], p[3]), dist(p[1], p[2]))
	return int(math.Max(4, math.Round(float64(H)*cw/math.Max(ch, 1))))
}

// cropInto writes the normalised crop of box (resized to H×W) into the
// [3,H,Wstride] planes at dst; columns ≥ W are left untouched (the caller
// zeroes them for padding). Pixels are sampled with a bilinear corner blend
// (exact for parallelograms) and nearest-pixel lookup, reading *image.RGBA /
// *image.NRGBA pixel buffers directly when possible.
func cropInto(dst []float32, img image.Image, box Box, H, W, Wstride int) {
	p := box.Pts
	plane := H * Wstride
	b := img.Bounds()
	bw, bh := b.Dx(), b.Dy()
	var pix []uint8
	var stride int
	switch im := img.(type) {
	case *image.RGBA:
		pix, stride = im.Pix, im.Stride
	case *image.NRGBA:
		pix, stride = im.Pix, im.Stride
	}
	sample := func(x, y float64) (float64, float64, float64) {
		xi := clampI(int(x+0.5), 0, bw-1)
		yi := clampI(int(y+0.5), 0, bh-1)
		if pix != nil {
			o := yi*stride + xi*4
			return float64(pix[o]), float64(pix[o+1]), float64(pix[o+2])
		}
		r, g, bl, _ := img.At(b.Min.X+xi, b.Min.Y+yi).RGBA()
		return float64(r >> 8), float64(g >> 8), float64(bl >> 8)
	}
	for oy := 0; oy < H; oy++ {
		v := (float64(oy) + 0.5) / float64(H)
		for ox := 0; ox < W; ox++ {
			u := (float64(ox) + 0.5) / float64(W)
			topx := p[0].X*(1-u) + p[1].X*u
			topy := p[0].Y*(1-u) + p[1].Y*u
			botx := p[3].X*(1-u) + p[2].X*u
			boty := p[3].Y*(1-u) + p[2].Y*u
			sx := topx*(1-v) + botx*v
			sy := topy*(1-v) + boty*v
			rr, gg, bb := sample(sx, sy)
			o := oy*Wstride + ox
			dst[o] = float32(rr/255*2 - 1)
			dst[plane+o] = float32(gg/255*2 - 1)
			dst[2*plane+o] = float32(bb/255*2 - 1)
		}
	}
}

func dist(a, b Point) float64 { return math.Hypot(a.X-b.X, a.Y-b.Y) }
