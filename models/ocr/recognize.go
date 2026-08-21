package ocr

import (
	"bufio"
	"fmt"
	"image"
	"math"
	"os"
	"strings"

	"github.com/giraffesyo/ocr/graph"
	"github.com/giraffesyo/ocr/onnx"
	"github.com/giraffesyo/ocr/tensor"
)

// Recognizer runs a CRNN/SVTR-style text recogniser with CTC decoding.
type Recognizer struct {
	sess    *graph.Session
	inName  string
	outName string
	dict    []string // index i -> character; index 0 is CTC blank
	height  int      // model input height (48 for PP-OCRv4)
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

// Recognize crops the box from img, runs recognition, and returns the decoded
// text and mean confidence.
func (r *Recognizer) Recognize(img image.Image, box Box) (string, float64, error) {
	in := cropToTensor(img, box, r.height)
	outs, err := r.sess.Run(map[string]*tensor.Tensor{r.inName: in})
	if err != nil {
		return "", 0, err
	}
	out := outs[r.outName]
	os := out.Shape()
	if os.Rank() != 3 {
		return "", 0, fmt.Errorf("recognizer: unexpected output %v", os)
	}
	return r.ctcDecode(out.F32(), os[1], os[2])
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

// cropToTensor perspective-crops the quad from img to a height-H strip and
// returns a normalised [1,3,H,W] tensor (RGB, (x/255-0.5)/0.5).
func cropToTensor(img image.Image, box Box, H int) *tensor.Tensor {
	p := box.Pts
	wTop := dist(p[0], p[1])
	wBot := dist(p[3], p[2])
	hL := dist(p[0], p[3])
	hR := dist(p[1], p[2])
	cw := math.Max(wTop, wBot)
	ch := math.Max(hL, hR)
	W := int(math.Max(4, math.Round(float64(H)*cw/math.Max(ch, 1))))
	t := tensor.New(tensor.F32, 1, 3, H, W)
	f := t.F32()
	plane := H * W
	b := img.Bounds()
	sample := func(x, y float64) (float64, float64, float64) {
		xi := clampI(int(x+0.5), 0, b.Dx()-1)
		yi := clampI(int(y+0.5), 0, b.Dy()-1)
		r, g, bl, _ := img.At(b.Min.X+xi, b.Min.Y+yi).RGBA()
		return float64(r >> 8), float64(g >> 8), float64(bl >> 8)
	}
	for oy := 0; oy < H; oy++ {
		v := (float64(oy) + 0.5) / float64(H)
		for ox := 0; ox < W; ox++ {
			u := (float64(ox) + 0.5) / float64(W)
			// bilinear blend of the four corners (exact for parallelograms)
			topx := p[0].X*(1-u) + p[1].X*u
			topy := p[0].Y*(1-u) + p[1].Y*u
			botx := p[3].X*(1-u) + p[2].X*u
			boty := p[3].Y*(1-u) + p[2].Y*u
			sx := topx*(1-v) + botx*v
			sy := topy*(1-v) + boty*v
			rr, gg, bb := sample(sx, sy)
			o := oy*W + ox
			f[o] = float32(rr/255*2 - 1)
			f[plane+o] = float32(gg/255*2 - 1)
			f[2*plane+o] = float32(bb/255*2 - 1)
		}
	}
	return t
}

func dist(a, b Point) float64 { return math.Hypot(a.X-b.X, a.Y-b.Y) }
