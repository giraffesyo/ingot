// Command ocr runs the OCR pipeline on an image. Currently: text detection
// (DBNet), drawing detected boxes to an output PNG.
//
//	ocr -det testdata/ocr/det.onnx -in image.png -out boxes.png
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"math"
	"os"
	"sort"

	"github.com/giraffesyo/ingot/models/ocr"
)

func main() {
	det := flag.String("det", "testdata/ocr/det.onnx", "detection model path")
	rec := flag.String("rec", "testdata/ocr/rec.onnx", "recognition model path")
	dict := flag.String("dict", "testdata/ocr/rec_dict.txt", "recognition char dictionary")
	parseq := flag.String("parseq", "", "use a PARSeq recognizer (ONNX path) instead of -rec; word-level, 94-char ASCII")
	charset := flag.String("charset", "testdata/models/parseq_charset.txt", "PARSeq charset file (with -parseq)")
	inPath := flag.String("in", "testdata/ocr/sample.png", "input image")
	outPath := flag.String("out", "det_boxes.png", "annotated output PNG")
	boxThr := flag.Float64("boxthr", 0.6, "box score threshold")
	norec := flag.Bool("norec", false, "detection only")
	flag.Parse()

	img, err := loadImage(*inPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	d, err := ocr.NewDetector(*det)
	if err != nil {
		fmt.Fprintln(os.Stderr, "detector:", err)
		os.Exit(1)
	}
	d.BoxThresh = *boxThr
	boxes, err := d.Detect(img)
	if err != nil {
		fmt.Fprintln(os.Stderr, "detect:", err)
		os.Exit(1)
	}
	fmt.Printf("detected %d text boxes\n", len(boxes))
	sortBoxesTopToBottom(boxes)
	var recog ocr.BoxRecognizer
	if !*norec && *parseq != "" {
		pr, perr := ocr.NewParseq(*parseq, *charset)
		if perr != nil {
			fmt.Fprintln(os.Stderr, "parseq:", perr)
			os.Exit(1)
		}
		recog = pr
	} else if !*norec {
		r, rerr := ocr.NewRecognizer(*rec, *dict)
		if rerr != nil {
			fmt.Fprintln(os.Stderr, "recognizer:", rerr)
			os.Exit(1)
		}
		recog = r
	}
	if recog != nil {
		// One batched forward per group of similar-width boxes (see ocr.Pipeline).
		p := &ocr.Pipeline{Det: d, Rec: recog, RecBatch: ocr.DefaultRecBatch, RecPadRatio: ocr.DefaultRecPadRatio}
		texts, confs, rerr := p.RecognizeBoxes(img, boxes)
		if rerr != nil {
			fmt.Fprintln(os.Stderr, "recognize:", rerr)
			os.Exit(1)
		}
		for i, b := range boxes {
			fmt.Printf("  box %d  det=%.2f rec=%.2f  %q\n", i, b.Score, confs[i], texts[i])
		}
	} else {
		for i, b := range boxes {
			fmt.Printf("  box %d score=%.2f pts=%v\n", i, b.Score, b.Pts)
		}
	}
	if err := drawBoxes(img, boxes, *outPath); err != nil {
		fmt.Fprintln(os.Stderr, "draw:", err)
		os.Exit(1)
	}
	fmt.Println("wrote", *outPath)
}

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func drawBoxes(img image.Image, boxes []ocr.Box, path string) error {
	b := img.Bounds()
	rgba := image.NewRGBA(b)
	draw.Draw(rgba, b, img, b.Min, draw.Src)
	red := color.RGBA{255, 0, 0, 255}
	for _, box := range boxes {
		for i := 0; i < 4; i++ {
			p, q := box.Pts[i], box.Pts[(i+1)%4]
			drawLine(rgba, int(p.X), int(p.Y), int(q.X), int(q.Y), red)
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, rgba)
}

// drawLine draws a Bresenham line.
func drawLine(img *image.RGBA, x0, y0, x1, y1 int, c color.Color) {
	dx := abs(x1 - x0)
	dy := -abs(y1 - y0)
	sx, sy := 1, 1
	if x0 > x1 {
		sx = -1
	}
	if y0 > y1 {
		sy = -1
	}
	err := dx + dy
	for {
		img.Set(x0, y0, c)
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

// sortBoxesTopToBottom orders boxes by their top y, then left x (reading order).
func sortBoxesTopToBottom(boxes []ocr.Box) {
	sort.Slice(boxes, func(i, j int) bool {
		yi := (boxes[i].Pts[0].Y + boxes[i].Pts[1].Y) / 2
		yj := (boxes[j].Pts[0].Y + boxes[j].Pts[1].Y) / 2
		if math.Abs(yi-yj) > 10 {
			return yi < yj
		}
		return boxes[i].Pts[0].X < boxes[j].Pts[0].X
	})
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
