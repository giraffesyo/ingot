// Package ocrcmd is ingot ocr: the OCR pipeline on an image — text
// detection (DBNet) and recognition, drawing detected boxes to an output PNG.
// With --format md or json it reads the page as a document instead — layout
// regions in reading order (PP-DocLayoutV3), their text, and tables
// (SLANet-plus).
//
//	ingot ocr --det testdata/ocr/det.onnx --in image.png --out boxes.png
//	ingot ocr --in page.png --format md
package ocrcmd

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"math"
	"os"
	"sort"

	"github.com/spf13/cobra"

	"github.com/giraffesyo/ingot/models/ocr"
)

type options struct {
	det, rec, dict, parseq, charset, in, out string
	device, format, layout, table            string
	boxThr                                   float64
	norec                                    bool
}

// Command returns the ocr subcommand.
func Command() *cobra.Command {
	var o options
	cmd := &cobra.Command{
		Use:   "ocr",
		Short: "Detect and recognize text in an image, or read a page as a document",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return run(o)
		},
	}
	f := cmd.Flags()
	f.StringVar(&o.det, "det", "testdata/ocr/det.onnx", "detection model path")
	f.StringVar(&o.rec, "rec", "testdata/ocr/rec.onnx", "recognition model path")
	f.StringVar(&o.dict, "dict", "testdata/ocr/rec_dict.txt", "recognition char dictionary")
	f.StringVar(&o.parseq, "parseq", "", "use a PARSeq recognizer (ONNX path) instead of --rec; word-level, 94-char ASCII")
	f.StringVar(&o.charset, "charset", "testdata/models/parseq_charset.txt", "PARSeq charset file (with --parseq)")
	f.StringVar(&o.in, "in", "testdata/ocr/sample.png", "input image")
	f.StringVar(&o.out, "out", "det_boxes.png", "annotated output PNG")
	f.Float64Var(&o.boxThr, "boxthr", 0.6, "box score threshold")
	f.BoolVar(&o.norec, "norec", false, "detection only")
	f.StringVar(&o.device, "device", "cpu", "device for every model: cpu, gpu (Metal), gpu-bf16 (faster, bf16 weight products) or auto")
	f.StringVar(&o.format, "format", "lines", "output: lines (boxes + text), md or json (document: layout, reading order, tables)")
	f.StringVar(&o.layout, "layout", "testdata/layout/pp_doc_layoutv3.onnx", "layout model (PP-DocLayoutV3) for --format md|json")
	f.StringVar(&o.table, "table", "testdata/layout/slanet-plus.onnx", "table structure model (SLANet-plus) for --format md|json; empty disables")
	return cmd
}

func run(o options) error {
	if o.format == "md" || o.format == "json" {
		return document(o.in, o.det, o.rec, o.dict, o.layout, o.table, o.device, o.format)
	}
	if o.format != "lines" {
		return fmt.Errorf("unknown --format %q (lines, md, json)", o.format)
	}

	img, err := loadImage(o.in)
	if err != nil {
		return fmt.Errorf("load: %w", err)
	}
	d, err := ocr.NewDetectorOn(o.det, o.device)
	if err != nil {
		return fmt.Errorf("detector: %w", err)
	}
	d.BoxThresh = o.boxThr
	boxes, err := d.Detect(img)
	if err != nil {
		return fmt.Errorf("detect: %w", err)
	}
	fmt.Printf("detected %d text boxes\n", len(boxes))
	sortBoxesTopToBottom(boxes)
	var recog ocr.BoxRecognizer
	if !o.norec && o.parseq != "" {
		pr, err := ocr.NewParseqOn(o.parseq, o.charset, o.device)
		if err != nil {
			return fmt.Errorf("parseq: %w", err)
		}
		recog = pr
	} else if !o.norec {
		r, err := ocr.NewRecognizerOn(o.rec, o.dict, o.device)
		if err != nil {
			return fmt.Errorf("recognizer: %w", err)
		}
		recog = r
	}
	if recog != nil {
		// One batched forward per group of similar-width boxes (see ocr.Pipeline).
		p := &ocr.Pipeline{Det: d, Rec: recog, RecBatch: ocr.DefaultRecBatch, RecPadRatio: ocr.DefaultRecPadRatio}
		texts, confs, err := p.RecognizeBoxes(img, boxes)
		if err != nil {
			return fmt.Errorf("recognize: %w", err)
		}
		for i, b := range boxes {
			fmt.Printf("  box %d  det=%.2f rec=%.2f  %q\n", i, b.Score, confs[i], texts[i])
		}
	} else {
		for i, b := range boxes {
			fmt.Printf("  box %d score=%.2f pts=%v\n", i, b.Score, b.Pts)
		}
	}
	if err := drawBoxes(img, boxes, o.out); err != nil {
		return fmt.Errorf("draw: %w", err)
	}
	fmt.Println("wrote", o.out)
	return nil
}

// document reads the page as a structured document and prints it.
func document(in, det, rec, dict, layout, table, device, format string) error {
	img, err := loadImage(in)
	if err != nil {
		return fmt.Errorf("load: %w", err)
	}
	p, err := ocr.NewPipelineOn(det, rec, dict, device, device)
	if err != nil {
		return fmt.Errorf("ocr models: %w", err)
	}
	l, err := ocr.NewLayoutDetector(layout, device)
	if err != nil {
		return fmt.Errorf("layout model: %w", err)
	}
	dp := &ocr.DocPipeline{OCR: p, Layout: l}
	if table != "" {
		if dp.Tables, err = ocr.NewTableRecognizer(table, device); err != nil {
			return fmt.Errorf("table model: %w", err)
		}
	}
	doc, err := dp.Run(img)
	if err != nil {
		return err
	}
	if format == "md" {
		fmt.Print(doc.Markdown())
		return nil
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(doc)
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
