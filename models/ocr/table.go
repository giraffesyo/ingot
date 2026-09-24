package ocr

import (
	"fmt"
	"image"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

// TableRecognizer runs PaddleOCR's SLANet-plus (ONNX, as published by
// RapidTable): a table image's HTML structure tokens plus one box per cell.
// Text comes from OCR over the same crop, matched to cells as RapidTable's
// TableMatch does (Recognize). Pre/post-processing follow RapidTable's
// pp_structure handler — the parity reference (tools/export/table.py).
type TableRecognizer struct {
	sess  graph.Runner
	in    string
	vocab []string // sos, dictionary (td merged), eos
}

const tableSize = 488 // SLANet-plus input side

// NewTableRecognizer loads SLANet-plus for a device (graph.CompileOn).
func NewTableRecognizer(path, device string) (*TableRecognizer, error) {
	m, err := onnx.DecodeFile(path)
	if err != nil {
		return nil, err
	}
	dict := strings.Split(strings.TrimRight(m.Metadata["character"], "\n"), "\n")
	if len(dict) < 4 {
		return nil, fmt.Errorf("table: %s has no structure vocabulary (metadata \"character\")", path)
	}
	// merge_no_span_structure: "<td></td>" is one token, "<td>" is not.
	if !slices.Contains(dict, "<td></td>") {
		dict = append(dict, "<td></td>")
	}
	if i := slices.Index(dict, "<td>"); i >= 0 {
		dict = slices.Delete(dict, i, i+1)
	}
	vocab := append(append([]string{"sos"}, dict...), "eos")
	g, err := graph.FromONNX(m)
	if err != nil {
		return nil, err
	}
	s, err := graph.CompileOn(g, device)
	if err != nil {
		return nil, err
	}
	if len(g.Inputs) != 1 {
		return nil, fmt.Errorf("table: want one image input, got %d", len(g.Inputs))
	}
	return &TableRecognizer{sess: s, in: g.Inputs[0].Name, vocab: vocab}, nil
}

// TableStructure is SLANet's output for one table image: HTML structure
// tokens (wrapped in <html><body><table>), one quadrilateral per cell in
// image pixels (x1 y1 x2 y2 x3 y3 x4 y4, cell order = td order), and the
// mean token confidence.
type TableStructure struct {
	Tokens []string
	Cells  [][8]float64
	Score  float64
}

// Structure recognises the table's structure.
func (r *TableRecognizer) Structure(img image.Image) (*TableStructure, error) {
	b := img.Bounds()
	W, H := b.Dx(), b.Dy()
	ratio := float64(tableSize) / float64(max(W, H))
	rw, rh := int(float64(W)*ratio), int(float64(H)*ratio)
	rgb := resizeBilinear(img, rw, rh)
	x := tensor.New(tensor.F32, 1, 3, tableSize, tableSize) // zero pad after normalising
	f, plane := x.F32(), tableSize*tableSize
	// ImageNet mean/std applied to cv2's BGR channel order, as RapidTable does.
	mean := [3]float32{0.485, 0.456, 0.406}
	std := [3]float32{0.229, 0.224, 0.225}
	for y := 0; y < rh; y++ {
		for xx := 0; xx < rw; xx++ {
			i := (y*rw + xx) * 3
			o := y*tableSize + xx
			bgr := [3]uint8{rgb[i+2], rgb[i+1], rgb[i]}
			for c := range 3 {
				f[c*plane+o] = (float32(bgr[c])/255 - mean[c]) / std[c]
			}
		}
	}
	outs, err := r.sess.Run(map[string]*tensor.Tensor{r.in: x})
	if err != nil {
		return nil, err
	}
	defer r.sess.Release(outs)
	var locT, probT *tensor.Tensor
	for _, t := range outs {
		s := t.Shape()
		switch {
		case len(s) == 3 && s[2] == 8:
			locT = t
		case len(s) == 3 && s[2] == len(r.vocab):
			probT = t
		}
	}
	if locT == nil || probT == nil {
		return nil, fmt.Errorf("table: outputs are not [1,L,8] and [1,L,%d]", len(r.vocab))
	}
	return r.decode(locT.F32(), probT.F32(), probT.Shape()[1], W, H), nil
}

// decode is TableLabelDecode + SLANet-plus's cell rescale.
func (r *TableRecognizer) decode(loc, prob []float32, L, W, H int) *TableStructure {
	V := len(r.vocab)
	eos := V - 1
	st := &TableStructure{Tokens: []string{"<html>", "<body>", "<table>"}}
	var scores float64
	var n int
	ratio := min(float64(tableSize)/float64(H), float64(tableSize)/float64(W))
	wr, hr := float64(tableSize)/(float64(W)*ratio), float64(tableSize)/(float64(H)*ratio)
	for i := 0; i < L; i++ {
		row := prob[i*V : (i+1)*V]
		best := 0
		for k, v := range row {
			if v > row[best] {
				best = k
			}
		}
		if i > 0 && best == eos {
			break
		}
		if best == 0 || best == eos {
			continue
		}
		tok := r.vocab[best]
		if tok == "<td>" || tok == "<td" || tok == "<td></td>" {
			var c [8]float64
			for k := range 8 {
				v := float64(loc[i*8+k])
				if k%2 == 0 {
					c[k] = v * float64(W) * wr
				} else {
					c[k] = v * float64(H) * hr
				}
			}
			if c != ([8]float64{}) { // blank placeholder boxes are dropped
				st.Cells = append(st.Cells, c)
			}
		}
		st.Tokens = append(st.Tokens, tok)
		scores += float64(row[best])
		n++
	}
	st.Tokens = append(st.Tokens, "</table>", "</body>", "</html>")
	if n > 0 {
		st.Score = scores / float64(n)
	}
	return st
}

// TableCell is one cell of a recognised table.
type TableCell struct {
	Text             string
	RowSpan, ColSpan int
	Box              [4]float64 // x1, y1, x2, y2 in the table image
}

// Table is a recognised table: rows of cells (spans as in HTML) and the
// equivalent HTML.
type Table struct {
	Rows [][]TableCell
	HTML string
}

// Recognize reads the table: structure, then the OCR results over the same
// image (ocr: text boxes in its pixels, e.g. Pipeline.Run) matched to cells.
func (r *TableRecognizer) Recognize(img image.Image, ocr []Result) (*Table, error) {
	st, err := r.Structure(img)
	if err != nil {
		return nil, err
	}
	return buildTable(st, ocr), nil
}

// buildTable is RapidTable's TableMatch: each OCR box goes to the cell with
// the highest IoU (ties by L1 corner distance); a cell's texts join with
// spaces in OCR order.
func buildTable(st *TableStructure, ocr []Result) *Table {
	cells := make([][4]float64, len(st.Cells))
	for i, c := range st.Cells {
		cells[i] = [4]float64{
			min(c[0], c[2], c[4], c[6]), min(c[1], c[3], c[5], c[7]),
			max(c[0], c[2], c[4], c[6]), max(c[1], c[3], c[5], c[7]),
		}
	}
	matched := map[int][]string{}
	if len(cells) > 0 {
		top := math.Inf(1)
		for _, c := range cells {
			top = min(top, c[1])
		}
		for _, o := range ocr {
			x0, y0, x1, y1 := boxBounds(o.Box)
			if y1 < top { // above the table (a caption)
				continue
			}
			ob := [4]float64{max(0, x0), max(0, y0), x1, y1}
			best, bestIoU, bestDist := -1, 0.0, math.Inf(1)
			for j, c := range cells {
				iou := iouRect(ob, c)
				d := l1Dist(ob, c)
				if best < 0 || iou > bestIoU || iou == bestIoU && d < bestDist {
					best, bestIoU, bestDist = j, iou, d
				}
			}
			if bestIoU > 1e-8 {
				matched[best] = append(matched[best], strings.TrimSpace(o.Text))
			}
		}
	}
	t := &Table{}
	var html strings.Builder
	td := 0
	toks := st.Tokens
	for i := 0; i < len(toks); i++ {
		tok := toks[i]
		switch {
		case tok == "<thead>" || tok == "</thead>" || tok == "<tbody>" || tok == "</tbody>":
			continue
		case tok == "<tr>":
			t.Rows = append(t.Rows, nil)
			html.WriteString(tok)
		case tok == "<td></td>" || tok == "<td":
			cell := TableCell{RowSpan: 1, ColSpan: 1}
			html.WriteString("<td")
			if tok == "<td" { // attribute tokens until ">"
				for i+1 < len(toks) && toks[i+1] != ">" {
					i++
					a := toks[i]
					html.WriteString(a)
					if k, v, ok := strings.Cut(strings.TrimSpace(a), "="); ok {
						n, _ := strconv.Atoi(strings.Trim(v, "\"'"))
						if k == "colspan" {
							cell.ColSpan = max(n, 1)
						} else if k == "rowspan" {
							cell.RowSpan = max(n, 1)
						}
					}
				}
				i++ // ">"
				if i+1 < len(toks) && toks[i+1] == "</td>" {
					i++
				}
			}
			cell.Text = strings.Join(matched[td], " ")
			if td < len(cells) {
				cell.Box = cells[td]
			}
			html.WriteString(">" + cell.Text + "</td>")
			if len(t.Rows) == 0 {
				t.Rows = append(t.Rows, nil)
			}
			t.Rows[len(t.Rows)-1] = append(t.Rows[len(t.Rows)-1], cell)
			td++
		default:
			html.WriteString(tok)
		}
	}
	t.HTML = html.String()
	return t
}

func boxBounds(b Box) (x0, y0, x1, y1 float64) {
	x0, y0, x1, y1 = b.Pts[0].X, b.Pts[0].Y, b.Pts[0].X, b.Pts[0].Y
	for _, p := range b.Pts[1:] {
		x0, y0, x1, y1 = min(x0, p.X), min(y0, p.Y), max(x1, p.X), max(y1, p.Y)
	}
	return
}

func iouRect(a, b [4]float64) float64 {
	iw := min(a[2], b[2]) - max(a[0], b[0])
	ih := min(a[3], b[3]) - max(a[1], b[1])
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
}

func l1Dist(a, b [4]float64) float64 {
	d2 := math.Abs(b[0]-a[0]) + math.Abs(b[1]-a[1])
	d3 := math.Abs(b[2]-a[2]) + math.Abs(b[3]-a[3])
	return d2 + d3 + min(d2, d3)
}
