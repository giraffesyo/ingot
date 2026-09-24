package ocr

import (
	"fmt"
	"image"
	"math"
	"slices"
	"sort"
	"strings"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

// Region is one layout element of a page: its class, confidence, box in
// original-image pixels, and 1-based reading order (0: outside the reading
// flow — figures, tables, headers/footers, see unorderedLabels).
type Region struct {
	Label string
	Score float64
	Box   [4]float64 // x1, y1, x2, y2
	Order int
}

// LayoutDetector runs PaddleOCR's PP-DocLayoutV3 (ONNX, as published by
// RapidLayout): page regions in 25 classes with a predicted reading order.
// Pre- and post-processing follow RapidLayout's pp_doc_layout handler — the
// parity reference (tools/export/layout.py).
type LayoutDetector struct {
	sess   graph.Runner
	labels []string
	// Thresh is the minimum score to keep a region (RapidLayout: 0.5).
	Thresh float64
}

const layoutSize = 800 // the model's square input

// unorderedLabels are regions outside the reading flow (RapidLayout's
// SKIP_ORDER_LABELS).
var unorderedLabels = map[string]bool{
	"figure_title": true, "vision_footnote": true, "image": true, "chart": true, "table": true,
	"header": true, "header_image": true, "footer": true, "footer_image": true, "footnote": true,
	"aside_text": true,
}

// NewLayoutDetector loads PP-DocLayoutV3 for a device (graph.CompileOn).
func NewLayoutDetector(path, device string) (*LayoutDetector, error) {
	m, err := onnx.DecodeFile(path)
	if err != nil {
		return nil, err
	}
	labels := strings.Split(strings.TrimSpace(m.Metadata["character"]), "\n")
	if len(labels) < 2 {
		return nil, fmt.Errorf("layout: %s has no class labels (metadata \"character\")", path)
	}
	g, err := graph.FromONNX(m)
	if err != nil {
		return nil, err
	}
	names := map[string]bool{}
	for _, v := range g.Inputs {
		names[v.Name] = true
	}
	if !names["image"] || !names["im_shape"] || !names["scale_factor"] {
		return nil, fmt.Errorf("layout: want inputs image, im_shape, scale_factor")
	}
	s, err := graph.CompileOn(g, device)
	if err != nil {
		return nil, err
	}
	return &LayoutDetector{sess: s, labels: labels, Thresh: 0.5}, nil
}

// Detect returns the page's regions in reading order (ordered regions by
// Order, then the unordered ones in the model's order).
func (d *LayoutDetector) Detect(img image.Image) ([]Region, error) {
	b := img.Bounds()
	W, H := b.Dx(), b.Dy()
	x := tensor.New(tensor.F32, 1, 3, layoutSize, layoutSize)
	rgb := resizeBicubic(img, layoutSize, layoutSize)
	f, plane := x.F32(), layoutSize*layoutSize
	for i := 0; i < plane; i++ { // BGR planes, /255 (RapidLayout feeds cv2's BGR)
		f[i] = float32(rgb[i*3+2]) / 255
		f[plane+i] = float32(rgb[i*3+1]) / 255
		f[2*plane+i] = float32(rgb[i*3]) / 255
	}
	feeds := map[string]*tensor.Tensor{
		"image":        x,
		"im_shape":     tensor.FromF32([]float32{layoutSize, layoutSize}, 1, 2),
		"scale_factor": tensor.FromF32([]float32{float32(layoutSize) / float32(H), float32(layoutSize) / float32(W)}, 1, 2),
	}
	outs, err := d.sess.Run(feeds)
	if err != nil {
		return nil, err
	}
	defer d.sess.Release(outs)
	var rows *tensor.Tensor
	for _, t := range outs {
		if s := t.Shape(); len(s) == 2 && (s[1] == 7 || s[1] == 6) && t.DType() == tensor.F32 {
			rows = t
		}
	}
	if rows == nil {
		return nil, fmt.Errorf("layout: no [N,7] detection output")
	}
	return d.post(rows.F32(), rows.Shape()[1], W, H), nil
}

type layoutBox struct {
	cls   int
	score float64
	box   [4]float64
	order float64
}

// post is RapidLayout's pp_doc_layout post-processing in rect mode.
func (d *LayoutDetector) post(v []float32, cols, W, H int) []Region {
	var boxes []layoutBox
	for r := 0; r+cols <= len(v); r += cols {
		b := layoutBox{cls: int(v[r]), score: float64(v[r+1])}
		for k := range 4 {
			b.box[k] = math.RoundToEven(float64(v[r+2+k]))
		}
		if cols == 7 {
			b.order = float64(v[r+6])
		}
		if b.score > d.Thresh && b.cls > -1 && b.cls < len(d.labels) {
			boxes = append(boxes, b)
		}
	}
	boxes = layoutNMS(boxes, 0.6, 0.98)
	// Drop "image" regions covering (nearly) the whole page.
	if len(boxes) > 1 {
		thres := 0.93
		if W > H {
			thres = 0.82
		}
		img := slices.Index(d.labels, "image")
		var keep []layoutBox
		for _, b := range boxes {
			if b.cls == img {
				x1, y1 := max(0, b.box[0]), max(0, b.box[1])
				x2, y2 := min(float64(W), b.box[2]), min(float64(H), b.box[3])
				if (x2-x1)*(y2-y1) > thres*float64(W*H) {
					continue
				}
			}
			keep = append(keep, b)
		}
		if len(keep) > 0 {
			boxes = keep
		}
	}
	if cols == 7 {
		sort.SliceStable(boxes, func(i, j int) bool { return boxes[i].order < boxes[j].order })
	}
	// Clip to the page (integer pixels), drop degenerate boxes.
	var regs []Region
	for _, b := range boxes {
		x1, y1 := math.Trunc(max(0, b.box[0])), math.Trunc(max(0, b.box[1]))
		x2, y2 := math.Trunc(min(float64(W), b.box[2])), math.Trunc(min(float64(H), b.box[3]))
		if x2 <= x1 || y2 <= y1 {
			continue
		}
		regs = append(regs, Region{Label: d.labels[b.cls], Score: b.score, Box: [4]float64{x1, y1, x2, y2}})
	}
	regs = filterOverlaps(regs)
	n := 1
	for i := range regs {
		if !unorderedLabels[regs[i].Label] {
			regs[i].Order = n
			n++
		}
	}
	return regs
}

// layoutNMS: greedy by score; a box suppresses same-class boxes above
// iouSame and other-class boxes above iouDiff (IoU with the +1 pixel
// convention). Returns the kept boxes in selection (score) order.
func layoutNMS(boxes []layoutBox, iouSame, iouDiff float64) []layoutBox {
	idx := make([]int, len(boxes))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return boxes[idx[a]].score > boxes[idx[b]].score })
	var out []layoutBox
	for len(idx) > 0 {
		cur := boxes[idx[0]]
		out = append(out, cur)
		var rest []int
		for _, i := range idx[1:] {
			thr := iouDiff
			if boxes[i].cls == cur.cls {
				thr = iouSame
			}
			if iouPlus1(cur.box, boxes[i].box) < thr {
				rest = append(rest, i)
			}
		}
		idx = rest
	}
	return out
}

func iouPlus1(a, b [4]float64) float64 {
	iw := max(0, min(a[2], b[2])-max(a[0], b[0])+1)
	ih := max(0, min(a[3], b[3])-max(a[1], b[1])+1)
	inter := iw * ih
	ua := (a[2]-a[0]+1)*(a[3]-a[1]+1) + (b[2]-b[0]+1)*(b[3]-b[1]+1) - inter
	return inter / ua
}

// filterOverlaps drops "reference" regions, regions under 6 px on a side,
// and the smaller of two regions overlapping by > 70% of the smaller one
// (except an image against a different class); inline formulas yield first.
func filterOverlaps(regs []Region) []Region {
	var boxes []Region
	for _, r := range regs {
		if r.Label != "reference" {
			boxes = append(boxes, r)
		}
	}
	drop := make([]bool, len(boxes))
	area := func(b [4]float64) float64 { return max(0, b[2]-b[0]) * max(0, b[3]-b[1]) }
	for i := range boxes {
		bi := boxes[i].Box
		if bi[2]-bi[0] < 6 || bi[3]-bi[1] < 6 {
			drop[i] = true
		}
		for j := i + 1; j < len(boxes); j++ {
			if drop[i] || drop[j] {
				continue
			}
			bj := boxes[j].Box
			iw := max(0, min(bi[2], bj[2])-max(bi[0], bj[0]))
			ih := max(0, min(bi[3], bj[3])-max(bi[1], bj[1]))
			small := min(area(bi), area(bj))
			ratio := 0.0
			if small > 0 {
				ratio = iw * ih / small
			}
			li, lj := boxes[i].Label, boxes[j].Label
			if li == "inline_formula" || lj == "inline_formula" {
				if ratio > 0.5 {
					drop[i] = drop[i] || li == "inline_formula"
					drop[j] = drop[j] || lj == "inline_formula"
					continue
				}
			}
			if ratio > 0.7 {
				if (li == "image" || lj == "image") && li != lj {
					continue
				}
				if area(bi) >= area(bj) {
					drop[j] = true
				} else {
					drop[i] = true
				}
			}
		}
	}
	var out []Region
	for i, r := range boxes {
		if !drop[i] {
			out = append(out, r)
		}
	}
	return out
}

// resizeBicubic resamples img to nw×nh packed RGB with OpenCV's INTER_CUBIC
// (Keys a = −0.75, half-pixel centres, replicated borders, rounded to u8).
func resizeBicubic(img image.Image, nw, nh int) []uint8 {
	src, W, H := toRGB(img)
	type taps struct {
		i [4]int
		w [4]float32
	}
	mk := func(n, in int) []taps {
		t := make([]taps, n)
		s := float64(in) / float64(n)
		for o := range t {
			f := (float64(o)+0.5)*s - 0.5
			i0 := int(math.Floor(f))
			x := f - float64(i0)
			const a = -0.75
			w := [4]float64{
				((a*(x+1)-5*a)*(x+1)+8*a)*(x+1) - 4*a,
				((a+2)*x-(a+3))*x*x + 1,
				((a+2)*(1-x)-(a+3))*(1-x)*(1-x) + 1,
			}
			w[3] = 1 - w[0] - w[1] - w[2]
			for k := range 4 {
				t[o].i[k] = clampI(i0-1+k, 0, in-1)
				t[o].w[k] = float32(w[k])
			}
		}
		return t
	}
	tx, ty := mk(nw, W), mk(nh, H)
	// Horizontal pass into f32 rows, then vertical.
	mid := make([]float32, H*nw*3)
	par.For(H, max(1, 4096/max(nw, 1)), func(y, _ int) {
		row := src[y*W*3:]
		dst := mid[y*nw*3:]
		for ox, t := range tx {
			for c := 0; c < 3; c++ {
				var s float32
				for k := range 4 {
					s += t.w[k] * float32(row[t.i[k]*3+c])
				}
				dst[ox*3+c] = s
			}
		}
	})
	out := make([]uint8, nw*nh*3)
	par.For(nh, max(1, 4096/max(nw, 1)), func(oy, _ int) {
		t := ty[oy]
		dst := out[oy*nw*3 : (oy+1)*nw*3]
		for i := range dst {
			var s float32
			for k := range 4 {
				s += t.w[k] * mid[t.i[k]*nw*3+i]
			}
			dst[i] = u8(float64(s))
		}
	})
	return out
}
