package ocr

import (
	"encoding/json"
	"fmt"
	"image"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

type gtLine struct {
	Text string       `json:"text"`
	Quad [][2]float64 `json:"quad"`
}
type gtImage struct {
	Image string   `json:"image"`
	W     int      `json:"w"`
	H     int      `json:"h"`
	Lines []gtLine `json:"lines"`
}

func quadBBoxG(q [][2]float64) (x0, y0, x1, y1 float64) {
	x0, y0 = math.Inf(1), math.Inf(1)
	x1, y1 = math.Inf(-1), math.Inf(-1)
	for _, p := range q {
		x0, y0 = math.Min(x0, p[0]), math.Min(y0, p[1])
		x1, y1 = math.Max(x1, p[0]), math.Max(y1, p[1])
	}
	return
}
func boxBBox(b Box) (x0, y0, x1, y1 float64) {
	x0, y0 = math.Inf(1), math.Inf(1)
	x1, y1 = math.Inf(-1), math.Inf(-1)
	for _, p := range b.Pts {
		x0, y0 = math.Min(x0, p.X), math.Min(y0, p.Y)
		x1, y1 = math.Max(x1, p.X), math.Max(y1, p.Y)
	}
	return
}
func iouAABB(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1 float64) float64 {
	ix0, iy0 := math.Max(ax0, bx0), math.Max(ay0, by0)
	ix1, iy1 := math.Min(ax1, bx1), math.Min(ay1, by1)
	iw, ih := ix1-ix0, iy1-iy0
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	a := (ax1 - ax0) * (ay1 - ay0)
	b := (bx1 - bx0) * (by1 - by0)
	return inter / (a + b - inter)
}

func levenshtein(a, b string) int {
	ar, br := []rune(a), []rune(b)
	m, n := len(ar), len(br)
	if m == 0 {
		return n
	}
	if n == 0 {
		return m
	}
	prev := make([]int, n+1)
	cur := make([]int, n+1)
	for j := 0; j <= n; j++ {
		prev[j] = j
	}
	for i := 1; i <= m; i++ {
		cur[0] = i
		for j := 1; j <= n; j++ {
			cost := 1
			if ar[i-1] == br[j-1] {
				cost = 0
			}
			cur[j] = min(cur[j-1]+1, min(prev[j]+1, prev[j-1]+cost))
		}
		prev, cur = cur, prev
	}
	return prev[n]
}

func normText(s string) string { return strings.TrimSpace(s) }

func TestCorpus(t *testing.T) {
	const dir = "../../testdata/ocr"
	mb, err := os.ReadFile(dir + "/corpus/manifest.json")
	if err != nil {
		t.Skip("no corpus (run tools/export/corpus.py)")
	}
	var corpus []gtImage
	if err := json.Unmarshal(mb, &corpus); err != nil {
		t.Fatal(err)
	}
	detPath, recPath := dir+"/det.onnx", dir+"/rec.onnx"
	if v := os.Getenv("OCR_DET_ONNX"); v != "" {
		detPath = v
	}
	if v := os.Getenv("OCR_REC_ONNX"); v != "" {
		recPath = v
	}
	p, err := NewPipeline(detPath, recPath, dir+"/rec_dict.txt")
	if v := os.Getenv("OCR_REC_BATCH"); v != "" {
		fmt.Sscan(v, &p.RecBatch)
	}
	if v := os.Getenv("OCR_REC_PAD_RATIO"); v != "" {
		fmt.Sscan(v, &p.RecPadRatio)
	}
	if err != nil {
		t.Skip(err)
	}

	var tp, fp, fn int
	var matched, exact int
	var cerNum, cerDen int
	const iouThr = 0.5

	for _, gi := range corpus {
		f, err := os.Open(filepath.Join(dir, "corpus", gi.Image))
		if err != nil {
			t.Fatal(err)
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		res, err := p.Run(img)
		if err != nil {
			t.Fatalf("%s: %v", gi.Image, err)
		}
		usedPred := make([]bool, len(res))
		usedGT := make([]bool, len(gi.Lines))
		type pair struct {
			g, r int
			iou  float64
		}
		var pairs []pair
		for g, gl := range gi.Lines {
			gx0, gy0, gx1, gy1 := quadBBoxG(gl.Quad)
			for r, rb := range res {
				bx0, by0, bx1, by1 := boxBBox(rb.Box)
				if v := iouAABB(gx0, gy0, gx1, gy1, bx0, by0, bx1, by1); v >= iouThr {
					pairs = append(pairs, pair{g, r, v})
				}
			}
		}
		sort.Slice(pairs, func(i, j int) bool { return pairs[i].iou > pairs[j].iou })
		for _, pr := range pairs {
			if usedGT[pr.g] || usedPred[pr.r] {
				continue
			}
			usedGT[pr.g] = true
			usedPred[pr.r] = true
			tp++
			matched++
			gtText := normText(gi.Lines[pr.g].Text)
			predText := normText(res[pr.r].Text)
			if gtText == predText {
				exact++
			} else if os.Getenv("OCR_DUMP") != "" {
				t.Logf("MISMATCH %s gt=%q pred=%q", gi.Image, gtText, predText)
			}
			cerNum += levenshtein(gtText, predText)
			cerDen += len([]rune(gtText))
		}
		for gi2, u := range usedGT {
			if !u {
				fn++
				if os.Getenv("OCR_DUMP") != "" {
					t.Logf("FN %s missed gt=%q", gi.Image, gi.Lines[gi2].Text)
				}
			}
		}
		for ri, u := range usedPred {
			if !u {
				fp++
				if os.Getenv("OCR_DUMP") != "" {
					t.Logf("FP %s spurious pred=%q", gi.Image, res[ri].Text)
				}
			}
		}
	}
	prec := float64(tp) / math.Max(1, float64(tp+fp))
	rec := float64(tp) / math.Max(1, float64(tp+fn))
	f1 := 2 * prec * rec / math.Max(1e-9, prec+rec)
	exactRate := float64(exact) / math.Max(1, float64(matched))
	cer := float64(cerNum) / math.Max(1, float64(cerDen))
	t.Logf("DETECTION: precision %.3f recall %.3f F1 %.3f (tp=%d fp=%d fn=%d)", prec, rec, f1, tp, fp, fn)
	t.Logf("RECOGNITION (matched boxes): exact-match %.3f, char-acc %.3f, CER %.3f (%d edits / %d chars)", exactRate, 1-cer, cer, cerNum, cerDen)
	t.Logf("  matched %d lines; exact %d", matched, exact)

	// Regression gates, set from the 2026-08-21 baseline (F1 0.917, exact 0.919,
	// CER 0.013) with margin. Tighten as the pipeline improves.
	if rec < 0.85 {
		t.Errorf("detection recall %.3f < 0.85 (regression)", rec)
	}
	if f1 < 0.85 {
		t.Errorf("detection F1 %.3f < 0.85 (regression)", f1)
	}
	if cer > 0.05 {
		t.Errorf("char error rate %.3f > 0.05 (regression)", cer)
	}
	if exactRate < 0.85 {
		t.Errorf("exact-match %.3f < 0.85 (regression)", exactRate)
	}
}
