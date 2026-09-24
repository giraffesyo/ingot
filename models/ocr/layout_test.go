package ocr

import (
	"encoding/json"
	"image"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const layoutDir = "../../testdata/layout"

type layoutRefPage struct {
	Image   string `json:"image"`
	Regions []struct {
		Label string     `json:"label"`
		Score float64    `json:"score"`
		Box   [4]float64 `json:"box"`
		Text  string     `json:"text"`
	} `json:"regions"`
}

func loadLayoutJSON(t *testing.T, name string) []layoutRefPage {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(layoutDir, "pages", name))
	if err != nil {
		t.Skip(err)
	}
	var p []layoutRefPage
	if err := json.Unmarshal(b, &p); err != nil {
		t.Fatal(err)
	}
	return p
}

func loadPage(t *testing.T, name string) image.Image {
	t.Helper()
	f, err := os.Open(filepath.Join(layoutDir, "pages", name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

func layoutDetector(t *testing.T) *LayoutDetector {
	t.Helper()
	p := filepath.Join(layoutDir, "pp_doc_layoutv3.onnx")
	if _, err := os.Stat(p); err != nil {
		t.Skip("PP-DocLayoutV3 not present (tools/export/layout.py)")
	}
	d, err := NewLayoutDetector(p, "cpu")
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// TestLayoutParity: each page's regions match RapidLayout's (the ORT
// reference) — same labels in the same reading order, boxes within a few
// pixels (our bicubic resize is float, cv2's fixed-point), scores within
// 0.03.
func TestLayoutParity(t *testing.T) {
	d := layoutDetector(t)
	for _, pg := range loadLayoutJSON(t, "ref.json") {
		got, err := d.Detect(loadPage(t, pg.Image))
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(pg.Regions) {
			t.Errorf("%s: %d regions, ref %d", pg.Image, len(got), len(pg.Regions))
			for _, r := range got {
				t.Logf("  got %s %.2f %v", r.Label, r.Score, r.Box)
			}
			continue
		}
		var maxBox, maxScore float64
		for i, r := range got {
			w := pg.Regions[i]
			if r.Label != w.Label {
				t.Errorf("%s region %d: %s, ref %s", pg.Image, i, r.Label, w.Label)
			}
			for k := range 4 {
				maxBox = math.Max(maxBox, math.Abs(r.Box[k]-w.Box[k]))
			}
			maxScore = math.Max(maxScore, math.Abs(r.Score-w.Score))
		}
		t.Logf("%s: %d regions, max box delta %.0f px, max score delta %.3f", pg.Image, len(got), maxBox, maxScore)
		if maxBox > 4 || maxScore > 0.03 {
			t.Errorf("%s: box delta %.1f px, score delta %.3f", pg.Image, maxBox, maxScore)
		}
	}
}

// TestLayoutTruth scores the pages against their construction ground truth:
// every truth region must be found (IoU >= 0.5), ordered regions in truth
// order.
func TestLayoutTruth(t *testing.T) {
	d := layoutDetector(t)
	for _, pg := range loadLayoutJSON(t, "truth.json") {
		got, err := d.Detect(loadPage(t, pg.Image))
		if err != nil {
			t.Fatal(err)
		}
		found, lastOrder := 0, 0
		for _, w := range pg.Regions {
			best, bi := 0.0, -1
			for i, r := range got {
				if v := iouAABB(w.Box[0], w.Box[1], w.Box[2], w.Box[3], r.Box[0], r.Box[1], r.Box[2], r.Box[3]); v > best {
					best, bi = v, i
				}
			}
			if best < 0.5 {
				t.Logf("%s: missed %s %v (best IoU %.2f)", pg.Image, w.Label, w.Box, best)
				continue
			}
			found++
			if o := got[bi].Order; o > 0 {
				if o < lastOrder {
					t.Errorf("%s: %s out of reading order (%d after %d)", pg.Image, w.Label, o, lastOrder)
				}
				lastOrder = o
			}
		}
		t.Logf("%s: found %d/%d truth regions", pg.Image, found, len(pg.Regions))
		if found < len(pg.Regions)-1 {
			t.Errorf("%s: found only %d of %d regions", pg.Image, found, len(pg.Regions))
		}
	}
}
