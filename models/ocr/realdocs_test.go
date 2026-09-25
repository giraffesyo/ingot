package ocr

import (
	"encoding/json"
	"image"
	_ "image/jpeg"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const realDir = "../../testdata/realdocs"

// TestRealDocs runs ingot over the real-document sample
// (tools/export/realdocs.py fetch) and writes its predictions to
// testdata/realdocs/ingot.json for realdocs.py score: layout regions per
// page, table HTML (ingot's OCR + TableRecognizer), and text CER per
// ground-truth region. Opt-in (INGOT_REALDOCS=1): it is an evaluation, not
// a gate.
func TestRealDocs(t *testing.T) {
	if os.Getenv("INGOT_REALDOCS") == "" {
		t.Skip("set INGOT_REALDOCS=1 (needs tools/export/realdocs.py fetch)")
	}
	var pages []struct {
		Image, Category string
		Regions         []struct {
			Label string
			Box   [4]float64
			Text  string
		}
	}
	var tables []struct{ Image, HTML string }
	readJSON(t, filepath.Join(realDir, "doclaynet", "gt.json"), &pages)
	readJSON(t, filepath.Join(realDir, "pubtabnet", "gt.json"), &tables)
	ocr := pipelineOn(t, "cpu", "cpu")
	lay, tab := layoutDetector(t), tableRecognizer(t)
	type region struct {
		Label string     `json:"label"`
		Score float64    `json:"score"`
		Box   [4]float64 `json:"box"`
	}
	out := struct {
		Pages   []map[string]any `json:"pages"`
		Tables  []map[string]any `json:"tables"`
		TextCER float64          `json:"text_cer"`
	}{}
	var edits, chars int
	for _, pg := range pages {
		img := loadImageFile(t, filepath.Join(realDir, "doclaynet", pg.Image))
		regs, err := lay.Detect(img)
		if err != nil {
			t.Fatal(err)
		}
		var rs []region
		for _, r := range regs {
			rs = append(rs, region{r.Label, r.Score, r.Box})
		}
		out.Pages = append(out.Pages, map[string]any{"image": pg.Image, "regions": rs})
		lines, err := ocr.Run(img)
		if err != nil {
			t.Fatal(err)
		}
		for _, g := range pg.Regions {
			if g.Text == "" || g.Label == "Picture" || g.Label == "Table" || g.Label == "Formula" {
				continue
			}
			var in []Result
			for _, l := range lines {
				x0, y0, x1, y1 := boxBounds(l.Box)
				cx, cy := (x0+x1)/2, (y0+y1)/2
				if cx >= g.Box[0] && cx <= g.Box[2] && cy >= g.Box[1] && cy <= g.Box[3] {
					in = append(in, l)
				}
			}
			sortLines(in)
			var texts []string
			for _, l := range in {
				texts = append(texts, l.Text)
			}
			edits += levenshtein(normSpace(g.Text), normSpace(strings.Join(texts, " ")))
			chars += len([]rune(normSpace(g.Text)))
		}
	}
	out.TextCER = float64(edits) / float64(max(chars, 1))
	scale := 1
	if v := os.Getenv("INGOT_TABLE_UPSCALE"); v == "1" {
		scale = 3
	}
	for _, tb := range tables {
		img := loadImageFile(t, filepath.Join(realDir, "pubtabnet", tb.Image))
		if scale > 1 { // small crops: upscale so the OCR can see the text
			b := img.Bounds()
			w, h := b.Dx()*scale, b.Dy()*scale
			rgb := resizeBicubic(img, w, h)
			up := image.NewRGBA(image.Rect(0, 0, w, h))
			for i := 0; i < w*h; i++ {
				copy(up.Pix[i*4:i*4+3], rgb[i*3:i*3+3])
				up.Pix[i*4+3] = 255
			}
			img = up
		}
		lines, err := ocr.Run(img)
		if err != nil {
			t.Fatal(err)
		}
		tt, err := tab.Recognize(img, lines)
		if err != nil {
			t.Fatal(err)
		}
		out.Tables = append(out.Tables, map[string]any{"image": tb.Image, "html": tt.HTML})
	}
	b, _ := json.MarshalIndent(out, "", " ")
	name := "ingot.json"
	if scale > 1 {
		name = "ingot_upscale.json"
	}
	if err := os.WriteFile(filepath.Join(realDir, name), b, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("%d pages, %d tables; text CER %.4f (%d edits / %d chars) -> %s", len(out.Pages), len(out.Tables), out.TextCER, edits, chars, name)
}

func normSpace(s string) string { return strings.Join(strings.Fields(s), " ") }

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Skip(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatal(err)
	}
}

func loadImageFile(t *testing.T, path string) image.Image {
	t.Helper()
	f, err := os.Open(path)
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
