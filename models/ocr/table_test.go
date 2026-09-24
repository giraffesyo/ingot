package ocr

import (
	"encoding/json"
	"image"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func tableRecognizer(t *testing.T) *TableRecognizer {
	t.Helper()
	p := filepath.Join(layoutDir, "slanet-plus.onnx")
	if _, err := os.Stat(p); err != nil {
		t.Skip("SLANet-plus not present (tools/export/table.py)")
	}
	r, err := NewTableRecognizer(p, "cpu")
	if err != nil {
		t.Fatal(err)
	}
	return r
}

type tableRef struct {
	Image  string       `json:"image"`
	Tokens []string     `json:"tokens"`
	Score  float64      `json:"score"`
	Cells  [][8]float64 `json:"cells"`
}

type tableTruth struct {
	Image string `json:"image"`
	Rows  [][]struct {
		Text    string `json:"text"`
		Colspan int    `json:"colspan"`
	} `json:"rows"`
}

func readTableJSON(t *testing.T, name string, v any) {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(layoutDir, "tables", name))
	if err != nil {
		t.Skip(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatal(err)
	}
}

// TestTableStructureParity: SLANet-plus structure tokens identical to
// RapidTable's, cell boxes within a few pixels (our bilinear resize vs
// cv2's fixed-point), mean score within 0.01.
func TestTableStructureParity(t *testing.T) {
	r := tableRecognizer(t)
	var refs []tableRef
	readTableJSON(t, "ref.json", &refs)
	for _, ref := range refs {
		img := loadTable(t, ref.Image)
		st, err := r.Structure(img)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Join(st.Tokens, "") != strings.Join(ref.Tokens, "") {
			t.Fatalf("%s tokens:\n got %s\nwant %s", ref.Image, strings.Join(st.Tokens, ""), strings.Join(ref.Tokens, ""))
		}
		if len(st.Cells) != len(ref.Cells) {
			t.Fatalf("%s: %d cells, ref %d", ref.Image, len(st.Cells), len(ref.Cells))
		}
		var maxd float64
		for i, c := range st.Cells {
			for k := range 8 {
				maxd = math.Max(maxd, math.Abs(c[k]-ref.Cells[i][k]))
			}
		}
		t.Logf("%s: %d cells, max cell delta %.1f px, score %.3f (ref %.3f)", ref.Image, len(st.Cells), maxd, st.Score, ref.Score)
		if maxd > 4 || math.Abs(st.Score-ref.Score) > 0.01 {
			t.Errorf("%s: cell delta %.1f px, score %.3f vs %.3f", ref.Image, maxd, st.Score, ref.Score)
		}
	}
}

// TestTableRecognize: structure + our OCR pipeline read each table back —
// the grid shape and colspans must match the ground truth, and at least
// 90% of cells their text exactly.
func TestTableRecognize(t *testing.T) {
	r := tableRecognizer(t)
	p := pipelineOn(t, "cpu", "cpu")
	var truth []tableTruth
	readTableJSON(t, "truth.json", &truth)
	for _, tt := range truth {
		img := loadTable(t, tt.Image)
		ocr, err := p.Run(img)
		if err != nil {
			t.Fatal(err)
		}
		tab, err := r.Recognize(img, ocr)
		if err != nil {
			t.Fatal(err)
		}
		if len(tab.Rows) != len(tt.Rows) {
			t.Fatalf("%s: %d rows, want %d\n%s", tt.Image, len(tab.Rows), len(tt.Rows), tab.HTML)
		}
		total, exact := 0, 0
		for i, row := range tt.Rows {
			if len(tab.Rows[i]) != len(row) {
				t.Fatalf("%s row %d: %d cells, want %d\n%s", tt.Image, i, len(tab.Rows[i]), len(row), tab.HTML)
			}
			for j, c := range row {
				got := tab.Rows[i][j]
				if got.ColSpan != c.Colspan {
					t.Errorf("%s [%d,%d]: colspan %d, want %d", tt.Image, i, j, got.ColSpan, c.Colspan)
				}
				total++
				if got.Text == c.Text {
					exact++
				} else {
					t.Logf("%s [%d,%d]: %q, want %q", tt.Image, i, j, got.Text, c.Text)
				}
			}
		}
		t.Logf("%s: %d/%d cells exact\n%s", tt.Image, exact, total, tab.HTML)
		if exact*10 < total*9 {
			t.Errorf("%s: only %d/%d cells exact", tt.Image, exact, total)
		}
	}
}

func loadTable(t *testing.T, name string) image.Image {
	t.Helper()
	f, err := os.Open(filepath.Join(layoutDir, "tables", name))
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
