package ocr

import (
	"strings"
	"testing"
)

// TestDocument reads the synthetic pages end to end (layout + OCR + tables)
// against their ground truth: every truth region's text is recovered by the
// region covering it (character accuracy >= 97% per page), the table's
// cells exactly, and the Markdown keeps the reading order.
func TestDocument(t *testing.T) {
	p := &DocPipeline{OCR: pipelineOn(t, "cpu", "cpu"), Layout: layoutDetector(t), Tables: tableRecognizer(t)}
	for _, pg := range loadLayoutJSON(t, "truth.json") {
		doc, err := p.Run(loadPage(t, pg.Image))
		if err != nil {
			t.Fatal(err)
		}
		var edits, chars int
		for _, w := range pg.Regions {
			if w.Text == "" || w.Label == "table" {
				continue
			}
			best, bi := 0.0, -1
			for i, r := range doc.Regions {
				if v := iouAABB(w.Box[0], w.Box[1], w.Box[2], w.Box[3], r.Box[0], r.Box[1], r.Box[2], r.Box[3]); v > best {
					best, bi = v, i
				}
			}
			if bi < 0 {
				t.Errorf("%s: no region for %s", pg.Image, w.Label)
				continue
			}
			got := doc.Regions[bi].Text
			edits += levenshtein(normText(w.Text), normText(got))
			chars += len([]rune(w.Text))
		}
		acc := 1 - float64(edits)/float64(max(chars, 1))
		t.Logf("%s: %d regions, char accuracy %.4f (%d edits / %d), %d unassigned lines", pg.Image, len(doc.Regions), acc, edits, chars, len(doc.Unassigned))
		if acc < 0.97 {
			t.Errorf("%s: char accuracy %.4f", pg.Image, acc)
		}
		for _, r := range doc.Regions {
			if r.Label == "table" {
				if r.Table == nil || len(r.Table.Rows) != 5 {
					t.Errorf("%s: table not recognised: %+v", pg.Image, r.Table)
				} else if r.Table.Rows[1][0].Text != "detector" || r.Table.Rows[2][2].Text != "59.0" {
					t.Errorf("%s: table cells %q %q", pg.Image, r.Table.Rows[1][0].Text, r.Table.Rows[2][2].Text)
				}
			}
		}
		if pg.Image == "page_1.png" { // two columns: Results before Discussion
			md := doc.Markdown()
			if i, j := strings.Index(md, "## Results"), strings.Index(md, "## Discussion"); i < 0 || j < i {
				t.Errorf("two-column reading order lost:\n%s", md)
			}
		}
	}
}
