package ocr

import (
	"fmt"
	"image"
	"image/draw"
	"sort"
	"strings"
)

// DocPipeline reads a page into a structured document: layout regions in
// reading order, each with its text lines (and, for tables, the recognised
// grid). OCR runs once over the page; lines are assigned to regions, and a
// table is rebuilt from the lines inside it.
type DocPipeline struct {
	OCR    *Pipeline
	Layout *LayoutDetector
	Tables *TableRecognizer // optional: nil leaves table regions as text
}

// DocRegion is a layout region with its content.
type DocRegion struct {
	Region
	Lines []Result `json:",omitempty"` // text lines inside, top to bottom
	Text  string   `json:",omitempty"` // the lines joined (paragraph text)
	Table *Table   `json:",omitempty"`
}

// Document is a read page.
type Document struct {
	Width, Height int
	Regions       []DocRegion
	// Unassigned holds text lines no layout region claimed (kept, never
	// dropped).
	Unassigned []Result `json:",omitempty"`
}

// Run reads img.
func (p *DocPipeline) Run(img image.Image) (*Document, error) {
	regs, err := p.Layout.Detect(img)
	if err != nil {
		return nil, fmt.Errorf("layout: %w", err)
	}
	lines, err := p.OCR.Run(img)
	if err != nil {
		return nil, fmt.Errorf("ocr: %w", err)
	}
	b := img.Bounds()
	doc := &Document{Width: b.Dx(), Height: b.Dy()}
	doc.Regions = make([]DocRegion, len(regs))
	for i, r := range regs {
		doc.Regions[i].Region = r
	}
	// Each line goes to the region covering most of it (at least half).
	for _, l := range lines {
		x0, y0, x1, y1 := boxBounds(l.Box)
		area := max(1e-9, (x1-x0)*(y1-y0))
		best, bestCover := -1, 0.5
		for i, r := range regs {
			iw := min(x1, r.Box[2]) - max(x0, r.Box[0])
			ih := min(y1, r.Box[3]) - max(y0, r.Box[1])
			if iw <= 0 || ih <= 0 {
				continue
			}
			if c := iw * ih / area; c >= bestCover {
				best, bestCover = i, c
			}
		}
		if best < 0 {
			doc.Unassigned = append(doc.Unassigned, l)
			continue
		}
		doc.Regions[best].Lines = append(doc.Regions[best].Lines, l)
	}
	for i := range doc.Regions {
		dr := &doc.Regions[i]
		sortLines(dr.Lines)
		var texts []string
		for _, l := range dr.Lines {
			texts = append(texts, strings.TrimSpace(l.Text))
		}
		dr.Text = strings.Join(texts, " ")
		if dr.Label == "table" && p.Tables != nil {
			t, err := p.table(img, dr)
			if err != nil {
				return nil, fmt.Errorf("table: %w", err)
			}
			dr.Table = t
		}
	}
	return doc, nil
}

// table recognises a table region from a crop, reusing the page's OCR lines
// (shifted into crop coordinates).
func (p *DocPipeline) table(img image.Image, dr *DocRegion) (*Table, error) {
	b := img.Bounds()
	r := image.Rect(int(dr.Box[0]), int(dr.Box[1]), int(dr.Box[2]+0.5), int(dr.Box[3]+0.5)).Add(b.Min).Intersect(b)
	crop := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	draw.Draw(crop, crop.Bounds(), img, r.Min, draw.Src)
	ox, oy := float64(r.Min.X-b.Min.X), float64(r.Min.Y-b.Min.Y)
	lines := make([]Result, len(dr.Lines))
	for i, l := range dr.Lines {
		lines[i] = l
		for k := range l.Box.Pts {
			lines[i].Box.Pts[k].X -= ox
			lines[i].Box.Pts[k].Y -= oy
		}
	}
	t, err := p.Tables.Recognize(crop, lines)
	if err != nil {
		return nil, err
	}
	for i := range t.Rows { // cell boxes back to page coordinates
		for j := range t.Rows[i] {
			bx := &t.Rows[i][j].Box
			bx[0], bx[1], bx[2], bx[3] = bx[0]+ox, bx[1]+oy, bx[2]+ox, bx[3]+oy
		}
	}
	return t, nil
}

// sortLines orders lines top to bottom, then left to right within a row
// (the same banding as Pipeline.Run).
func sortLines(ls []Result) {
	sort.SliceStable(ls, func(i, j int) bool {
		yi := (ls[i].Box.Pts[0].Y + ls[i].Box.Pts[1].Y) / 2
		yj := (ls[j].Box.Pts[0].Y + ls[j].Box.Pts[1].Y) / 2
		if yi-yj > 10 || yj-yi > 10 {
			return yi < yj
		}
		return ls[i].Box.Pts[0].X < ls[j].Box.Pts[0].X
	})
}

// Markdown renders the document's reading flow: titles as headings, text as
// paragraphs, tables as pipe tables (spanned cells repeated), figures as
// placeholders; page furniture (headers, footers, page numbers) is left
// out.
func (d *Document) Markdown() string {
	var sb strings.Builder
	for _, r := range d.Regions {
		switch r.Label {
		case "header", "footer", "number", "header_image", "footer_image":
			continue
		case "doc_title":
			fmt.Fprintf(&sb, "# %s\n\n", r.Text)
		case "paragraph_title":
			fmt.Fprintf(&sb, "## %s\n\n", r.Text)
		case "table":
			if r.Table != nil && len(r.Table.Rows) > 0 {
				sb.WriteString(markdownTable(r.Table))
				sb.WriteString("\n")
				continue
			}
			if r.Text != "" {
				fmt.Fprintf(&sb, "%s\n\n", r.Text)
			}
		case "image", "chart", "seal":
			fmt.Fprintf(&sb, "![%s]()\n\n", r.Label)
		case "figure_title":
			fmt.Fprintf(&sb, "*%s*\n\n", r.Text)
		default:
			if r.Text != "" {
				fmt.Fprintf(&sb, "%s\n\n", r.Text)
			}
		}
	}
	return sb.String()
}

func markdownTable(t *Table) string {
	var sb strings.Builder
	cols := 0
	for _, row := range t.Rows {
		n := 0
		for _, c := range row {
			n += c.ColSpan
		}
		cols = max(cols, n)
	}
	for i, row := range t.Rows {
		var cells []string
		for _, c := range row {
			txt := strings.ReplaceAll(c.Text, "|", `\|`)
			for range c.ColSpan {
				cells = append(cells, txt)
			}
		}
		for len(cells) < cols {
			cells = append(cells, "")
		}
		sb.WriteString("| " + strings.Join(cells, " | ") + " |\n")
		if i == 0 {
			sb.WriteString("|" + strings.Repeat(" --- |", cols) + "\n")
		}
	}
	return sb.String()
}
