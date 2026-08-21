package ocr

import (
	"math"
	"sort"
)

// Point is a 2-D point (float coordinates).
type Point struct{ X, Y float64 }

// Box is a detected text quadrilateral (4 corners, clockwise) with a score.
type Box struct {
	Pts   [4]Point
	Score float64
}

// boxesFromProb converts a DBNet probability map into text boxes in original
// image coordinates via: binarise → connected components → min-area rotated
// rectangle → unclip → score/size filter. sx,sy are resized/original scale.
func boxesFromProb(prob []float32, H, W int, sx, sy float64, d *Detector) []Box {
	bin := make([]bool, H*W)
	thr := float32(d.BinThresh)
	for i, p := range prob {
		bin[i] = p > thr
	}
	comps := connectedComponents(bin, H, W)
	var boxes []Box
	for _, comp := range comps {
		if len(comp) < 4 {
			continue
		}
		pts := make([]Point, len(comp))
		for i, idx := range comp {
			pts[i] = Point{float64(idx % W), float64(idx / W)}
		}
		rect := minAreaRect(pts)
		if math.Min(rect.w, rect.h) < 3 {
			continue
		}
		score := boxScore(prob, H, W, comp)
		if score < d.BoxThresh {
			continue
		}
		rect = rect.unclip(d.Unclip)
		corners := rect.corners()
		// map to original coords
		var b Box
		b.Score = score
		short := math.Inf(1)
		for i, c := range corners {
			b.Pts[i] = Point{c.X / sx, c.Y / sy}
		}
		short = math.Min(rect.w/sx, rect.h/sy)
		if short < float64(d.MinSize) {
			continue
		}
		boxes = append(boxes, b)
	}
	return boxes
}

// connectedComponents returns 4-connected components (pixel indices) of a binary
// mask, via iterative BFS.
func connectedComponents(bin []bool, H, W int) [][]int {
	seen := make([]bool, H*W)
	var comps [][]int
	queue := make([]int, 0, 256)
	for start := 0; start < H*W; start++ {
		if !bin[start] || seen[start] {
			continue
		}
		queue = queue[:0]
		queue = append(queue, start)
		seen[start] = true
		var comp []int
		for len(queue) > 0 {
			p := queue[len(queue)-1]
			queue = queue[:len(queue)-1]
			comp = append(comp, p)
			y, x := p/W, p%W
			if x > 0 && bin[p-1] && !seen[p-1] {
				seen[p-1] = true
				queue = append(queue, p-1)
			}
			if x < W-1 && bin[p+1] && !seen[p+1] {
				seen[p+1] = true
				queue = append(queue, p+1)
			}
			if y > 0 && bin[p-W] && !seen[p-W] {
				seen[p-W] = true
				queue = append(queue, p-W)
			}
			if y < H-1 && bin[p+W] && !seen[p+W] {
				seen[p+W] = true
				queue = append(queue, p+W)
			}
		}
		comps = append(comps, comp)
	}
	return comps
}

// boxScore is the mean probability over a component's pixels.
func boxScore(prob []float32, H, W int, comp []int) float64 {
	var sum float64
	for _, idx := range comp {
		sum += float64(prob[idx])
	}
	return sum / float64(len(comp))
}

// oriented rectangle: center, half-extents along its axes, and rotation.
type rect struct {
	cx, cy, w, h, angle float64
}

func (r rect) corners() [4]Point {
	c, s := math.Cos(r.angle), math.Sin(r.angle)
	hw, hh := r.w/2, r.h/2
	// local corners CW from top-left
	local := [4][2]float64{{-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}}
	var out [4]Point
	for i, l := range local {
		out[i] = Point{r.cx + l[0]*c - l[1]*s, r.cy + l[0]*s + l[1]*c}
	}
	return out
}

// unclip grows the rectangle outward by area*ratio/perimeter on each side.
func (r rect) unclip(ratio float64) rect {
	area := r.w * r.h
	peri := 2 * (r.w + r.h)
	if peri == 0 {
		return r
	}
	d := area * ratio / peri
	r.w += 2 * d
	r.h += 2 * d
	return r
}

// minAreaRect computes the minimum-area enclosing rectangle of pts via convex
// hull + rotating calipers (checking each hull edge orientation).
func minAreaRect(pts []Point) rect {
	hull := convexHull(pts)
	if len(hull) < 3 {
		return aabbRect(pts)
	}
	best := math.Inf(1)
	var br rect
	n := len(hull)
	for i := 0; i < n; i++ {
		a := hull[i]
		b := hull[(i+1)%n]
		ex, ey := b.X-a.X, b.Y-a.Y
		l := math.Hypot(ex, ey)
		if l < 1e-9 {
			continue
		}
		ux, uy := ex/l, ey/l // edge direction
		vx, vy := -uy, ux    // normal
		var minU, maxU, minV, maxV = math.Inf(1), math.Inf(-1), math.Inf(1), math.Inf(-1)
		for _, p := range hull {
			du := (p.X-a.X)*ux + (p.Y-a.Y)*uy
			dv := (p.X-a.X)*vx + (p.Y-a.Y)*vy
			minU, maxU = math.Min(minU, du), math.Max(maxU, du)
			minV, maxV = math.Min(minV, dv), math.Max(maxV, dv)
		}
		w, h := maxU-minU, maxV-minV
		if area := w * h; area < best {
			best = area
			cu, cv := (minU+maxU)/2, (minV+maxV)/2
			br = rect{
				cx:    a.X + cu*ux + cv*vx,
				cy:    a.Y + cu*uy + cv*vy,
				w:     w,
				h:     h,
				angle: math.Atan2(uy, ux),
			}
		}
	}
	return br
}

func aabbRect(pts []Point) rect {
	minX, minY, maxX, maxY := math.Inf(1), math.Inf(1), math.Inf(-1), math.Inf(-1)
	for _, p := range pts {
		minX, maxX = math.Min(minX, p.X), math.Max(maxX, p.X)
		minY, maxY = math.Min(minY, p.Y), math.Max(maxY, p.Y)
	}
	return rect{(minX + maxX) / 2, (minY + maxY) / 2, maxX - minX, maxY - minY, 0}
}

// convexHull returns the convex hull (CCW) via Andrew's monotone chain.
func convexHull(pts []Point) []Point {
	p := append([]Point(nil), pts...)
	sort.Slice(p, func(i, j int) bool {
		if p[i].X != p[j].X {
			return p[i].X < p[j].X
		}
		return p[i].Y < p[j].Y
	})
	n := len(p)
	if n < 3 {
		return p
	}
	cross := func(o, a, b Point) float64 {
		return (a.X-o.X)*(b.Y-o.Y) - (a.Y-o.Y)*(b.X-o.X)
	}
	h := make([]Point, 0, 2*n)
	for _, pt := range p { // lower
		for len(h) >= 2 && cross(h[len(h)-2], h[len(h)-1], pt) <= 0 {
			h = h[:len(h)-1]
		}
		h = append(h, pt)
	}
	l := len(h) + 1
	for i := n - 2; i >= 0; i-- { // upper
		pt := p[i]
		for len(h) >= l && cross(h[len(h)-2], h[len(h)-1], pt) <= 0 {
			h = h[:len(h)-1]
		}
		h = append(h, pt)
	}
	return h[:len(h)-1]
}
