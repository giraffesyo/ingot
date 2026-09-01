package ops

import (
	"sort"

	"github.com/giraffesyo/ingot/tensor"
)

// topKOp: ONNX TopK (opset 11+). Inputs X, K (int64 scalar/1-elem); attrs
// axis (default -1), largest (1), sorted (1). Outputs Values, Indices (i64).
// Ties resolve to the smaller index, matching ONNX Runtime.
type topKOp struct {
	n       NodeInfo
	axis    int
	largest bool
	sorted  bool
}

func (o *topKOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("TopK: need X and K")
	}
	x, kt := in[0], in[1]
	if x.DType() != tensor.F32 || kt.DType() != tensor.I64 || kt.Numel() != 1 {
		return nil, o.n.Errorf("TopK: want f32 X and scalar int64 K, got %s/%s[%d]", x.DType(), kt.DType(), kt.Numel())
	}
	k := int(kt.I64()[0])
	xs := x.Shape()
	axis := o.axis
	if axis < 0 {
		axis += len(xs)
	}
	if axis < 0 || axis >= len(xs) {
		return nil, o.n.Errorf("TopK: axis %d out of range for %v", o.axis, xs)
	}
	n := xs[axis]
	if k < 0 || k > n {
		return nil, o.n.Errorf("TopK: k=%d out of range for axis size %d", k, n)
	}
	outer, inner := 1, 1
	for i := 0; i < axis; i++ {
		outer *= xs[i]
	}
	for i := axis + 1; i < len(xs); i++ {
		inner *= xs[i]
	}
	os := append([]int(nil), xs...)
	os[axis] = k
	vals := ctx.NewUninit(tensor.F32, os...)
	idxs := ctx.NewUninit(tensor.I64, os...)
	xf, vf, ifc := x.F32(), vals.F32(), idxs.I64()
	ord := make([]int, n)
	for oi := 0; oi < outer; oi++ {
		for ii := 0; ii < inner; ii++ {
			base := oi*n*inner + ii
			for j := range ord {
				ord[j] = j
			}
			at := func(j int) float32 { return xf[base+j*inner] }
			sort.SliceStable(ord, func(a, b int) bool {
				va, vb := at(ord[a]), at(ord[b])
				if va == vb {
					return ord[a] < ord[b]
				}
				if o.largest {
					return va > vb
				}
				return va < vb
			})
			dst := oi*k*inner + ii
			for j := 0; j < k; j++ {
				vf[dst+j*inner] = at(ord[j])
				ifc[dst+j*inner] = int64(ord[j])
			}
		}
	}
	outs := ctx.OutPad(2, vals)
	outs[1] = idxs
	return outs, nil
}

// nmsOp: ONNX NonMaxSuppression (opset 10/11). Inputs boxes [B,S,4],
// scores [B,C,S], then optional max_output_boxes_per_class, iou_threshold,
// score_threshold. Attr center_point_box: 0 = [y1,x1,y2,x2] corners (any
// order), 1 = [cx,cy,w,h]. Output selected_indices [n,3] int64 rows of
// (batch, class, box), in the greedy selection order per (batch, class).
type nmsOp struct {
	n      NodeInfo
	center bool
}

func (o *nmsOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("NonMaxSuppression: need boxes and scores")
	}
	boxes, scores := in[0], in[1]
	bs, ss := boxes.Shape(), scores.Shape()
	if len(bs) != 3 || bs[2] != 4 || len(ss) != 3 || bs[0] != ss[0] || bs[1] != ss[2] {
		return nil, o.n.Errorf("NonMaxSuppression: shapes boxes=%v scores=%v", bs, ss)
	}
	B, S, C := bs[0], bs[1], ss[1]
	maxOut := S
	if len(in) > 2 && in[2] != nil && in[2].Numel() == 1 {
		maxOut = int(in[2].I64()[0])
	}
	var iouTh, scoreTh float32
	hasScoreTh := false
	if len(in) > 3 && in[3] != nil && in[3].Numel() == 1 {
		iouTh = in[3].F32()[0]
	}
	if len(in) > 4 && in[4] != nil && in[4].Numel() == 1 {
		scoreTh = in[4].F32()[0]
		hasScoreTh = true
	}
	bf, sf := boxes.F32(), scores.F32()
	// Corner form with min/max normalisation (spec allows either corner order).
	rect := func(b, s int) (y1, x1, y2, x2 float32) {
		p := bf[(b*S+s)*4:]
		if o.center {
			cx, cy, w, h := p[0], p[1], p[2], p[3]
			return cy - h/2, cx - w/2, cy + h/2, cx + w/2
		}
		y1, x1, y2, x2 = p[0], p[1], p[2], p[3]
		if y1 > y2 {
			y1, y2 = y2, y1
		}
		if x1 > x2 {
			x1, x2 = x2, x1
		}
		return
	}
	iou := func(a, b [4]float32) float32 {
		yy1, xx1 := max(a[0], b[0]), max(a[1], b[1])
		yy2, xx2 := min(a[2], b[2]), min(a[3], b[3])
		iw, ih := xx2-xx1, yy2-yy1
		if iw <= 0 || ih <= 0 {
			return 0
		}
		inter := iw * ih
		areaA := (a[2] - a[0]) * (a[3] - a[1])
		areaB := (b[2] - b[0]) * (b[3] - b[1])
		den := areaA + areaB - inter
		if den <= 0 {
			return 0
		}
		return inter / den
	}
	var sel [][3]int64
	ord := make([]int, 0, S)
	kept := make([][4]float32, 0, S)
	for b := 0; b < B; b++ {
		for c := 0; c < C; c++ {
			row := sf[(b*C+c)*S:]
			ord = ord[:0]
			for s := 0; s < S; s++ {
				if !hasScoreTh || row[s] > scoreTh {
					ord = append(ord, s)
				}
			}
			sort.SliceStable(ord, func(i, j int) bool {
				if row[ord[i]] == row[ord[j]] {
					return ord[i] < ord[j]
				}
				return row[ord[i]] > row[ord[j]]
			})
			kept = kept[:0]
			for _, s := range ord {
				if len(kept) >= maxOut {
					break
				}
				y1, x1, y2, x2 := rect(b, s)
				cand := [4]float32{y1, x1, y2, x2}
				ok := true
				for _, kb := range kept {
					if iou(cand, kb) > iouTh {
						ok = false
						break
					}
				}
				if ok {
					kept = append(kept, cand)
					sel = append(sel, [3]int64{int64(b), int64(c), int64(s)})
				}
			}
		}
	}
	out := ctx.NewUninit(tensor.I64, len(sel), 3)
	of := out.I64()
	for i, s := range sel {
		of[i*3], of[i*3+1], of[i*3+2] = s[0], s[1], s[2]
	}
	return ctx.Out(out), nil
}

func init() {
	Register("", "TopK", 11, func(n NodeInfo) (Op, error) {
		return &topKOp{
			n:       n,
			axis:    int(n.Attrs.Int("axis", -1)),
			largest: n.Attrs.Int("largest", 1) != 0,
			sorted:  n.Attrs.Int("sorted", 1) != 0,
		}, nil
	})
	Register("", "NonMaxSuppression", 10, func(n NodeInfo) (Op, error) {
		return &nmsOp{n: n, center: n.Attrs.Int("center_point_box", 0) != 0}, nil
	})
}
