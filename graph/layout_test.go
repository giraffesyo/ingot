package graph

import (
	"math"
	"testing"

	"github.com/giraffesyo/ingot/ops"
	"github.com/giraffesyo/ingot/tensor"
)

// buildResidualStack builds an mv2/effnet-style stack of two
// inverted-residual blocks (all channels 8, 4x4 spatial): a seed pointwise
// conv, then per block pw → dw → pw → Add(skip), with an SE island between
// the second block's dw and pw. The first block's output y1 feeds the second
// block's convs, the second Add, and an external Relu, so the pass must keep
// y1 blocked inside the region and export one NCHW copy for the Relu.
func buildResidualStack() *Graph {
	tg := newTestGraph()
	tg.g.Opsets["ingot"] = 1
	shape4 := func(names ...string) {
		for _, name := range names {
			v := tg.val(name)
			v.DType, v.Shape, v.HasShape = tensor.F32, []int{1, 8, 4, 4}, true
		}
	}
	wpw := func(name string, scale float32) {
		data := make([]float32, 8*8)
		for i := range data {
			data[i] = scale * float32(i%7-3) / 8
		}
		tg.constF32(name, []int{8, 8, 1, 1}, data...)
	}
	wdw := func(name string, scale float32) {
		data := make([]float32, 8*9)
		for i := range data {
			data[i] = scale * float32(i%5-2) / 8
		}
		tg.constF32(name, []int{8, 1, 3, 3}, data...)
	}
	dwAttrs := func(n *Node) {
		n.Attrs = ops.Attrs{
			"group": {Kind: ops.KindInt, I: 8},
			"pads":  {Kind: ops.KindInts, Ints: []int64{1, 1, 1, 1}},
		}
	}
	wpw("w0", 1)
	wpw("w1", 0.5)
	wdw("wd1", 1)
	wpw("w2", 0.7)
	wpw("w3", 0.6)
	wdw("wd2", 0.8)
	wpw("w4", 0.9)
	tg.node("Conv", []string{"x", "w0"}, "s")
	tg.node("Conv", []string{"s", "w1"}, "a")
	dwAttrs(tg.node("Conv", []string{"a", "wd1"}, "b"))
	tg.node("Conv", []string{"b", "w2"}, "c")
	tg.node("Add", []string{"c", "s"}, "y1")
	tg.node("Conv", []string{"y1", "w3"}, "d")
	dwAttrs(tg.node("Conv", []string{"d", "wd2"}, "e"))
	sew1 := make([]float32, 4*8)
	sew2 := make([]float32, 8*4)
	for i := range sew1 {
		sew1[i] = float32(i%5-2) / 4
		sew2[i] = float32(i%3-1) / 4
	}
	tg.constF32("wse1", []int{4, 8, 1, 1}, sew1...)
	tg.constF32("bse1", []int{4}, 0.1, -0.2, 0.3, 0.05)
	tg.constF32("wse2", []int{8, 4, 1, 1}, sew2...)
	tg.constF32("bse2", []int{8}, 1, 0.9, 1.1, 1, 0.8, 1.2, 1, 1)
	se := tg.node("SE", []string{"e", "wse1", "bse1", "wse2", "bse2"}, "es")
	se.Domain = "ingot"
	tg.node("Conv", []string{"es", "w4"}, "f")
	tg.node("Add", []string{"f", "y1"}, "y2")
	tg.node("Relu", []string{"y1"}, "z")
	// Only the input carries a shape: propagateShapes must derive the rest
	// (conv arithmetic, dw pads, Add broadcast, SE passthrough).
	shape4("x")
	return tg.finish([]string{"x"}, []string{"y2", "z"})
}

// TestBlockedLayoutResidual checks that assignBlockedLayout grows one region
// through both residual Adds (including y1's fan-out to the next block and
// the skip), converts at exactly the region edges, and preserves numerics.
func TestBlockedLayoutResidual(t *testing.T) {
	graw, gopt := buildResidualStack(), buildResidualStack()
	stats := Optimize(gopt)
	if stats["blk-regions"] != 1 || stats["assign-blk"] != 7 || stats["blk-add"] != 2 || stats["blk-se"] != 1 || stats["fuse-blk-res"] != 2 {
		t.Fatalf("stats = %v, want blk-regions:1 assign-blk:7 blk-add:2 blk-se:1 fuse-blk-res:2", stats)
	}
	count := map[string]int{}
	for _, n := range gopt.Nodes {
		count[n.OpType]++
	}
	// One entry (x), two exits (y1 for the Relu, y2 as graph output).
	if count["ToBlk8"] != 1 || count["FromBlk8"] != 2 {
		t.Fatalf("conversions = %v, want 1 ToBlk8, 2 FromBlk8", count)
	}
	// Both residual Adds fold into their project convs (fuse-blk-res).
	if count["Conv"] != 0 || count["ConvDwBlk"] != 2 || count["ConvPwBlk"] != 5 || count["Add"] != 0 || count["SE"] != 1 {
		t.Fatalf("node mix = %v", count)
	}
	// The Relu must read the NCHW copy of y1 (produced by a FromBlk8), while
	// the blocked consumers read the blocked value directly.
	for _, n := range gopt.Nodes {
		if n.OpType == "Relu" {
			if p := n.Inputs[0].Producer; p == nil || p.OpType != "FromBlk8" {
				t.Fatalf("Relu input produced by %v, want FromBlk8", p)
			}
		}
	}

	x := tensor.New(tensor.F32, 1, 8, 4, 4)
	xf := x.F32()
	for i := range xf {
		xf[i] = float32((i*31)%17-8) / 8
	}
	feeds := map[string]*tensor.Tensor{"x": x}
	raw, err := CompileRaw(graw)
	if err != nil {
		t.Fatal(err)
	}
	opt, err := CompileRaw(gopt)
	if err != nil {
		t.Fatal(err)
	}
	ra, err := raw.Run(feeds)
	if err != nil {
		t.Fatal(err)
	}
	ro, err := opt.Run(feeds)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"y2", "z"} {
		fa, fb := ra[name].F32(), ro[name].F32()
		if len(fa) != len(fb) {
			t.Fatalf("%s: len %d vs %d", name, len(fa), len(fb))
		}
		for i := range fa {
			if d := math.Abs(float64(fa[i] - fb[i])); d > 1e-5 {
				t.Fatalf("%s[%d] = %v raw vs %v blocked", name, i, fa[i], fb[i])
			}
		}
	}
}

// TestBlockedLayoutPrunesSingleConv: a lone eligible conv is not worth the
// two conversion edges and must stay NCHW.
func TestBlockedLayoutPrunesSingleConv(t *testing.T) {
	tg := newTestGraph()
	data := make([]float32, 8*8)
	for i := range data {
		data[i] = float32(i%7-3) / 8
	}
	tg.constF32("w0", []int{8, 8, 1, 1}, data...)
	tg.node("Conv", []string{"x", "w0"}, "y")
	xv := tg.val("x")
	xv.DType, xv.Shape, xv.HasShape = tensor.F32, []int{1, 8, 4, 4}, true
	g := tg.finish([]string{"x"}, []string{"y"})
	stats := Optimize(g)
	if stats["blk-regions"] != 0 || stats["assign-blk"] != 0 {
		t.Fatalf("stats = %v, want no blocked assignment", stats)
	}
	if len(g.Nodes) != 1 || g.Nodes[0].OpType != "Conv" {
		t.Fatalf("nodes = %v, want a single Conv", g.Nodes)
	}
}
