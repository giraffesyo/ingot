package ops

import (
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

func TestSliceNegStep(t *testing.T) {
	// x = [0..11] shape [3,4]
	x := iota32(3, 4)
	// Slice axis1 [3:0:-1] -> cols 3,2,1
	op := mkOp(t, "Slice", 10, nil, 5, 1)
	out := run(t, op, x,
		i64t([]int{1}, 3),  // starts
		i64t([]int{1}, 0),  // ends
		i64t([]int{1}, 1),  // axes
		i64t([]int{1}, -1), // steps
	)[0]
	// row r: [r*4+3, r*4+2, r*4+1]
	eqF32(t, "sliceNeg", out, []int{3, 3}, []float32{3, 2, 1, 7, 6, 5, 11, 10, 9})
}

func TestSliceStridedForward(t *testing.T) {
	x := iota32(2, 6)
	op := mkOp(t, "Slice", 10, nil, 5, 1)
	out := run(t, op, x,
		i64t([]int{1}, 1), i64t([]int{1}, 6), i64t([]int{1}, 1), i64t([]int{1}, 2))[0]
	// cols 1,3,5
	eqF32(t, "sliceStride", out, []int{2, 3}, []float32{1, 3, 5, 7, 9, 11})
}

func TestSliceNegativeIndices(t *testing.T) {
	x := iota32(1, 5)
	op := mkOp(t, "Slice", 10, nil, 3, 1)
	out := run(t, op, x, i64t([]int{1}, -3), i64t([]int{1}, -1), i64t([]int{1}, 1))[0]
	eqF32(t, "sliceNegIdx", out, []int{1, 2}, []float32{2, 3})
}

func TestTransposePerm(t *testing.T) {
	x := iota32(2, 3, 4) // 24
	op := mkOp(t, "Transpose", 1, Attrs{"perm": {Kind: KindInts, Ints: []int64{2, 0, 1}}}, 1, 1)
	out := run(t, op, x)[0]
	if !out.Shape().Equal(tensor.Shape{4, 2, 3}) {
		t.Fatalf("shape %v", out.Shape())
	}
	// verify a few elements: out[a,b,c] = x[b,c,a]
	xf := x.F32()
	of := out.F32()
	get := func(a, b, c, d0, d1, d2 int, f []float32) float32 { return f[(a*d1+b)*d2+c] }
	for a := 0; a < 4; a++ {
		for b := 0; b < 2; b++ {
			for c := 0; c < 3; c++ {
				want := get(b, c, a, 2, 3, 4, xf)
				got := get(a, b, c, 4, 2, 3, of)
				if want != got {
					t.Fatalf("transpose[%d,%d,%d]=%g want %g", a, b, c, got, want)
				}
			}
		}
	}
}

func TestGatherAxisAndNeg(t *testing.T) {
	x := iota32(4, 2) // rows 0..3
	op := mkOp(t, "Gather", 1, Attrs{"axis": {Kind: KindInt, I: 0}}, 2, 1)
	out := run(t, op, x, i64t([]int{3}, 0, -1, 2))[0] // rows 0,3,2
	eqF32(t, "gather0", out, []int{3, 2}, []float32{0, 1, 6, 7, 4, 5})

	op2 := mkOp(t, "Gather", 1, Attrs{"axis": {Kind: KindInt, I: 1}}, 2, 1)
	out2 := run(t, op2, x, i64t([]int{1}, 1))[0] // col 1
	eqF32(t, "gather1", out2, []int{4, 1}, []float32{1, 3, 5, 7})
}

func TestConcatAxes(t *testing.T) {
	a := iota32(2, 2)
	b := f32t([]int{2, 2}, 10, 11, 12, 13)
	op := mkOp(t, "Concat", 4, Attrs{"axis": {Kind: KindInt, I: 1}}, 2, 1)
	out := run(t, op, a, b)[0]
	eqF32(t, "concat1", out, []int{2, 4}, []float32{0, 1, 10, 11, 2, 3, 12, 13})

	op0 := mkOp(t, "Concat", 4, Attrs{"axis": {Kind: KindInt, I: 0}}, 2, 1)
	out0 := run(t, op0, a, b)[0]
	eqF32(t, "concat0", out0, []int{4, 2}, []float32{0, 1, 2, 3, 10, 11, 12, 13})
}

func TestSplitEqualAndUneven(t *testing.T) {
	x := iota32(1, 6)
	op := mkOp(t, "Split", 13, Attrs{"axis": {Kind: KindInt, I: 1}, "num_outputs": {Kind: KindInt, I: 3}}, 1, 3)
	outs := run(t, op, x)
	if len(outs) != 3 {
		t.Fatalf("got %d outputs", len(outs))
	}
	eqF32(t, "split0", outs[0], []int{1, 2}, []float32{0, 1})
	eqF32(t, "split2", outs[2], []int{1, 2}, []float32{4, 5})

	op2 := mkOp(t, "Split", 13, Attrs{"axis": {Kind: KindInt, I: 1}}, 2, 2)
	outs2 := run(t, op2, x, i64t([]int{2}, 4, 2))
	eqF32(t, "splitU0", outs2[0], []int{1, 4}, []float32{0, 1, 2, 3})
	eqF32(t, "splitU1", outs2[1], []int{1, 2}, []float32{4, 5})
}

func TestSqueezeUnsqueeze(t *testing.T) {
	x := iota32(1, 3, 1)
	sq := mkOp(t, "Squeeze", 13, nil, 2, 1)
	out := run(t, sq, x, i64t([]int{2}, 0, 2))[0]
	if !out.Shape().Equal(tensor.Shape{3}) {
		t.Fatalf("squeeze shape %v", out.Shape())
	}
	un := mkOp(t, "Unsqueeze", 13, nil, 2, 1)
	out2 := run(t, un, out, i64t([]int{2}, 0, 2))[0]
	if !out2.Shape().Equal(tensor.Shape{1, 3, 1}) {
		t.Fatalf("unsqueeze shape %v", out2.Shape())
	}
}

func TestTile(t *testing.T) {
	x := f32t([]int{2, 2}, 1, 2, 3, 4)
	op := mkOp(t, "Tile", 6, nil, 2, 1)
	out := run(t, op, x, i64t([]int{2}, 2, 1))[0]
	eqF32(t, "tile", out, []int{4, 2}, []float32{1, 2, 3, 4, 1, 2, 3, 4})
}

func TestReshapeNegOneAndZero(t *testing.T) {
	x := iota32(2, 3, 4) // 24
	op := mkOp(t, "Reshape", 5, nil, 2, 1)
	out := run(t, op, x, i64t([]int{3}, 0, -1, 2))[0] // [2, 6, 2]
	if !out.Shape().Equal(tensor.Shape{2, 6, 2}) {
		t.Fatalf("reshape shape %v", out.Shape())
	}
}

func TestCastRoundtrip(t *testing.T) {
	x := f32t([]int{4}, 1.9, -2.1, 3.5, 0)
	toI := mkOp(t, "Cast", 6, Attrs{"to": {Kind: KindInt, I: 7}}, 1, 1) // to int64
	oi := run(t, toI, x)[0]
	eqI64(t, "castI64", oi, []int{4}, []int64{1, -2, 3, 0})
}

func TestWhere(t *testing.T) {
	cond := tensor.New(tensor.Bool, 2, 2)
	cb := cond.Bool()
	cb[0], cb[1], cb[2], cb[3] = true, false, false, true
	a := f32t([]int{2, 2}, 1, 2, 3, 4)
	b := f32t([]int{1}, 9)
	op := mkOp(t, "Where", 9, nil, 3, 1)
	out := run(t, op, cond, a, b)[0]
	eqF32(t, "where", out, []int{2, 2}, []float32{1, 9, 9, 4})
}

func TestExpand(t *testing.T) {
	x := f32t([]int{3, 1}, 1, 2, 3)
	op := mkOp(t, "Expand", 8, nil, 2, 1)
	out := run(t, op, x, i64t([]int{2}, 3, 4))[0]
	eqF32(t, "expand", out, []int{3, 4}, []float32{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3})
}

// TestConcatInnerAxis: an innermost-axis concat over many rows (the small-
// row batched path) with unequal widths, vs a direct oracle.
func TestConcatInnerAxis(t *testing.T) {
	const rows = 70001
	widths := []int{1, 3, 2}
	var in []*tensor.Tensor
	for i, w := range widths {
		x := tensor.New(tensor.F32, rows, w)
		for j := range x.F32() {
			x.F32()[j] = float32(i*1000000 + j)
		}
		in = append(in, x)
	}
	out := run(t, mkOp(t, "Concat", 4, Attrs{"axis": {Kind: KindInt, I: -1}}, 3, 1), in...)[0]
	W := 6
	if !out.Shape().Equal(tensor.Shape{rows, W}) {
		t.Fatalf("shape %v", out.Shape())
	}
	for r := range rows {
		c := 0
		for i, w := range widths {
			for k := range w {
				if got, want := out.F32()[r*W+c], float32(i*1000000+r*w+k); got != want {
					t.Fatalf("[%d,%d] = %g, want %g", r, c, got, want)
				}
				c++
			}
		}
	}
}

// BenchmarkConcat times the RoPE-shaped innermost concat ([T, H, 64, 1] ×
// 2 → [T, H, 64, 2] at T=1024, H=32).
func BenchmarkConcat(b *testing.B) {
	a, c := tensor.New(tensor.F32, 1024, 32, 64, 1), tensor.New(tensor.F32, 1024, 32, 64, 1)
	bld, _ := Lookup("", "Concat", 13)
	op, _ := bld(NodeInfo{Name: "c", OpType: "Concat", Version: 13, NumIn: 2, NumOut: 1, Attrs: Attrs{"axis": {Kind: KindInt, I: 3}}})
	ctx := &Ctx{Pool: tensor.NewPool()}
	b.Run("shape=1024x32x64x1+1", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			out, err := op.Run(ctx, []*tensor.Tensor{a, c})
			if err != nil {
				b.Fatal(err)
			}
			ctx.Pool.Put(out[0])
		}
	})
}
