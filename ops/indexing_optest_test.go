package ops

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

func i64T(v []int64, shape ...int) *tensor.Tensor {
	t := tensor.New(tensor.I64, shape...)
	copy(t.I64(), v)
	return t
}

func approx(t *testing.T, what string, got []float32, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i])-want[i]) > 1e-4*(1+math.Abs(want[i])) {
			t.Fatalf("%s [%d] = %g, want %g", what, i, got[i], want[i])
		}
	}
}

// TestGatherScatterND: GatherND (plain, negative indices, batch_dims) and
// ScatterND (none, add) vs explicit loops.
func TestGatherScatterND(t *testing.T) {
	r := rand.New(rand.NewPCG(41, 41))
	data := randT(r, 2, 3, 4)
	d := data.F32()
	at := func(a, b, c int) float64 { return float64(d[(a*3+b)*4+c]) }

	// indices [3,2] into dims (0,1): slices of 4.
	idx := []int64{1, 2, 0, -1, -2, 0}
	got := run(t, mkOp(t, "GatherND", 13, nil, 2, 1), data, i64T(idx, 3, 2))[0]
	var want []float64
	for q := 0; q < 3; q++ {
		a, b := int(idx[2*q]), int(idx[2*q+1])
		if a < 0 {
			a += 2
		}
		if b < 0 {
			b += 3
		}
		for c := 0; c < 4; c++ {
			want = append(want, at(a, b, c))
		}
	}
	approx(t, "GatherND", got.F32(), want)

	// batch_dims=1: indices [2,2,1] pick dim 1 per batch → [2,2,4].
	bi := []int64{2, 0, 1, 1}
	got = run(t, mkOp(t, "GatherND", 13, Attrs{"batch_dims": {Kind: KindInt, I: 1}}, 2, 1), data, i64T(bi, 2, 2, 1))[0]
	want = want[:0]
	for bb := 0; bb < 2; bb++ {
		for q := 0; q < 2; q++ {
			for c := 0; c < 4; c++ {
				want = append(want, at(bb, int(bi[bb*2+q]), c))
			}
		}
	}
	approx(t, "GatherND batch_dims", got.F32(), want)

	// ScatterND: full-index updates [2] at (0,1,2), (1,0,3).
	for _, red := range []string{"none", "add"} {
		upd := randT(r, 2)
		si := []int64{0, 1, 2, 1, 0, 3}
		op := mkOp(t, "ScatterND", 16, Attrs{"reduction": {Kind: KindString, S: red}}, 3, 1)
		got = run(t, op, data, i64T(si, 2, 3), upd)[0]
		w := make([]float64, len(d))
		for i, v := range d {
			w[i] = float64(v)
		}
		for q := 0; q < 2; q++ {
			o := (int(si[3*q])*3+int(si[3*q+1]))*4 + int(si[3*q+2])
			if red == "add" {
				w[o] += float64(upd.F32()[q])
			} else {
				w[o] = float64(upd.F32()[q])
			}
		}
		approx(t, "ScatterND "+red, got.F32(), w)
	}
}

// TestGatherScatterElements: both along axis 1 with negative indices;
// ScatterElements none and add (duplicate targets accumulate).
func TestGatherScatterElements(t *testing.T) {
	r := rand.New(rand.NewPCG(42, 42))
	data := randT(r, 3, 5)
	idx := []int64{4, 0, -1, 2, 2, 1}
	got := run(t, mkOp(t, "GatherElements", 13, Attrs{"axis": {Kind: KindInt, I: 1}}, 2, 1), data, i64T(idx, 3, 2))[0]
	var want []float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			c := int(idx[i*2+j])
			if c < 0 {
				c += 5
			}
			want = append(want, float64(data.F32()[i*5+c]))
		}
	}
	approx(t, "GatherElements", got.F32(), want)
	for _, red := range []string{"none", "add"} {
		upd := randT(r, 3, 2)
		op := mkOp(t, "ScatterElements", 16, Attrs{"axis": {Kind: KindInt, I: 1}, "reduction": {Kind: KindString, S: red}}, 3, 1)
		got = run(t, op, data, i64T(idx, 3, 2), upd)[0]
		w := make([]float64, 15)
		for i, v := range data.F32() {
			w[i] = float64(v)
		}
		for i := 0; i < 3; i++ {
			for j := 0; j < 2; j++ {
				c := int(idx[i*2+j])
				if c < 0 {
					c += 5
				}
				if red == "add" {
					w[i*5+c] += float64(upd.F32()[i*2+j])
				} else {
					w[i*5+c] = float64(upd.F32()[i*2+j])
				}
			}
		}
		approx(t, "ScatterElements "+red, got.F32(), w)
	}
}

// TestOneHotMod: OneHot on both axes (negative and out-of-range indices),
// integer Mod (divisor's sign) and fmod (dividend's sign).
func TestOneHotMod(t *testing.T) {
	vals := tensor.FromF32([]float32{-1, 7}, 2)
	depth := i64T([]int64{4}, 1)
	idx := i64T([]int64{0, 3, -1, 9}, 2, 2)
	for _, axis := range []int64{-1, 0} {
		got := run(t, mkOp(t, "OneHot", 11, Attrs{"axis": {Kind: KindInt, I: axis}}, 3, 1), idx, depth, vals)[0]
		gs := got.Shape()
		for p, v := range []int64{0, 3, -1, 9} {
			hot := int(v)
			if hot < 0 {
				hot += 4
			}
			for k := 0; k < 4; k++ {
				var e int
				if axis == -1 {
					e = p*4 + k // [2,2,4]
				} else {
					e = k*4 + p // [4,2,2]
				}
				want := float32(-1)
				if k == hot && v < 4 {
					want = 7
				}
				if got.F32()[e] != want {
					t.Fatalf("OneHot axis %d shape %v: idx %d k %d = %g, want %g", axis, gs, v, k, got.F32()[e], want)
				}
			}
		}
	}
	a, b := i64T([]int64{7, -7, 7, -7}, 4), i64T([]int64{3, 3, -3, -3}, 4)
	for _, c := range []struct {
		fmod int64
		want []int64
	}{{0, []int64{1, 2, -2, -1}}, {1, []int64{1, -1, 1, -1}}} {
		got := run(t, mkOp(t, "Mod", 13, Attrs{"fmod": {Kind: KindInt, I: c.fmod}}, 2, 1), a, b)[0].I64()
		for i, w := range c.want {
			if got[i] != w {
				t.Fatalf("Mod fmod=%d [%d] = %d, want %d", c.fmod, i, got[i], w)
			}
		}
	}
	fa, fb := tensor.FromF32([]float32{5.5, -5.5}, 2), tensor.FromF32([]float32{2}, 1)
	got := run(t, mkOp(t, "Mod", 13, Attrs{"fmod": {Kind: KindInt, I: 1}}, 2, 1), fa, fb)[0].F32()
	if got[0] != 1.5 || got[1] != -1.5 {
		t.Fatalf("fmod = %v, want [1.5 -1.5]", got)
	}
}

// TestEinsum: the GEMM fast path vs the direct evaluator, and the direct
// evaluator vs explicit loops, over matmul-family, transpose, trace,
// diagonal, reduction and broadcast equations.
func TestEinsum(t *testing.T) {
	r := rand.New(rand.NewPCG(43, 43))
	for _, c := range []struct {
		eq     string
		shapes [][]int
	}{
		{"bmd,bnd->bmn", [][]int{{2, 5, 7}, {2, 6, 7}}},
		{"ij,jk->ik", [][]int{{4, 9}, {9, 3}}},
		{"bhid,bhjd->bhij", [][]int{{1, 3, 4, 8}, {1, 3, 5, 8}}},
		{"ij,jk->ki", [][]int{{4, 6}, {6, 5}}},
		{"i,i->", [][]int{{10}, {10}}},
		{"ij,kj->ikj", [][]int{{3, 4}, {2, 4}}},
		{"ij->ji", [][]int{{3, 5}}},
		{"ii->i", [][]int{{4, 4}}},
		{"ii", [][]int{{5, 5}}},
		{"ij->", [][]int{{3, 4}}},
		{"bij,bjk->bik", [][]int{{1, 3, 4}, {2, 4, 5}}}, // broadcast batch
	} {
		ins := make([]*tensor.Tensor, len(c.shapes))
		for i, s := range c.shapes {
			ins[i] = randT(r, s...)
		}
		op := mkOp(t, "Einsum", 12, Attrs{"equation": {Kind: KindString, S: c.eq}}, len(ins), 1)
		got := run(t, op, ins...)[0]
		// The direct evaluator on the same inputs (the GEMM path's oracle).
		eo := op.(*einsumOp)
		dims := map[rune]int{}
		for i, s := range eo.ins {
			for k, ch := range s {
				dims[ch] = max(dims[ch], ins[i].Shape()[k])
			}
		}
		direct := make([]float32, got.Numel())
		einsumDirect(eo.ins, eo.out, ins, dims, direct)
		w := make([]float64, len(direct))
		for i, v := range direct {
			w[i] = float64(v)
		}
		approx(t, c.eq+" vs direct", got.F32(), w)
	}
	// The direct evaluator itself vs explicit loops.
	a, b := randT(r, 3, 4), randT(r, 4, 2)
	dims := map[rune]int{'i': 3, 'j': 4, 'k': 2}
	out := make([]float32, 6)
	einsumDirect([]string{"ij", "jk"}, "ik", []*tensor.Tensor{a, b}, dims, out)
	var want []float64
	for i := 0; i < 3; i++ {
		for k := 0; k < 2; k++ {
			var s float64
			for j := 0; j < 4; j++ {
				s += float64(a.F32()[i*4+j]) * float64(b.F32()[j*2+k])
			}
			want = append(want, s)
		}
	}
	approx(t, "direct ij,jk", out, want)
	sq := randT(r, 4, 4)
	tr := make([]float32, 1)
	einsumDirect([]string{"ii"}, "", []*tensor.Tensor{sq}, map[rune]int{'i': 4}, tr)
	var trace float64
	for i := 0; i < 4; i++ {
		trace += float64(sq.F32()[i*5])
	}
	approx(t, "direct trace", tr, []float64{trace})
	// Unsupported equations fail at load.
	b2, _ := Lookup("", "Einsum", 12)
	if _, err := b2(NodeInfo{OpType: "Einsum", Attrs: Attrs{"equation": {Kind: KindString, S: "...ij,...jk->...ik"}}}); err == nil {
		t.Fatal("ellipsis equation: want a load error")
	}
}

// TestReduceInt: integer ReduceMax/Min/Sum over a middle axis vs loops.
func TestReduceInt(t *testing.T) {
	x := tensor.New(tensor.I32, 2, 3, 2)
	for i := range x.I32() {
		x.I32()[i] = int32((i*7)%11 - 5)
	}
	for kind, f := range map[string]func(a, b int32) int32{
		"ReduceMax": func(a, b int32) int32 { return max(a, b) },
		"ReduceMin": func(a, b int32) int32 { return min(a, b) },
		"ReduceSum": func(a, b int32) int32 { return a + b },
	} {
		got := run(t, mkOp(t, kind, 11, Attrs{"axes": {Kind: KindInts, Ints: []int64{1}}, "keepdims": {Kind: KindInt, I: 0}}, 1, 1), x)[0]
		if got.DType() != tensor.I32 || !got.Shape().Equal(tensor.Shape{2, 2}) {
			t.Fatalf("%s: %s %v", kind, got.DType(), got.Shape())
		}
		for a := 0; a < 2; a++ {
			for c := 0; c < 2; c++ {
				w := x.I32()[(a*3)*2+c]
				for b := 1; b < 3; b++ {
					w = f(w, x.I32()[(a*3+b)*2+c])
				}
				if g := got.I32()[a*2+c]; g != w {
					t.Fatalf("%s [%d,%d] = %d, want %d", kind, a, c, g, w)
				}
			}
		}
	}
}

// TestBinaryInt: integer Max/Min/Add broadcast, int64 and int32 (dtype kept).
func TestBinaryInt(t *testing.T) {
	a := i64T([]int64{5, -2, 7, 0}, 2, 2)
	b := i64T([]int64{3, 1}, 2)
	for name, f := range map[string]func(x, y int64) int64{
		"Max": func(x, y int64) int64 { return max(x, y) },
		"Min": func(x, y int64) int64 { return min(x, y) },
		"Add": func(x, y int64) int64 { return x + y },
	} {
		got := run(t, mkOp(t, name, 13, nil, 2, 1), a, b)[0].I64()
		for i, x := range a.I64() {
			if w := f(x, b.I64()[i%2]); got[i] != w {
				t.Fatalf("%s [%d] = %d, want %d", name, i, got[i], w)
			}
		}
	}
	a32, b32 := tensor.New(tensor.I32, 3), tensor.New(tensor.I32, 3)
	copy(a32.I32(), []int32{4, -9, 2})
	copy(b32.I32(), []int32{1, 5, 2})
	got := run(t, mkOp(t, "Max", 13, nil, 2, 1), a32, b32)[0]
	if got.DType() != tensor.I32 || got.I32()[0] != 4 || got.I32()[1] != 5 || got.I32()[2] != 2 {
		t.Fatalf("int32 Max = %s %v", got.DType(), got.I32())
	}
}

// TestScalarBroadcastRank: a rank-0 × [1] product broadcasts to [1] (NumPy
// rules), whichever side the scalar is on.
func TestScalarBroadcastRank(t *testing.T) {
	s, one := tensor.FromF32([]float32{3}), tensor.FromF32([]float32{2}, 1)
	for _, pair := range [][2]*tensor.Tensor{{s, one}, {one, s}} {
		got := run(t, mkOp(t, "Mul", 14, nil, 2, 1), pair[0], pair[1])[0]
		if !got.Shape().Equal(tensor.Shape{1}) || got.F32()[0] != 6 {
			t.Fatalf("Mul(%v, %v) = %v %v, want [1] [6]", pair[0].Shape(), pair[1].Shape(), got.Shape(), got.F32())
		}
	}
}
