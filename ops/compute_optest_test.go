package ops

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// convRef is an independent, fully-general NCHW conv oracle (groups, stride,
// dilation, explicit pads).
func convRef(x, w, bias []float32, N, C, H, W, M, KH, KW, G, sh, sw, dh, dw, pt, pl int) ([]float32, int, int) {
	Cg := C / G
	Mg := M / G
	OH := (H+2*pt-(dh*(KH-1)+1))/sh + 1
	OW := (W+2*pl-(dw*(KW-1)+1))/sw + 1
	out := make([]float32, N*M*OH*OW)
	for n := 0; n < N; n++ {
		for m := 0; m < M; m++ {
			g := m / Mg
			for oh := 0; oh < OH; oh++ {
				for ow := 0; ow < OW; ow++ {
					var acc float32
					if bias != nil {
						acc = bias[m]
					}
					for cg := 0; cg < Cg; cg++ {
						c := g*Cg + cg
						for kh := 0; kh < KH; kh++ {
							ih := oh*sh + kh*dh - pt
							if ih < 0 || ih >= H {
								continue
							}
							for kw := 0; kw < KW; kw++ {
								iw := ow*sw + kw*dw - pl
								if iw < 0 || iw >= W {
									continue
								}
								acc += x[((n*C+c)*H+ih)*W+iw] * w[((m*Cg+cg)*KH+kh)*KW+kw]
							}
						}
					}
					out[((n*M+m)*OH+oh)*OW+ow] = acc
				}
			}
		}
	}
	return out, OH, OW
}

func TestConvVariants(t *testing.T) {
	// Exercise the Winograd path too (opt-in at runtime, always tested).
	old := winogradEnabled
	winogradEnabled = true
	defer func() { winogradEnabled = old }()
	r := rand.New(rand.NewPCG(1, 2))
	rnd := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = r.Float32()*2 - 1
		}
		return s
	}
	type cfg struct {
		N, C, H, W, M, KH, KW, G, sh, sw, dh, dw, pt, pl int
	}
	cfgs := []cfg{
		{1, 4, 10, 12, 6, 3, 3, 1, 1, 1, 1, 1, 1, 1}, // basic pad-same
		{2, 4, 9, 9, 4, 3, 3, 1, 2, 2, 1, 1, 1, 1},   // stride 2
		{1, 4, 11, 11, 8, 3, 3, 1, 1, 1, 2, 2, 2, 2}, // dilation 2
		{1, 6, 8, 8, 6, 3, 3, 6, 1, 1, 1, 1, 1, 1},   // depthwise
		{1, 8, 7, 9, 8, 3, 3, 4, 1, 1, 1, 1, 1, 1},   // grouped (G=4)
		{1, 3, 12, 12, 5, 5, 5, 1, 2, 2, 1, 1, 2, 2}, // 5x5 stride2
		{1, 4, 6, 6, 4, 1, 1, 1, 1, 1, 1, 1, 0, 0},   // pointwise
		{1, 4, 8, 8, 4, 3, 3, 1, 1, 1, 1, 1, 0, 0},   // valid pad
		{1, 6, 11, 13, 6, 5, 5, 6, 2, 2, 1, 1, 2, 2}, // depthwise 5x5 stride 2
		{2, 8, 10, 17, 8, 3, 3, 8, 2, 2, 1, 1, 1, 1}, // depthwise 3x3 stride 2, odd width
		{1, 6, 9, 9, 6, 3, 3, 6, 2, 2, 1, 1, 0, 0},   // depthwise 3x3 stride 2, no pad
		// Winograd F(2,3) path (3x3, s1, groups 1, Cin>=16):
		{1, 16, 12, 14, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, // even/odd extents, pad 1
		{2, 24, 9, 11, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1}, // batch 2, odd OH/OW
		{1, 16, 8, 8, 4, 3, 3, 1, 1, 1, 1, 1, 0, 0},   // valid pad
		{1, 32, 5, 5, 8, 3, 3, 1, 1, 1, 1, 1, 2, 2},   // pad 2 (output > input)
	}
	for ci, c := range cfgs {
		x := rnd(c.N * c.C * c.H * c.W)
		w := rnd(c.M * (c.C / c.G) * c.KH * c.KW)
		bias := rnd(c.M)
		want, OH, OW := convRef(x, w, bias, c.N, c.C, c.H, c.W, c.M, c.KH, c.KW, c.G, c.sh, c.sw, c.dh, c.dw, c.pt, c.pl)
		op := mkOp(t, "Conv", 1, Attrs{
			"group":        {Kind: KindInt, I: int64(c.G)},
			"strides":      {Kind: KindInts, Ints: []int64{int64(c.sh), int64(c.sw)}},
			"dilations":    {Kind: KindInts, Ints: []int64{int64(c.dh), int64(c.dw)}},
			"pads":         {Kind: KindInts, Ints: []int64{int64(c.pt), int64(c.pl), int64(c.pt), int64(c.pl)}},
			"kernel_shape": {Kind: KindInts, Ints: []int64{int64(c.KH), int64(c.KW)}},
		}, 3, 1)
		out := run(t, op,
			tensor.FromF32(x, c.N, c.C, c.H, c.W),
			tensor.FromF32(w, c.M, c.C/c.G, c.KH, c.KW),
			tensor.FromF32(bias, c.M))[0]
		eqF32(t, "conv", out, []int{c.N, c.M, OH, OW}, want)
		_ = ci
	}
}

func TestConvSamePad(t *testing.T) {
	// SAME_UPPER should preserve spatial size for stride 1.
	x := iota32(1, 1, 5, 5)
	w := f32t([]int{1, 1, 3, 3}, 0, 0, 0, 0, 1, 0, 0, 0, 0) // identity
	op := mkOp(t, "Conv", 1, Attrs{"auto_pad": {Kind: KindString, S: "SAME_UPPER"}, "kernel_shape": {Kind: KindInts, Ints: []int64{3, 3}}}, 2, 1)
	out := run(t, op, x, w)[0]
	if !out.Shape().Equal(tensor.Shape{1, 1, 5, 5}) {
		t.Fatalf("same pad shape %v", out.Shape())
	}
	// identity kernel -> output equals input
	eqF32(t, "convsame", out, []int{1, 1, 5, 5}, x.F32())
}

func TestMaxPoolCeilMode(t *testing.T) {
	// 5x5 input, 2x2 pool stride 2: floor -> 2x2, ceil -> 3x3.
	x := iota32(1, 1, 5, 5)
	floor := mkOp(t, "MaxPool", 1, Attrs{"kernel_shape": {Kind: KindInts, Ints: []int64{2, 2}}, "strides": {Kind: KindInts, Ints: []int64{2, 2}}}, 1, 1)
	of := run(t, floor, x)[0]
	if !of.Shape().Equal(tensor.Shape{1, 1, 2, 2}) {
		t.Fatalf("floor pool shape %v", of.Shape())
	}
	eqF32(t, "poolfloor", of, []int{1, 1, 2, 2}, []float32{6, 8, 16, 18})
	ceil := mkOp(t, "MaxPool", 1, Attrs{"kernel_shape": {Kind: KindInts, Ints: []int64{2, 2}}, "strides": {Kind: KindInts, Ints: []int64{2, 2}}, "ceil_mode": {Kind: KindInt, I: 1}}, 1, 1)
	oc := run(t, ceil, x)[0]
	if !oc.Shape().Equal(tensor.Shape{1, 1, 3, 3}) {
		t.Fatalf("ceil pool shape %v", oc.Shape())
	}
	// last row/col are edge maxes: [6,8,9; 16,18,19; 21,23,24]
	eqF32(t, "poolceil", oc, []int{1, 1, 3, 3}, []float32{6, 8, 9, 16, 18, 19, 21, 23, 24})
}

func TestAvgPoolCountIncludePad(t *testing.T) {
	x := f32t([]int{1, 1, 2, 2}, 1, 2, 3, 4)
	// 3x3 pool, pad 1, stride 1 -> output 2x2. Corner (0,0) covers 1 real value.
	excl := mkOp(t, "AveragePool", 1, Attrs{"kernel_shape": {Kind: KindInts, Ints: []int64{3, 3}}, "pads": {Kind: KindInts, Ints: []int64{1, 1, 1, 1}}}, 1, 1)
	oe := run(t, excl, x)[0]
	// exclude pad: (0,0) window has 4 valid cells (1,2,3,4)/4 = 2.5
	if math.Abs(float64(oe.F32()[0])-2.5) > 1e-5 {
		t.Fatalf("avg exclude pad (0,0) = %g want 2.5", oe.F32()[0])
	}
	incl := mkOp(t, "AveragePool", 1, Attrs{"kernel_shape": {Kind: KindInts, Ints: []int64{3, 3}}, "pads": {Kind: KindInts, Ints: []int64{1, 1, 1, 1}}, "count_include_pad": {Kind: KindInt, I: 1}}, 1, 1)
	oi := run(t, incl, x)[0]
	// include pad: sum(1+2+3+4)=10 / 9 = 1.111
	if math.Abs(float64(oi.F32()[0])-10.0/9) > 1e-4 {
		t.Fatalf("avg include pad (0,0) = %g want %g", oi.F32()[0], 10.0/9)
	}
}

func TestReduceAxesKeepdims(t *testing.T) {
	x := iota32(2, 3, 4)
	// ReduceSum over axis 1, keepdims 0
	op := mkOp(t, "ReduceSum", 13, Attrs{"keepdims": {Kind: KindInt, I: 0}}, 2, 1)
	out := run(t, op, x, i64t([]int{1}, 1))[0]
	if !out.Shape().Equal(tensor.Shape{2, 4}) {
		t.Fatalf("reduce shape %v", out.Shape())
	}
	// out[n,w] = sum over c of x[n,c,w]
	xf := x.F32()
	want := make([]float32, 8)
	for n := 0; n < 2; n++ {
		for w := 0; w < 4; w++ {
			var s float32
			for c := 0; c < 3; c++ {
				s += xf[(n*3+c)*4+w]
			}
			want[n*4+w] = s
		}
	}
	eqF32(t, "reducesum", out, []int{2, 4}, want)
}

func TestReduceMeanNegAxisKeep(t *testing.T) {
	x := iota32(2, 4)
	op := mkOp(t, "ReduceMean", 18, Attrs{"keepdims": {Kind: KindInt, I: 1}}, 2, 1)
	out := run(t, op, x, i64t([]int{1}, -1))[0]
	eqF32(t, "reducemean", out, []int{2, 1}, []float32{1.5, 5.5})
}

func TestSoftmaxAxis(t *testing.T) {
	x := f32t([]int{2, 3}, 1, 2, 3, 1, 1, 1)
	op := mkOp(t, "Softmax", 13, Attrs{"axis": {Kind: KindInt, I: 1}}, 1, 1)
	out := run(t, op, x)[0].F32()
	// row0 softmax(1,2,3), row1 uniform 1/3
	e := func(v float32) float64 { return math.Exp(float64(v)) }
	s := e(1) + e(2) + e(3)
	if math.Abs(float64(out[0])-e(1)/s) > 1e-5 || math.Abs(float64(out[5])-1.0/3) > 1e-5 {
		t.Fatalf("softmax got %v", out)
	}
}

func TestSoftmaxMiddleAxis(t *testing.T) {
	// axis in the middle (strided) exercises the gather path.
	x := iota32(2, 3, 2)
	op := mkOp(t, "Softmax", 13, Attrs{"axis": {Kind: KindInt, I: 1}}, 1, 1)
	out := run(t, op, x)[0].F32()
	// each column over the 3 middle entries must sum to 1
	for n := 0; n < 2; n++ {
		for w := 0; w < 2; w++ {
			var s float32
			for c := 0; c < 3; c++ {
				s += out[(n*3+c)*2+w]
			}
			if math.Abs(float64(s)-1) > 1e-5 {
				t.Fatalf("softmax col sum %g", s)
			}
		}
	}
}

func TestLayerNorm(t *testing.T) {
	x := f32t([]int{1, 4}, 1, 2, 3, 4)
	scale := f32t([]int{4}, 1, 1, 1, 1)
	bias := f32t([]int{4}, 0, 0, 0, 0)
	op := mkOp(t, "LayerNormalization", 17, Attrs{"axis": {Kind: KindInt, I: -1}, "epsilon": {Kind: KindFloat, F: 0}}, 3, 1)
	out := run(t, op, x, scale, bias)[0].F32()
	// mean 2.5, var 1.25, std sqrt(1.25)
	std := math.Sqrt(1.25)
	for i, v := range []float64{1, 2, 3, 4} {
		want := (v - 2.5) / std
		if math.Abs(float64(out[i])-want) > 1e-4 {
			t.Fatalf("layernorm[%d]=%g want %g", i, out[i], want)
		}
	}
}

func TestGemmTransB(t *testing.T) {
	// A[2x3] * B^T where B is [4x3] -> [2x4], plus bias, alpha/beta.
	a := iota32(2, 3)
	b := iota32(4, 3)
	c := f32t([]int{4}, 100, 200, 300, 400)
	op := mkOp(t, "Gemm", 7, Attrs{"transB": {Kind: KindInt, I: 1}, "alpha": {Kind: KindFloat, F: 2}, "beta": {Kind: KindFloat, F: 1}}, 3, 1)
	out := run(t, op, a, b, c)[0]
	// oracle
	af, bf, cf := a.F32(), b.F32(), c.F32()
	want := make([]float32, 8)
	for i := 0; i < 2; i++ {
		for j := 0; j < 4; j++ {
			var acc float32
			for k := 0; k < 3; k++ {
				acc += af[i*3+k] * bf[j*3+k]
			}
			want[i*4+j] = 2*acc + cf[j]
		}
	}
	eqF32(t, "gemmTransB", out, []int{2, 4}, want)
}

func TestMatMulBatched(t *testing.T) {
	// [2,2,3] x [2,3,2] -> [2,2,2] with distinct batches
	a := iota32(2, 2, 3)
	b := iota32(2, 3, 2)
	op := mkOp(t, "MatMul", 1, nil, 2, 1)
	out := run(t, op, a, b)[0]
	af, bf := a.F32(), b.F32()
	want := make([]float32, 8)
	for bt := 0; bt < 2; bt++ {
		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				var acc float32
				for k := 0; k < 3; k++ {
					acc += af[(bt*2+i)*3+k] * bf[(bt*3+k)*2+j]
				}
				want[(bt*2+i)*2+j] = acc
			}
		}
	}
	eqF32(t, "matmulBatch", out, []int{2, 2, 2}, want)
}

// TestMatMulBroadcastBatch: a 2-D rhs shared across A's batch dims (the
// Linear-on-a-3-D/4-D-tensor form, collapsed to one GEMM) against a float64
// oracle, including the sequence-first [T, 1, K] shape whose per-batch M is 1
// and a rank-4 batch.
func TestMatMulBroadcastBatch(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for _, c := range []struct{ as, bs []int }{
		{[]int{2, 2, 3}, []int{3, 2}},
		{[]int{7, 1, 384}, []int{384, 96}},
		{[]int{3, 5, 17}, []int{17, 9}},
		{[]int{2, 3, 4, 33}, []int{33, 5}},
	} {
		a := tensor.New(tensor.F32, c.as...)
		b := tensor.New(tensor.F32, c.bs...)
		for i := range a.F32() {
			a.F32()[i] = rng.Float32()*2 - 1
		}
		for i := range b.F32() {
			b.F32()[i] = rng.Float32()*2 - 1
		}
		op := mkOp(t, "MatMul", 1, nil, 2, 1)
		out := run(t, op, a, b)[0]
		K, N := c.bs[0], c.bs[1]
		rows := a.Numel() / K
		want := make([]float32, rows*N)
		af, bf := a.F32(), b.F32()
		for r := 0; r < rows; r++ {
			for j := 0; j < N; j++ {
				var acc float64
				for k := 0; k < K; k++ {
					acc += float64(af[r*K+k]) * float64(bf[k*N+j])
				}
				want[r*N+j] = float32(acc)
			}
		}
		ws := append(append([]int{}, c.as[:len(c.as)-1]...), N)
		eqF32(t, "matmulBcast", out, ws, want)
	}
}

func TestArgMax(t *testing.T) {
	x := f32t([]int{2, 3}, 1, 5, 2, 9, 0, 3)
	op := mkOp(t, "ArgMax", 1, Attrs{"axis": {Kind: KindInt, I: 1}, "keepdims": {Kind: KindInt, I: 0}}, 1, 1)
	out := run(t, op, x)[0]
	eqI64(t, "argmax", out, []int{2}, []int64{1, 0})
}

// TestMatMulBias: the optimizer-attached bias input, on the packed-B path
// (constant weight, nb == 1 and the flattened batch) and the fallback
// (broadcast batch on B), against a float64 oracle.
func TestMatMulBias(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	for _, c := range []struct{ as, bs []int }{
		{[]int{128, 384}, []int{384, 1152}},
		{[]int{1, 16, 48}, []int{48, 144}},
		{[]int{7, 1, 384}, []int{384, 96}},
		{[]int{5, 13}, []int{13, 7}},
		{[]int{2, 3, 4, 33}, []int{33, 5}},
		{[]int{2, 8, 32}, []int{2, 32, 96}}, // batched rhs: fallback add
	} {
		a := tensor.New(tensor.F32, c.as...)
		b := tensor.New(tensor.F32, c.bs...)
		N := c.bs[len(c.bs)-1]
		bias := tensor.New(tensor.F32, N)
		for i := range a.F32() {
			a.F32()[i] = rng.Float32()*2 - 1
		}
		for i := range b.F32() {
			b.F32()[i] = rng.Float32()*2 - 1
		}
		for i := range bias.F32() {
			bias.F32()[i] = rng.Float32()*2 - 1
		}
		op := mkOp(t, "MatMul", 1, nil, 3, 1)
		out := run(t, op, a, b, bias)[0]
		plain := run(t, mkOp(t, "MatMul", 1, nil, 2, 1), a, b)[0]
		pf, of := plain.F32(), out.F32()
		K := c.bs[len(c.bs)-2]
		for i := range of {
			want := float64(pf[i]) + float64(bias.F32()[i%N])
			if math.Abs(float64(of[i])-want) > 1e-5*math.Sqrt(float64(K))*(1+math.Abs(want)) {
				t.Fatalf("%v·%v+bias[%d] = %g, want %g", c.as, c.bs, i, of[i], want)
			}
		}
		// Second run reuses the packed weight and must be identical.
		again := run(t, op, a, b, bias)[0]
		for i := range of {
			if again.F32()[i] != of[i] {
				t.Fatalf("run 2 differs at %d", i)
			}
		}
	}
}

// TestAddLayerNorm: the fused residual+LayerNorm against Add followed by
// LayerNormalization, both outputs.
func TestAddLayerNorm(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 22))
	for _, sh := range [][]int{{1, 5, 384}, {3, 7, 48}, {2, 3, 4, 33}} {
		D := sh[len(sh)-1]
		a := tensor.New(tensor.F32, sh...)
		b := tensor.New(tensor.F32, sh...)
		sc := tensor.New(tensor.F32, D)
		bs := tensor.New(tensor.F32, D)
		for _, tt := range []*tensor.Tensor{a, b, sc, bs} {
			for i := range tt.F32() {
				tt.F32()[i] = rng.Float32()*4 - 2
			}
		}
		attrs := Attrs{"axis": {Kind: KindInt, I: -1}, "epsilon": {Kind: KindFloat, F: 1e-5}}
		bld, err := Lookup("ingot", "AddLayerNorm", 1)
		if err != nil {
			t.Fatal(err)
		}
		fused, err := bld(NodeInfo{Name: "aln", OpType: "AddLayerNorm", Domain: "ingot", Version: 1, Attrs: attrs, NumIn: 4, NumOut: 2})
		if err != nil {
			t.Fatal(err)
		}
		outs := run(t, fused, a, b, sc, bs)
		sum := run(t, mkOp(t, "Add", 14, nil, 2, 1), a, b)[0]
		want := run(t, mkOp(t, "LayerNormalization", 17, attrs, 3, 1), sum, sc, bs)[0]
		eqF32(t, "sum", outs[0], sh, sum.F32())
		eqF32(t, "normed", outs[1], sh, want.F32())
	}
}
