package ops

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

func u8t(shape []int, data ...uint8) *tensor.Tensor {
	t := tensor.New(tensor.U8, shape...)
	copy(t.U8(), data)
	return t
}

func i8t(shape []int, data ...int8) *tensor.Tensor {
	t := tensor.New(tensor.I8, shape...)
	copy(t.I8(), data)
	return t
}

func i32t(shape []int, data ...int32) *tensor.Tensor {
	t := tensor.New(tensor.I32, shape...)
	copy(t.I32(), data)
	return t
}

func TestQuantizeLinear(t *testing.T) {
	op := mkOp(t, "QuantizeLinear", 10, Attrs{}, 3, 1)
	// round-half-to-even and saturation, u8 zp 128
	x := f32t([]int{6}, 0, 0.5, 1.5, -300, 300, 2.4)
	out := run(t, op, x, f32t([]int{1}, 1), u8t([]int{1}, 128))[0]
	want := []uint8{128, 128, 130, 0, 255, 130}
	for i, w := range want {
		if out.U8()[i] != w {
			t.Fatalf("u8[%d]=%d want %d", i, out.U8()[i], w)
		}
	}
	// s8 output
	out = run(t, op, f32t([]int{3}, -3.5, 0.5, 200), f32t([]int{1}, 1), i8t([]int{1}, 0))[0]
	wantI := []int8{-4, 0, 127}
	for i, w := range wantI {
		if out.I8()[i] != w {
			t.Fatalf("i8[%d]=%d want %d", i, out.I8()[i], w)
		}
	}
	// per-axis (axis 1 default): [1,2,1,2] scales per channel
	opA := mkOp(t, "QuantizeLinear", 13, Attrs{}, 3, 1)
	x = f32t([]int{1, 2, 2}, 1, 2, 30, 40)
	out = run(t, opA, x, f32t([]int{2}, 1, 10), u8t([]int{2}, 0, 100))[0]
	wantA := []uint8{1, 2, 103, 104}
	for i, w := range wantA {
		if out.U8()[i] != w {
			t.Fatalf("axis[%d]=%d want %d", i, out.U8()[i], w)
		}
	}
}

func TestDequantizeLinear(t *testing.T) {
	op := mkOp(t, "DequantizeLinear", 10, Attrs{}, 3, 1)
	out := run(t, op, u8t([]int{3}, 0, 128, 255), f32t([]int{1}, 0.5), u8t([]int{1}, 128))[0]
	want := []float32{-64, 0, 63.5}
	for i, w := range want {
		if out.F32()[i] != w {
			t.Fatalf("[%d]=%g want %g", i, out.F32()[i], w)
		}
	}
	out = run(t, op, i8t([]int{2}, -128, 127), f32t([]int{1}, 2), i8t([]int{1}, -8))[0]
	if out.F32()[0] != -240 || out.F32()[1] != 270 {
		t.Fatalf("i8 dequant got %v", out.F32())
	}
}

func TestDynamicQuantizeLinear(t *testing.T) {
	op := mkOp(t, "DynamicQuantizeLinear", 11, Attrs{}, 1, 3)
	// ONNX spec example: x = [0, 2, -3, -2.5, 1.34, 0.5]
	outs := run(t, op, f32t([]int{6}, 0, 2, -3, -2.5, 1.34, 0.5))
	y, sc, zp := outs[0].U8(), outs[1].F32()[0], outs[2].U8()[0]
	if math.Abs(float64(sc)-0.0196078438) > 1e-6 || zp != 153 {
		t.Fatalf("scale %g zp %d", sc, zp)
	}
	// The spec example's 179 assumes exact arithmetic (25.5 rounds to even);
	// f32 legitimately lands one ulp below the tie. ±1 quantum, as everywhere
	// in quantized comparisons.
	want := []uint8{153, 255, 0, 26, 221, 179}
	for i, w := range want {
		if d := int(y[i]) - int(w); d < -1 || d > 1 {
			t.Fatalf("y[%d]=%d want %d±1", i, y[i], w)
		}
	}
}

// qlinearConvRef: independent integer reference for QLinearConv.
func qlinearConvRef(x []uint8, zx int32, w []int8, bias []int32, mult []float32, zy int32,
	N, C, H, W, M, KH, KW, G, sh, sw, pt, pl int) ([]uint8, int, int) {
	OH := (H+2*pt-KH)/sh + 1
	OW := (W+2*pl-KW)/sw + 1
	Cg, Mg := C/G, M/G
	out := make([]uint8, N*M*OH*OW)
	for n := 0; n < N; n++ {
		for m := 0; m < M; m++ {
			g := m / Mg
			for oh := 0; oh < OH; oh++ {
				for ow := 0; ow < OW; ow++ {
					var acc int64
					for cg := 0; cg < Cg; cg++ {
						c := g*Cg + cg
						for kh := 0; kh < KH; kh++ {
							for kw := 0; kw < KW; kw++ {
								ih, iw := oh*sh+kh-pt, ow*sw+kw-pl
								xv := zx // padded with zero point → contributes 0
								if ih >= 0 && ih < H && iw >= 0 && iw < W {
									xv = int32(x[((n*C+c)*H+ih)*W+iw])
								}
								wv := int32(w[((m*Cg+cg)*KH+kh)*KW+kw])
								acc += int64(xv-zx) * int64(wv)
							}
						}
					}
					if bias != nil {
						acc += int64(bias[m])
					}
					v := float32(acc)*mult[m] + float32(zy)
					out[((n*M+m)*OH+oh)*OW+ow] = satU8(v)
				}
			}
		}
	}
	return out, OH, OW
}

func TestQLinearConv(t *testing.T) {
	r := rand.New(rand.NewPCG(23, 24))
	type cfg struct{ N, C, H, W, M, KH, KW, G, sh, sw, pt, pl int }
	for _, c := range []cfg{
		{1, 3, 8, 9, 4, 3, 3, 1, 1, 1, 1, 1},
		{2, 4, 7, 7, 6, 3, 3, 1, 2, 2, 0, 0},
		{1, 8, 6, 6, 8, 3, 3, 8, 1, 1, 1, 1},   // depthwise 3x3 s1 (SIMD)
		{1, 6, 9, 11, 6, 5, 5, 6, 1, 1, 2, 2},  // depthwise 5x5 s1 (SIMD)
		{2, 8, 10, 9, 8, 3, 3, 8, 2, 2, 1, 1},  // depthwise 3x3 s2 (SIMD, deinterleave)
		{1, 6, 11, 12, 6, 5, 5, 6, 2, 2, 2, 2}, // depthwise 5x5 s2 (SIMD)
		{1, 4, 8, 8, 4, 3, 3, 4, 1, 1, 0, 0},   // depthwise no-pad
		{1, 6, 5, 5, 4, 1, 1, 2, 1, 1, 0, 0},   // grouped pointwise
		{1, 16, 9, 9, 8, 3, 3, 1, 1, 1, 1, 1},
	} {
		x := make([]uint8, c.N*c.C*c.H*c.W)
		for i := range x {
			x[i] = uint8(r.UintN(256))
		}
		w := make([]int8, c.M*(c.C/c.G)*c.KH*c.KW)
		for i := range w {
			w[i] = int8(r.UintN(256))
		}
		bias := make([]int32, c.M)
		for i := range bias {
			bias[i] = int32(r.UintN(2048)) - 1024
		}
		zx, zy := int32(120), int32(17)
		xs, ys := float32(0.02), float32(0.4)
		wsc := make([]float32, c.M)
		mult := make([]float32, c.M)
		for m := range wsc {
			wsc[m] = 0.01 + float32(m)*0.003
			mult[m] = xs * wsc[m] / ys
		}
		want, OH, OW := qlinearConvRef(x, zx, w, bias, mult, zy, c.N, c.C, c.H, c.W, c.M, c.KH, c.KW, c.G, c.sh, c.sw, c.pt, c.pl)
		op := mkOp(t, "QLinearConv", 10, Attrs{
			"group":        {Kind: KindInt, I: int64(c.G)},
			"strides":      {Kind: KindInts, Ints: []int64{int64(c.sh), int64(c.sw)}},
			"pads":         {Kind: KindInts, Ints: []int64{int64(c.pt), int64(c.pl), int64(c.pt), int64(c.pl)}},
			"kernel_shape": {Kind: KindInts, Ints: []int64{int64(c.KH), int64(c.KW)}},
		}, 9, 1)
		got := run(t, op,
			u8t([]int{c.N, c.C, c.H, c.W}, x...),
			f32t([]int{1}, xs), u8t([]int{1}, uint8(zx)),
			i8t([]int{c.M, c.C / c.G, c.KH, c.KW}, w...),
			f32t([]int{c.M}, wsc...), i8t([]int{c.M}, make([]int8, c.M)...),
			f32t([]int{1}, ys), u8t([]int{1}, uint8(zy)),
			i32t([]int{c.M}, bias...))[0]
		gf := got.U8()
		if !got.Shape().Equal(tensor.Shape{c.N, c.M, OH, OW}) {
			t.Fatalf("shape %v", got.Shape())
		}
		for i := range want {
			d := int(gf[i]) - int(want[i])
			if d < -1 || d > 1 { // float requant may differ by 1 LSB at ties
				t.Fatalf("cfg %+v: y[%d]=%d want %d", c, i, gf[i], want[i])
			}
		}
	}
}

func TestQLinearMatMulAndMatMulInteger(t *testing.T) {
	r := rand.New(rand.NewPCG(25, 26))
	m, k, n := 7, 33, 12
	a := make([]uint8, m*k)
	for i := range a {
		a[i] = uint8(r.UintN(256))
	}
	b := make([]int8, k*n)
	for i := range b {
		b[i] = int8(r.UintN(256))
	}
	za := int32(110)
	raw := make([]int32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s int32
			for p := 0; p < k; p++ {
				s += (int32(a[i*k+p]) - za) * int32(b[p*n+j])
			}
			raw[i*n+j] = s
		}
	}
	// MatMulInteger
	op := mkOp(t, "MatMulInteger", 10, Attrs{}, 4, 1)
	got := run(t, op, u8t([]int{m, k}, a...), i8t([]int{k, n}, b...), u8t([]int{1}, uint8(za)), i8t([]int{1}, 0))[0]
	for i := range raw {
		if got.I32()[i] != raw[i] {
			t.Fatalf("MatMulInteger[%d]=%d want %d", i, got.I32()[i], raw[i])
		}
	}
	// QLinearMatMul
	sa, sb, sy := float32(0.1), float32(0.05), float32(0.7)
	zy := int32(3)
	opQ := mkOp(t, "QLinearMatMul", 10, Attrs{}, 8, 1)
	gotQ := run(t, opQ,
		u8t([]int{m, k}, a...), f32t([]int{1}, sa), u8t([]int{1}, uint8(za)),
		i8t([]int{k, n}, b...), f32t([]int{1}, sb), i8t([]int{1}, 0),
		f32t([]int{1}, sy), u8t([]int{1}, uint8(zy)))[0]
	for i := range raw {
		want := satU8(float32(raw[i])*(sa*sb/sy) + float32(zy))
		d := int(gotQ.U8()[i]) - int(want)
		if d < -1 || d > 1 {
			t.Fatalf("QLinearMatMul[%d]=%d want %d", i, gotQ.U8()[i], want)
		}
	}
}
