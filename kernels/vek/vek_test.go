package vek

import (
	"math"
	"math/rand/v2"
	"testing"
)

func refBin(op byte, a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range out {
		switch op {
		case '+':
			out[i] = a[i] + b[i]
		case '-':
			out[i] = a[i] - b[i]
		case '*':
			out[i] = a[i] * b[i]
		case '/':
			out[i] = a[i] / b[i]
		case 'x':
			out[i] = max(a[i], b[i])
		case 'n':
			out[i] = min(a[i], b[i])
		}
	}
	return out
}

func randf(r *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = r.Float32()*4 - 2
	}
	return x
}

func eq(t *testing.T, name string, got, want []float32) {
	t.Helper()
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("%s[%d]: got %g want %g", name, i, got[i], want[i])
		}
	}
}

func TestVek(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	// Cover all tail residues and cross the 16-lane unroll boundary.
	for _, n := range []int{0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 64, 100, 257, 1000} {
		a, b := randf(r, n), randf(r, n)
		out := make([]float32, n)
		Add(out, a, b)
		eq(t, "Add", out, refBin('+', a, b))
		Sub(out, a, b)
		eq(t, "Sub", out, refBin('-', a, b))
		Mul(out, a, b)
		eq(t, "Mul", out, refBin('*', a, b))
		Div(out, a, b)
		eq(t, "Div", out, refBin('/', a, b))
		MaxPair(out, a, b)
		eq(t, "MaxPair", out, refBin('x', a, b))
		MinPair(out, a, b)
		eq(t, "MinPair", out, refBin('n', a, b))

		want := make([]float32, n)
		Relu(out, a)
		for i := range want {
			want[i] = max(a[i], 0)
		}
		eq(t, "Relu", out, want)

		HardSwish(out, a)
		for i := range want {
			want[i] = a[i] * clamp01(a[i]/6+0.5)
		}
		eq(t, "HardSwish", out, want)

		HardSigmoid(out, a, 0.2, 0.5)
		for i := range want {
			want[i] = clamp01(0.2*a[i] + 0.5)
		}
		eq(t, "HardSigmoid", out, want)

		Clip(out, a, -0.5, 1.5)
		for i := range want {
			want[i] = min(max(a[i], -0.5), 1.5)
		}
		eq(t, "Clip", out, want)

		LeakyRelu(out, a, 0.1)
		for i := range want {
			v := a[i]
			if v < 0 {
				v *= 0.1
			}
			want[i] = v
		}
		eq(t, "LeakyRelu", out, want)

		AddScalar(out, a, 3)
		for i := range want {
			want[i] = a[i] + 3
		}
		eq(t, "AddScalar", out, want)

		MulScalar(out, a, 3)
		for i := range want {
			want[i] = a[i] * 3
		}
		eq(t, "MulScalar", out, want)

		acc := randf(r, n)
		copy(out, acc)
		Axpy(out, a, 1.7)
		for i := range want {
			want[i] = acc[i] + 1.7*a[i]
		}
		eq(t, "Axpy", out, want)

		// Exp/Sigmoid over a wide range (incl. the saturation bounds); the
		// SIMD polynomial is compared at a relative 4e-7.
		wide := make([]float32, n)
		for i := range wide {
			wide[i] = a[i] * 30
		}
		if n > 8 {
			copy(wide, []float32{-100, -87.4, -87.3, -20, -1e-3, 0, 1e-3, 20, 88.3, 88.4, 100}[:min(n, 11)])
		}
		Exp(out, wide)
		for i := range want {
			want[i] = expScalar(wide[i])
		}
		eqRel(t, "Exp", out, want, 4e-7)
		// Sigmoid/SiLU may flush subnormal sigmoid outputs to zero (amd64
		// does — a subnormal operand in a downstream multiply is ~100
		// cycles on x86); mirror the flush in the reference.
		flush := func(v float32) float32 {
			if v < 1.17549435e-38 {
				return 0
			}
			return v
		}
		Sigmoid(out, wide)
		for i := range want {
			want[i] = 1 / (1 + expScalar(-wide[i]))
			if out[i] == 0 {
				want[i] = flush(want[i])
			}
		}
		eqRel(t, "Sigmoid", out, want, 4e-7)
		SiLU(out, wide)
		for i := range want {
			sg := 1 / (1 + expScalar(-wide[i]))
			if sgf := flush(sg); sgf == 0 && out[i] == 0 {
				sg = 0
			}
			want[i] = wide[i] * sg
		}
		eqRel(t, "SiLU", out, want, 4e-7)

		// Erf / Gelu: absolute error bound (A&S 7.1.26 is 1.5e-7 absolute).
		erfIn := make([]float32, n)
		for i := range erfIn {
			erfIn[i] = a[i] * 6
		}
		Erf(out, erfIn)
		for i := range want {
			want[i] = float32(math.Erf(float64(erfIn[i])))
		}
		eqAbs(t, "Erf", out, want, 1e-6)
		Gelu(out, erfIn)
		for i := range want {
			x := float64(erfIn[i])
			want[i] = float32(0.5 * x * (1 + math.Erf(x/math.Sqrt2)))
		}
		eqAbs(t, "Gelu", out, want, 2e-6)
	}
}

func eqAbs(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("%s[%d]: got %g want %g", name, i, got[i], want[i])
		}
	}
}

func eqRel(t *testing.T, name string, got, want []float32, rel float64) {
	t.Helper()
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > rel*math.Abs(float64(want[i]))+1e-38 {
			t.Fatalf("%s[%d]: got %g want %g", name, i, got[i], want[i])
		}
	}
}

func TestAlias(t *testing.T) {
	// dst == src must work (in-place).
	r := rand.New(rand.NewPCG(3, 4))
	a := randf(r, 100)
	want := make([]float32, 100)
	for i := range want {
		want[i] = max(a[i], 0)
	}
	Relu(a, a)
	eq(t, "ReluInPlace", a, want)
}

func BenchmarkVek(b *testing.B) {
	r := rand.New(rand.NewPCG(5, 6))
	const n = 200704 // 16*112*112, a real early-layer activation size
	x, y := randf(r, n), randf(r, n)
	out := make([]float32, n)
	run := func(name string, f func()) {
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				f()
			}
			b.ReportMetric(float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "Gelem/s")
		})
	}
	run("HardSwish", func() { HardSwish(out, x) })
	run("Relu", func() { Relu(out, x) })
	run("Mul", func() { Mul(out, x, y) })
	run("Add", func() { Add(out, x, y) })
	run("Exp", func() { Exp(out, x) })
	run("Dot", func() { _ = Dot(x, y) })
	run("SiLU", func() { SiLU(out, x) })
	run("Erf", func() { Erf(out, x) })
	run("Gelu", func() { Gelu(out, x) })
	run("Sigmoid", func() { Sigmoid(out, x) })
	s8 := randf(r, 8)
	run("MulBlk8", func() { MulBlk8(out, x, s8) })
	var acc [8]float32
	run("SumBlk8", func() { SumBlk8(acc[:], x) })
}

// TestDwRowS1 checks every depthwise row kernel shape against a scalar
// reference over all tail residues.
func TestDwRowS1(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	for _, k := range [][2]int{{3, 3}, {5, 5}, {3, 2}, {3, 1}, {5, 3}, {5, 2}, {4, 4}} {
		KH, KW := k[0], k[1]
		for _, ncols := range []int{0, 1, 3, 4, 7, 8, 9, 16, 17, 33, 100} {
			W := ncols + KW + 3
			src := randf(r, KH*W)
			w := randf(r, KH*KW)
			wp := make([]float32, (KH*KW+7)/8*8)
			copy(wp, w)
			bias := randf(r, ncols)
			got := append([]float32(nil), bias...)
			DwRowS1(got, src, wp, ncols, W, KH, KW)
			want := append([]float32(nil), bias...)
			for c := 0; c < ncols; c++ {
				var mag float64
				for kh := 0; kh < KH; kh++ {
					for kw := 0; kw < KW; kw++ {
						term := w[kh*KW+kw] * src[kh*W+c+kw]
						want[c] += term
						mag += math.Abs(float64(term))
					}
				}
				// FMA vs separate multiply/add reorder the rounding; bound the
				// error by the magnitude of the summed terms, not the (possibly
				// cancelled) result.
				if d := math.Abs(float64(got[c] - want[c])); d > 2e-6*(1+mag) {
					t.Fatalf("DwRowS1_%dx%d_n%d[%d]: got %g want %g (mag %g)", KH, KW, ncols, c, got[c], want[c], mag)
				}
			}
		}
	}
}

func TestDot(t *testing.T) {
	r := rand.New(rand.NewPCG(13, 14))
	for _, n := range []int{0, 1, 3, 15, 16, 17, 31, 32, 33, 63, 64, 100, 1000, 4097} {
		a, b := randf(r, n), randf(r, n)
		var want, mag float64
		for i := range a {
			want += float64(a[i]) * float64(b[i])
			mag += math.Abs(float64(a[i]) * float64(b[i]))
		}
		got := Dot(a, b)
		if d := math.Abs(float64(got) - want); d > 1e-6*(1+mag) {
			t.Fatalf("Dot n=%d: got %g want %g", n, got, want)
		}
	}
}

func TestQuantKernels(t *testing.T) {
	r := rand.New(rand.NewPCG(31, 32))
	for _, n := range []int{0, 1, 15, 16, 17, 33, 257, 1000} {
		f := make([]float32, n)
		for i := range f {
			f[i] = r.Float32()*600 - 300
		}
		scale, zp := float32(1.7), float32(19)
		u := make([]uint8, n)
		QuantU8(u, f, scale, zp)
		s8 := make([]int8, n)
		QuantI8(s8, f, scale, zp-30)
		for i := range f {
			if w := qSatU8(f[i]/scale + zp); u[i] != w {
				t.Fatalf("QuantU8[%d]=%d want %d", i, u[i], w)
			}
			if w := qSatI8(f[i]/scale + zp - 30); s8[i] != w {
				t.Fatalf("QuantI8[%d]=%d want %d", i, s8[i], w)
			}
		}
		acc := make([]int32, n)
		for i := range acc {
			acc[i] = int32(r.UintN(200000)) - 100000
		}
		mult, off := float32(0.00137), float32(101.5)
		RequantU8(u, acc, mult, off, 777)
		RequantI8(s8, acc, mult, off-120, -3333)
		for i := range acc {
			if w := qSatU8(float32(acc[i]+777)*mult + off); u[i] != w {
				t.Fatalf("RequantU8[%d]=%d want %d", i, u[i], w)
			}
			if w := qSatI8(float32(acc[i]-3333)*mult + off - 120); s8[i] != w {
				t.Fatalf("RequantI8[%d]=%d want %d", i, s8[i], w)
			}
		}
		sh := make([]int8, n)
		ShiftU8S8(sh, u)
		for i := range u {
			if w := int8(int32(u[i]) - 128); sh[i] != w {
				t.Fatalf("ShiftU8S8[%d]=%d want %d", i, sh[i], w)
			}
		}
		for i := range u {
			u[i] = uint8(r.UintN(256))
			s8[i] = int8(r.UintN(256))
		}
		out := make([]float32, n)
		DequantU8(out, u, 0.31, 117)
		for i := range u {
			if w := float32(int32(u[i])-117) * 0.31; out[i] != w {
				t.Fatalf("DequantU8[%d]=%g want %g", i, out[i], w)
			}
		}
		DequantI8(out, s8, 0.031, -5)
		for i := range s8 {
			if w := float32(int32(s8[i])+5) * 0.031; out[i] != w {
				t.Fatalf("DequantI8[%d]=%g want %g", i, out[i], w)
			}
		}
	}
}

func TestQDwRowS1(t *testing.T) {
	r := rand.New(rand.NewPCG(41, 42))
	for _, k := range [][2]int{{3, 3}, {5, 5}, {3, 2}, {3, 1}, {5, 3}, {5, 2}, {4, 4}} {
		KH, KW := k[0], k[1]
		for _, ncols := range []int{0, 1, 7, 8, 9, 16, 17, 33, 100} {
			W := ncols + KW + 3
			src := make([]int16, KH*W)
			for i := range src {
				src[i] = int16(r.UintN(256)) - 128
			}
			wp := make([]int16, (KH*KW+7)/8*8)
			for i := 0; i < KH*KW; i++ {
				wp[i] = int16(r.UintN(256)) - 128
			}
			got := make([]int32, ncols)
			want := make([]int32, ncols)
			for i := range got {
				got[i] = int32(i) - 5
				want[i] = got[i]
			}
			QDwRowS1(got, src, wp, ncols, W, KH, KW)
			qdwTail(want, src, wp, 0, ncols, W, KH, KW)
			for c := range want {
				if got[c] != want[c] {
					t.Fatalf("%dx%d n=%d acc[%d]=%d want %d", KH, KW, ncols, c, got[c], want[c])
				}
			}
		}
	}
}

func TestQLut(t *testing.T) {
	var tab [256]uint8
	for i := range tab {
		tab[i] = uint8((i*7 + 13) % 256)
	}
	for _, n := range []int{0, 1, 15, 16, 17, 64, 255, 4096} {
		src := make([]uint8, n)
		for i := range src {
			i8 := uint8(i * 31)
			src[i] = i8
		}
		dst := make([]uint8, n)
		QLut(dst, src, &tab)
		for i := range src {
			if dst[i] != tab[src[i]] {
				t.Fatalf("n=%d: QLut[%d]=%d want %d", n, i, dst[i], tab[src[i]])
			}
		}
	}
}

func TestZip2(t *testing.T) {
	for _, n := range []int{0, 1, 7, 8, 9, 33, 257} {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range a {
			a[i], b[i] = float32(i), float32(-i)
		}
		dst := make([]float32, 2*n)
		Zip2(dst, a, b, 0.5)
		for i := 0; i < n; i++ {
			if dst[2*i] != a[i]+0.5 || dst[2*i+1] != b[i]+0.5 {
				t.Fatalf("n=%d i=%d: got (%v,%v) want (%v,%v)", n, i, dst[2*i], dst[2*i+1], a[i]+0.5, b[i]+0.5)
			}
		}
	}
}

func TestWidenDeint(t *testing.T) {
	for _, n := range []int{0, 1, 15, 16, 17, 100} {
		src := make([]int8, n)
		for i := range src {
			src[i] = int8(i*13 - 60)
		}
		dst := make([]int16, n)
		WidenS8S16(dst, src)
		for i := range src {
			if dst[i] != int16(src[i]) {
				t.Fatalf("widen[%d]=%d", i, dst[i])
			}
		}
		s16 := make([]int16, n)
		for i := range s16 {
			s16[i] = int16(i*7 - 300)
		}
		ev := make([]int16, n/2)
		od := make([]int16, n/2)
		DeinterleaveS16(ev, od, s16)
		for i := 0; i < n/2; i++ {
			if ev[i] != s16[2*i] || od[i] != s16[2*i+1] {
				t.Fatalf("deint[%d]", i)
			}
		}
	}
}

func TestBF16DotAxpy(t *testing.T) {
	for _, n := range []int{0, 1, 15, 16, 33, 100, 256} {
		a := make([]float32, n)
		b := make([]uint16, n)
		dst := make([]float32, n)
		ref := make([]float32, n)
		for i := range a {
			a[i] = float32(i%17)*0.25 - 2
			b[i] = uint16(0x3f80 + i%64) // bf16 values near 1
			dst[i] = float32(i % 5)
			ref[i] = dst[i]
		}
		var want float32
		for i := 0; i < n; i++ {
			want += a[i] * math.Float32frombits(uint32(b[i])<<16)
		}
		if got := DotBF16(a, b); math.Abs(float64(got-want)) > 1e-4*math.Max(1, math.Abs(float64(want))) {
			t.Fatalf("DotBF16 n=%d: got %g want %g", n, got, want)
		}
		AxpyBF16(dst, b, 0.5)
		for i := 0; i < n; i++ {
			ref[i] += 0.5 * math.Float32frombits(uint32(b[i])<<16)
			if math.Abs(float64(dst[i]-ref[i])) > 1e-5 {
				t.Fatalf("AxpyBF16 n=%d i=%d: got %g want %g", n, i, dst[i], ref[i])
			}
		}
	}
}

func TestDwBlk8S1(t *testing.T) {
	for _, v := range []struct{ K, S int }{{3, 1}, {3, 2}, {5, 1}, {5, 2}} {
		for _, tc := range []struct{ ncols, wp int }{{1, 8}, {4, 12}, {14, 32}, {30, 64}} {
			src := make([]float32, (v.K*tc.wp+tc.ncols*v.S+v.K)*8)
			w := make([]float32, v.K*v.K*8)
			for i := range src {
				src[i] = float32(i%23)*0.1 - 1
			}
			for i := range w {
				w[i] = float32(i%7)*0.25 - 0.5
			}
			got := make([]float32, tc.ncols*8)
			want := make([]float32, tc.ncols*8)
			DwBlk8(got, src, w, tc.ncols, tc.wp, v.K, v.S)
			dwBlk8Ref(want, src, w, tc.ncols, tc.wp, v.K, v.S)
			for i := range want {
				if d := got[i] - want[i]; d > 1e-4 || d < -1e-4 {
					t.Fatalf("K%dS%d ncols=%d i=%d: got %g want %g", v.K, v.S, tc.ncols, i, got[i], want[i])
				}
			}
		}
	}
}

func TestPwBlk6x16(t *testing.T) {
	for _, cin := range []int{8, 16, 96, 576} {
		nb := cin / 8
		const pos = 6
		x := make([]float32, nb*pos*8)
		w := make([]float32, cin*16)
		for i := range x {
			x[i] = float32(i%19)*0.05 - 0.4
		}
		for i := range w {
			w[i] = float32(i%23)*0.04 - 0.5
		}
		g0 := make([]float32, pos*8)
		g1 := make([]float32, pos*8)
		w0 := make([]float32, pos*8)
		w1 := make([]float32, pos*8)
		PwBlk6x16(g0, g1, x, w, cin, pos*8*4)
		pwBlkRef(w0, w1, x, w, cin, pos*8*4)
		for i := range w0 {
			if d := g0[i] - w0[i]; d > 1e-3 || d < -1e-3 {
				t.Fatalf("cin=%d dst0[%d]: got %g want %g", cin, i, g0[i], w0[i])
			}
			if d := g1[i] - w1[i]; d > 1e-3 || d < -1e-3 {
				t.Fatalf("cin=%d dst1[%d]: got %g want %g", cin, i, g1[i], w1[i])
			}
		}
	}
}

func TestMulBlk8(t *testing.T) {
	r := rand.New(rand.NewPCG(41, 42))
	for _, n := range []int{0, 8, 16, 24, 32, 40, 200, 1568, 4096, 7, 9, 23, 100} {
		src, s := randf(r, n), randf(r, 8)
		dst := make([]float32, n)
		MulBlk8(dst, src, s)
		for i := range src {
			want := src[i] * s[i&7]
			if dst[i] != want {
				t.Fatalf("MulBlk8 n=%d i=%d: got %g want %g", n, i, dst[i], want)
			}
		}
	}
}

func TestSumBlk8(t *testing.T) {
	r := rand.New(rand.NewPCG(43, 44))
	for _, n := range []int{0, 8, 16, 24, 32, 200, 1568, 4096, 7, 9, 23, 100} {
		src := randf(r, n)
		dst := randf(r, 8) // must be overwritten
		SumBlk8(dst, src)
		var want [8]float64
		var mag [8]float64
		for i, v := range src {
			want[i&7] += float64(v)
			mag[i&7] += math.Abs(float64(v))
		}
		for l := 0; l < 8; l++ {
			if d := math.Abs(float64(dst[l]) - want[l]); d > 1e-6*(1+mag[l]) {
				t.Fatalf("SumBlk8 n=%d lane=%d: got %g want %g", n, l, dst[l], want[l])
			}
		}
	}
}
