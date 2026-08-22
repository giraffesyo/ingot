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
		Sigmoid(out, wide)
		for i := range want {
			want[i] = 1 / (1 + expScalar(-wide[i]))
		}
		eqRel(t, "Sigmoid", out, want, 4e-7)
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
	run("Sigmoid", func() { Sigmoid(out, x) })
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
