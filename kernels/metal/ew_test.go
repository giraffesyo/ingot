//go:build darwin && arm64

package metal

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestEW: binary broadcast ops, unary ops, transpose, row reduce and the
// Stream (all recorded across Encode calls, one Flush) vs oracles.
func TestEW(t *testing.T) {
	d := openDev(t)
	if err := d.PrepareEW(); err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(13, 13))
	const n, nb = 6 * 35, 35
	a, b, o := buf(t, d, n), buf(t, d, nb), buf(t, d, n)
	af, bf := fill(r, a), fill(r, b)
	u, uo := buf(t, d, n), buf(t, d, n)
	uf := fill(r, u)
	dims, perm := []int{2, 3, 5, 7}, []int{3, 1, 0, 2}
	x, xt := buf(t, d, 210), buf(t, d, 210)
	xf := fill(r, x)
	rr, ro := buf(t, d, 7*333), buf(t, d, 7)
	rf := fill(r, rr)
	s := d.NewStream()
	check := func(what string, got float32, want float64, tol float64) {
		t.Helper()
		if math.Abs(float64(got)-want) > tol*(1+math.Abs(want)) || (math.IsNaN(float64(got)) != math.IsNaN(want)) {
			t.Fatalf("%s: %g, want %g", what, got, want)
		}
	}
	for op, fn := range []func(x, y float64) float64{
		func(x, y float64) float64 { return x + y }, func(x, y float64) float64 { return x - y },
		func(x, y float64) float64 { return x * y }, func(x, y float64) float64 { return x / y },
		func(x, y float64) float64 { return math.Pow(x, 2) }, math.Max, math.Min,
	} {
		bb := b
		if op == OpPow { // integer exponent with negative bases
			bb = buf(t, d, 1)
			f32s(bb.Bytes())[0] = 2
		}
		s.Encode(func(e *Encoder) {
			e.Binary(op, a.At(0), bb.At(0), o.At(0), n, n, map[bool]int{true: 1, false: nb}[op == OpPow])
		})
		if err := s.Flush(); err != nil {
			t.Fatal(err)
		}
		for i := range n {
			check("binary", f32s(o.Bytes())[i], fn(float64(af[i]), float64(bf[i%nb])), 1e-5)
		}
	}
	for op, fn := range []func(float64) float64{
		func(v float64) float64 { return math.Max(v, 0) }, func(v float64) float64 { return 1 / (1 + math.Exp(-v)) },
		math.Tanh, func(v float64) float64 { return 0.5 * v * (1 + math.Erf(v/math.Sqrt2)) },
		func(v float64) float64 { return 0.5 * v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*v*v*v))) },
		func(v float64) float64 { return v / (1 + math.Exp(-v)) }, math.Sqrt, func(v float64) float64 { return 1 / v },
		math.Exp, func(v float64) float64 { return -v }, math.Abs, math.Erf,
	} {
		s.Encode(func(e *Encoder) { e.Unary(op, u.At(0), uo.At(0), n) })
		if err := s.Flush(); err != nil {
			t.Fatal(err)
		}
		for i := range n {
			check("unary", f32s(uo.Bytes())[i], fn(float64(uf[i])), 2e-6)
		}
	}
	s.Encode(func(e *Encoder) { e.Transpose(x.At(0), xt.At(0), dims, perm) })
	s.Encode(func(e *Encoder) { e.ReduceRows(rr.At(0), ro.At(0), 7, 333, 333, true) })
	if err := s.Flush(); err != nil {
		t.Fatal(err)
	}
	od := []int{dims[3], dims[1], dims[0], dims[2]}
	for i0 := range od[0] {
		for i1 := range od[1] {
			for i2 := range od[2] {
				for i3 := range od[3] {
					src := ((i2*dims[1]+i1)*dims[2]+i3)*dims[3] + i0 // input coords (i2, i1, i3, i0)
					if f32s(xt.Bytes())[((i0*od[1]+i1)*od[2]+i2)*od[3]+i3] != xf[src] {
						t.Fatal("transpose")
					}
				}
			}
		}
	}
	for row := range 7 {
		var sum float64
		for c := range 333 {
			sum += float64(rf[row*333+c])
		}
		check("reduce mean", f32s(ro.Bytes())[row], sum/333, 1e-5)
	}
}
