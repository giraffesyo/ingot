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
	// Per-row scalar broadcast ([6, 35] * [6, 1]) and Copy2D / GatherRowsConst.
	rs := buf(t, d, 6)
	rsf := fill(r, rs)
	ob := buf(t, d, n)
	c2 := buf(t, d, 6*50)
	gi := buf(t, d, 3*35)
	s.Encode(func(e *Encoder) {
		e.BinaryBcast(OpMul, a.At(0), rs.At(0), ob.At(0), n, 1, n, 35, 6)
		e.Copy2D(a.At(0), c2.At(4*10), 6, 35, 35, 50)
		e.GatherRowsConst(a.At(0), gi.At(0), []uint32{5, 0, 2}, 35)
	})
	if err := s.Flush(); err != nil {
		t.Fatal(err)
	}
	for i := range n {
		check("row broadcast", f32s(ob.Bytes())[i], float64(af[i])*float64(rsf[i/35]), 1e-6)
		if f32s(c2.Bytes())[(i/35)*50+10+i%35] != af[i] {
			t.Fatal("copy2d")
		}
	}
	for k, src := range []int{5, 0, 2} {
		for c := range 35 {
			if f32s(gi.Bytes())[k*35+c] != af[src*35+c] {
				t.Fatal("gather const")
			}
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

// TestND: the strided N-d path — broadcast binary (non-contiguous operand
// axes), a reversed strided slice copy, and where — vs direct indexing.
func TestND(t *testing.T) {
	d := openDev(t)
	if err := d.PrepareEW(); err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(21, 21))
	dims := []int{3, 4, 5}
	n := 60
	// a [3,1,5] and b [1,4,1] broadcast to [3,4,5].
	a, b, o := buf(t, d, 15), buf(t, d, 4), buf(t, d, n)
	af, bf := fill(r, a), fill(r, b)
	// x [6,8]: slice rows 5,3,1 (step -2) and cols 1..7 step 3 → [3, 3].
	x, xo := buf(t, d, 48), buf(t, d, 9)
	xf := fill(r, x)
	cb, err := d.NewBuffer(n)
	if err != nil {
		t.Fatal(err)
	}
	defer cb.Release()
	cond := cb.Bytes()
	for i := range cond {
		cond[i] = byte(r.IntN(2))
	}
	wo := buf(t, d, n)
	err = d.Run(func(e *Encoder) {
		e.BinaryND(OpSub, a.At(0), b.At(0), o.At(0), dims, []int{5, 0, 1}, []int{0, 1, 0})
		e.CopyND(x.At(4*(5*8+1)), xo.At(0), []int{3, 3}, []int{-16, 3})
		e.WhereND(cb.At(0), o.At(0), a.At(0), wo.At(0), dims, []int{20, 5, 1}, []int{20, 5, 1}, []int{5, 0, 1})
	})
	if err != nil {
		t.Fatal(err)
	}
	of, wf := f32s(o.Bytes()), f32s(wo.Bytes())
	for i := range 3 {
		for j := range 4 {
			for k := range 5 {
				idx := (i*4+j)*5 + k
				want := af[i*5+k] - bf[j]
				if of[idx] != want {
					t.Fatalf("binary [%d %d %d] = %g, want %g", i, j, k, of[idx], want)
				}
				ww := af[i*5+k]
				if cond[idx] != 0 {
					ww = want
				}
				if wf[idx] != ww {
					t.Fatalf("where [%d %d %d] = %g, want %g", i, j, k, wf[idx], ww)
				}
			}
		}
	}
	for i := range 3 {
		for j := range 3 {
			if got, want := f32s(xo.Bytes())[i*3+j], xf[(5-2*i)*8+1+3*j]; got != want {
				t.Fatalf("slice [%d %d] = %g, want %g", i, j, got, want)
			}
		}
	}
}
