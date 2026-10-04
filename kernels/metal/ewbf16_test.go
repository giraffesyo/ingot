//go:build darwin && arm64

package metal

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestBinaryBcastBF16: every binary op over [rows, cols] with a trailing
// vector, and a per-row scalar, written as bf16 — each result must be the
// f32 kernel's result rounded to nearest-even bf16 (what a cast pass after
// the f32 kernel would have stored).
func TestBinaryBcastBF16(t *testing.T) {
	d := openDev(t)
	if err := d.PrepareEW(); err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(17, 17))
	const rows, cols = 9, 37
	const n = rows * cols
	a, b, rs := buf(t, d, n), buf(t, d, cols), buf(t, d, rows)
	fill(r, a)
	fill(r, b)
	fill(r, rs)
	ref, o16 := buf(t, d, n), buf(t, d, n) // o16 is oversized: bf16 fills its first half
	bf16 := func(i int) float32 {
		raw := o16.Bytes()
		return math.Float32frombits(uint32(raw[2*i])<<16 | uint32(raw[2*i+1])<<24)
	}
	for _, op := range []int{OpAdd, OpSub, OpMul, OpDiv, OpMax, OpMin} {
		for _, perRow := range []bool{false, true} {
			y, db, mb := b, 1, cols
			if perRow {
				y, db, mb = rs, cols, rows
			}
			if err := d.Run(func(e *Encoder) {
				e.BinaryBcast(op, a.At(0), y.At(0), ref.At(0), n, 1, n, db, mb)
				e.BinaryBcastBF16(op, a.At(0), y.At(0), o16.At(0), n, 1, n, db, mb)
			}); err != nil {
				t.Fatal(err)
			}
			for i, want := range f32s(ref.Bytes()) {
				if got := bf16(i); float64(got) != bf16Round(float64(want)) {
					t.Fatalf("op %d perRow=%v [%d]: bf16 %g, f32 %g rounds to %g", op, perRow, i, got, want, bf16Round(float64(want)))
				}
			}
		}
	}
}

// BenchmarkBinaryBcast: a modulated activation (4096 tokens × 1536 channels
// plus a per-channel shift) written as f32 and then cast for a bf16 GEMM,
// vs written as bf16 directly.
func BenchmarkBinaryBcast(b *testing.B) {
	d := openDev(b)
	for _, p := range []func() error{d.Prepare, d.PrepareEW} {
		if err := p(); err != nil {
			b.Fatal(err)
		}
	}
	const rows, cols = 4096, 1536
	const n = rows * cols
	nb := func(bytes int) *Buffer {
		x, err := d.NewBuffer(bytes)
		if err != nil {
			b.Fatal(err)
		}
		b.Cleanup(x.Release)
		return x
	}
	x, shift, o, o16 := nb(4*n), nb(4*cols), nb(4*n), nb(2*n)
	run := func(name string, enc func(e *Encoder)) {
		b.Run(name+"/shape=4096x1536", func(b *testing.B) {
			b.SetBytes(4 * n)
			for i := 0; i < b.N; i++ {
				if err := d.Run(enc); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
	run("f32+cast", func(e *Encoder) {
		e.BinaryBcast(OpAdd, x.At(0), shift.At(0), o.At(0), n, 1, n, 1, cols)
		e.CastBF16(o.At(0), o16.At(0), rows, cols, cols, cols)
	})
	run("bf16", func(e *Encoder) {
		e.BinaryBcastBF16(OpAdd, x.At(0), shift.At(0), o16.At(0), n, 1, n, 1, cols)
	})
}
