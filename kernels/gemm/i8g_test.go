package gemm

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/kernels/vek"
)

// TestGemvI8G: the quantised GEMV against a float64 oracle over the
// dequantised weights and the split activations (the kernel's job),
// and the weight quantisation error (at most half a step per weight).
func TestGemvI8G(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	const n, k = 37, 192
	w := make([]float32, n*k)
	for i := range w {
		w[i] = (r.Float32()*2 - 1) * float32(1+i%7)
	}
	x := make([]float32, k)
	for i := range x {
		x[i] = r.Float32()*2 - 1
	}
	q := QuantizeI8GF32(w, k, n, k)
	y := make([]float32, n)
	for i := range y {
		y[i] = 1
	}
	GemvI8G(y, q, x, 0.5, 2)
	xh, xl, sx := make([]int8, k), make([]int8, k), make([]float32, k/64)
	vek.QuantizeX16(xh, xl, sx, x)
	for j := range n {
		var deq float64
		for c := range k {
			s := float64(q.S[j*k/64+c/64])
			xv := float64(sx[c/64]) * float64(128*int32(xh[c])+int32(xl[c]))
			deq += xv * s * float64(q.Q[j*k+c])
			if e := math.Abs(s*float64(q.Q[j*k+c]) - float64(w[j*k+c])); e > s/2+1e-6 {
				t.Fatalf("w[%d,%d] quantised error %g > half step %g", j, c, e, s/2)
			}
		}
		want := 0.5*deq + 2
		if d := math.Abs(float64(y[j]) - want); d > 1e-5*math.Sqrt(k)*(1+math.Abs(want)) {
			t.Fatalf("row %d: %g want %g", j, y[j], want)
		}
	}
}

// BenchmarkGemvI8G mirrors BenchmarkGemv's bf16 shapes.
func BenchmarkGemvI8G(b *testing.B) {
	r := rand.New(rand.NewPCG(7, 8))
	for _, sh := range []struct{ n, k int }{{4096, 4096}, {2048, 1024}, {1024, 1024}, {3072, 1024}, {1024, 3072}, {6144, 2048}, {2048, 6144}} {
		w := make([]float32, sh.n*sh.k)
		for i := range w {
			w[i] = r.Float32() - 0.5
		}
		x := make([]float32, sh.k)
		for i := range x {
			x[i] = r.Float32() - 0.5
		}
		q := QuantizeI8GF32(w, sh.k, sh.n, sh.k)
		y := make([]float32, sh.n)
		b.Run("n"+fmtInt(sh.n)+"_k"+fmtInt(sh.k), func(b *testing.B) {
			for b.Loop() {
				GemvI8G(y, q, x, 1, 0)
			}
			b.ReportMetric(float64(sh.n*sh.k)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GB/s")
		})
	}
}
