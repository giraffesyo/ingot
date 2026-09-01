package gemm

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestMInvariance: c[i,j] must be bit-identical whether row i is computed in
// a small-M or large-M call (batching must not change results — the OCR rec
// batch test enforces this end to end). Every m >= 2 routes through the panel
// sweeps, whose kernels share one accumulation grouping: sums built from
// zeroed accumulators, C added at the store (never seeded into the FMA
// chain), so full panels, edge tiles, and paired panels all agree.
//
// KNOWN EXEMPTION: m == 1 takes the GEMV fast path, which streams B with its
// own accumulation order — single-row calls are not bit-comparable to the
// same row inside an m >= 2 call. No product requirement compares them
// (decode-vs-prefill equality is not asserted anywhere); documented here so
// the exemption is a decision, not an accident.
func TestMInvariance(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	const k, n = 512, 6656
	for _, ms := range [][2]int{{127, 381}, {6, 384}, {96, 1024}, {100, 144}, {145, 381}, {144, 145}, {2, 8}, {2, 144}, {7, 1024}} {
		m1, m2 := ms[0], ms[1]
		a := make([]float32, m2*k)
		b := make([]float32, k*n)
		for i := range a {
			a[i] = r.Float32()*2 - 1
		}
		for i := range b {
			b[i] = r.Float32()*2 - 1
		}
		c1 := make([]float32, m1*n)
		c2 := make([]float32, m2*n)
		Sgemm(m1, n, k, 1, a, k, b, n, 0, c1, n)
		Sgemm(m2, n, k, 1, a, k, b, n, 0, c2, n)
		bad := 0
		for i := 0; i < m1*n; i++ {
			if math.Float32bits(c1[i]) != math.Float32bits(c2[i]) {
				bad++
			}
		}
		if bad > 0 {
			t.Errorf("m=%d vs m=%d: %d/%d shared elements differ bitwise", m1, m2, bad, m1*n)
		}
	}
}
