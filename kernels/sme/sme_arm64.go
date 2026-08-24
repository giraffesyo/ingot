//go:build arm64

package sme

func svl() int64
func probePeak(iters int64, src *float32)
func probeLoad(iters int64, a, b *float32)
func outerK(a, b *float32, k int64, out *float32)
func probeN(iters int64, src *float32, variant int64)
func zakernel(kc int64, ap, bp, c *float32, ldc4 int64)

// SVL returns the streaming vector length in bytes (0 if SME is unavailable).
func SVL() int {
	if !Available() {
		return 0
	}
	return int(svl())
}

// ProbePeak runs iters × 4 FMOPA f32 outer products from resident registers.
func ProbePeak(iters int64, src []float32) { probePeak(iters, &src[0]) }

// ProbeLoad runs iters × (2 loads + 4 FMOPA).
func ProbeLoad(iters int64, a, b []float32) { probeLoad(iters, &a[0], &b[0]) }

// OuterK computes out[SVLe×SVLe] = Σ_{p<k} a[p·SVLe:...] ⊗ b[p·SVLe:...] where
// SVLe = SVL()/4 f32 lanes; a and b hold k stacked vectors.
func OuterK(a, b []float32, k int, out []float32) { outerK(&a[0], &b[0], int64(k), &out[0]) }

// ProbeN runs iters iterations of `variant` FMOPAs (1, 2, or 16 per loop).
func ProbeN(iters int64, src []float32, variant int64) { probeN(iters, &src[0], variant) }
