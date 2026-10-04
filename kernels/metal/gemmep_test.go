//go:build darwin && arm64

package metal

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// bf16Round is round-to-nearest-even to bf16, as the kernels' bfloat cast.
func bf16Round(v float64) float64 {
	u := math.Float32bits(float32(v))
	return float64(math.Float32frombits((u + 0x7fff + (u>>16)&1) >> 16 << 16))
}

// TestGemmEp: the fused-epilogue bf16 GEMM vs a float64 oracle over the
// same bf16 operands — NT and NN weights, strided operands, ragged tiles on
// every edge, with and without bias, identity and GELU epilogues, f32 and
// bf16 destinations (the latter within one bf16 rounding of the oracle).
// The destination starts dirty and its row padding must stay untouched.
func TestGemmEp(t *testing.T) {
	d := prepared(t)
	r := rand.New(rand.NewPCG(31, 32))
	geluTanh := func(v float64) float64 {
		return 0.5 * v * (1 + math.Tanh(0.7978845608*(v+0.044715*v*v*v)))
	}
	for _, s := range []struct{ m, n, k int }{{64, 64, 32}, {128, 192, 256}, {100, 70, 50}, {1, 5, 3}, {257, 129, 33}} {
		for _, transB := range []bool{true, false} {
			for _, c16 := range []bool{false, true} {
				for _, act := range []int{ActNone, ActGeluTanh} {
					for _, withBias := range []bool{true, false} {
						lda, ldc := s.k+3, s.n+5
						bRows, bCols := s.k, s.n
						if transB {
							bRows, bCols = s.n, s.k
						}
						ldb := bCols + 2
						a, av := bf16buf(t, d, r, s.m*lda)
						w, wv := bf16buf(t, d, r, bRows*ldb)
						bias := buf(t, d, s.n)
						bf := fill(r, bias)
						c := buf(t, d, s.m*ldc)
						const dirty = 1e9
						for i := range f32s(c.Bytes()) {
							f32s(c.Bytes())[i] = dirty
						}
						g := GemmEp{M: s.m, N: s.n, K: s.k, A: a.At(0), B: w.At(0), C: c.At(0), LDA: lda, LDB: ldb, LDC: ldc,
							TransB: transB, CBF16: c16, Act: act}
						if withBias {
							g.Bias = bias.At(0)
						}
						if err := d.Run(func(e *Encoder) { e.GemmEp(g) }); err != nil {
							t.Fatal(err)
						}
						name := fmt.Sprintf("%v transB=%v c16=%v act=%d bias=%v", s, transB, c16, act, withBias)
						cb := c.Bytes()
						for i := range s.m {
							for j := range s.n {
								var want float64
								for p := range s.k {
									if transB {
										want += av[i*lda+p] * wv[j*ldb+p]
									} else {
										want += av[i*lda+p] * wv[p*ldb+j]
									}
								}
								if withBias {
									want += float64(bf[j])
								}
								if act == ActGeluTanh {
									want = geluTanh(want)
								}
								tol := 1e-5 * math.Sqrt(float64(s.k))
								var got float32
								if c16 {
									o := 2 * (i*ldc + j)
									got = math.Float32frombits(uint32(cb[o])<<16 | uint32(cb[o+1])<<24)
									tol += 1.0 / 256 // one bf16 rounding
								} else {
									got = f32s(cb)[i*ldc+j]
								}
								if dd := math.Abs(float64(got) - want); dd > tol*(1+math.Abs(want)) || math.IsNaN(float64(got)) {
									t.Fatalf("%s: C[%d,%d] = %g, want %g", name, i, j, got, want)
								}
								if c16 && act == ActNone && float64(got) != bf16Round(float64(got)) {
									t.Fatalf("%s: C[%d,%d] = %g is not a bf16 value", name, i, j, got)
								}
							}
						}
						if !c16 { // row padding untouched
							for i := range s.m - 1 {
								for j := s.n; j < ldc; j++ {
									if f32s(cb)[i*ldc+j] != dirty {
										t.Fatalf("%s: wrote padding at [%d,%d]", name, i, j)
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

// BenchmarkGemmEp: a transformer MLP's two products at 4096 tokens
// (1536 → 8192 with bias and GELU into a bf16 operand, 8192 → 1536 with
// bias), fused epilogue vs the plain bf16 GEMM followed by separate bias,
// activation and cast passes.
func BenchmarkGemmEp(b *testing.B) {
	d := openDev(b)
	for _, p := range []func() error{d.Prepare, d.PrepareEW, d.PrepareCNN} {
		if err := p(); err != nil {
			b.Fatal(err)
		}
	}
	nb := func(n int) *Buffer {
		x, err := d.NewBuffer(n)
		if err != nil {
			b.Fatal(err)
		}
		b.Cleanup(x.Release)
		return x
	}
	for _, s := range []struct {
		m, n, k int
		gelu    bool
	}{{4096, 8192, 1536, true}, {4096, 1536, 8192, false}} {
		a, w, bias := nb(2*s.m*s.k), nb(2*s.n*s.k), nb(4*s.n)
		c, c2 := nb(4*s.m*s.n), nb(4*s.m*s.n)
		act := ActNone
		if s.gelu {
			act = ActGeluTanh
		}
		flops := 2 * float64(s.m) * float64(s.n) * float64(s.k)
		shape := fmt.Sprintf("shape=%dx%dx%d", s.m, s.n, s.k)
		run := func(name string, enc func(e *Encoder)) {
			b.Run(name+"/"+shape, func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					if err := d.Run(enc); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOPS")
			})
		}
		run("fused", func(e *Encoder) {
			e.GemmEp(GemmEp{M: s.m, N: s.n, K: s.k, A: a.At(0), B: w.At(0), C: c.At(0), TransB: true, Bias: bias.At(0), Act: act, CBF16: s.gelu})
		})
		run("separate", func(e *Encoder) {
			e.Gemm(Gemm{M: s.m, N: s.n, K: s.k, A: a.At(0), B: w.At(0), C: c.At(0), TransB: true, BF16: true, ABF16: true})
			e.BiasAct(c.At(0), bias.At(0), s.m*s.n, 1, s.n, ConvEpilogue{})
			if s.gelu {
				e.Unary(UnGeluTanh, c.At(0), c2.At(0), s.m*s.n)
				e.CastBF16(c2.At(0), c.At(0), s.m, s.n, s.n, s.n)
			}
		})
	}
}
