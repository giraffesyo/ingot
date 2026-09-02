package gemm

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// epiRef is the float64 oracle for SgemmPackedBEpi.
func epiRef(m, n, k int, a, b, bias, res []float32, ldres int, act func(float64) float64) []float64 {
	out := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var acc float64
			for p := 0; p < k; p++ {
				acc += float64(a[i*k+p]) * float64(b[p*n+j])
			}
			if bias != nil {
				acc += float64(bias[j])
			}
			if res != nil {
				acc += float64(res[i*ldres+j])
			}
			if act != nil {
				acc = act(acc)
			}
			out[i*n+j] = acc
		}
	}
	return out
}

func TestSgemmPackedBEpi(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 13))
	sq := func(row []float32) {
		for i, v := range row {
			row[i] = v * v // non-idempotent: a double application fails loudly
		}
	}
	sqF := func(v float64) float64 { return v * v }
	for _, c := range []struct{ m, n, k int }{
		{7, 13, 5}, {1, 64, 32}, {6, 16, 384}, {128, 1152, 384}, {145, 1024, 64},
		{300, 384, 1536}, {17, 47, 700}, {2, 1536, 384}, {64, 4096, 64}, {9, 2000, 40},
	} {
		for variant := 0; variant < 4; variant++ {
			withBias, withRes, withAct := variant != 1, variant >= 2, variant%2 == 1
			a := make([]float32, c.m*c.k)
			b := make([]float32, c.k*c.n)
			for i := range a {
				a[i] = rng.Float32() - 0.5
			}
			for i := range b {
				b[i] = rng.Float32() - 0.5
			}
			var bias, res []float32
			if withBias {
				bias = make([]float32, c.n)
				for i := range bias {
					bias[i] = rng.Float32()*2 - 1
				}
			}
			ldres := c.n + 3
			if withRes {
				res = make([]float32, c.m*ldres)
				for i := range res {
					res[i] = rng.Float32()*2 - 1
				}
			}
			e := &Epilogue{Bias: bias, Res: res, LdRes: ldres}
			var actF func(float64) float64
			if withAct {
				e.Act, actF = sq, sqF
			}
			want := epiRef(c.m, c.n, c.k, a, b, bias, res, ldres, actF)
			pb := PackB(false, c.k, c.n, b, c.n)
			ldc := c.n + 5
			got := make([]float32, c.m*ldc)
			for i := range got {
				got[i] = 7 // must be overwritten, never accumulated onto
			}
			SgemmPackedBEpi(c.m, a, c.k, pb, got, ldc, e)
			tol := 1e-5 * math.Sqrt(float64(c.k)) * 4 // post-activation squares the error
			for i := 0; i < c.m; i++ {
				for j := 0; j < c.n; j++ {
					g, w := float64(got[i*ldc+j]), want[i*c.n+j]
					if math.Abs(g-w) > tol*(1+math.Abs(w)) {
						t.Fatalf("m=%d n=%d k=%d bias=%v res=%v act=%v: [%d,%d] = %g, want %g", c.m, c.n, c.k, withBias, withRes, withAct, i, j, g, w)
					}
				}
			}
		}
	}
}

// BenchmarkSgemmPackedBEpi: fused bias+act epilogue vs the same work as a
// plain GEMM followed by separate bias and activation passes, at a
// transformer fc1 shape.
func BenchmarkSgemmPackedBEpi(b *testing.B) {
	for _, c := range []struct{ m, n, k int }{{128, 1152, 384}, {128, 1536, 384}, {1024, 384, 1536}} {
		a := make([]float32, c.m*c.k)
		w := make([]float32, c.k*c.n)
		bias := make([]float32, c.n)
		res := make([]float32, c.m*c.n)
		for i := range a {
			a[i] = float32(i%7) * 0.1
		}
		for i := range w {
			w[i] = float32(i%5) * 0.1
		}
		pb := PackB(false, c.k, c.n, w, c.n)
		out := make([]float32, c.m*c.n)
		act := func(row []float32) {
			for i, v := range row {
				if v < 0 {
					row[i] = 0
				}
			}
		}
		name := fmt.Sprintf("m=%d/n=%d/k=%d", c.m, c.n, c.k)
		b.Run(name+"/plain", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				SgemmPackedB(c.m, 1, a, c.k, pb, 0, out, c.n)
			}
		})
		b.Run(name+"/separate", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				SgemmPackedB(c.m, 1, a, c.k, pb, 0, out, c.n)
				for r := 0; r < c.m; r++ {
					row := out[r*c.n : (r+1)*c.n]
					for j := range row {
						row[j] += bias[j] + res[r*c.n+j]
					}
					act(row)
				}
			}
		})
		for _, v := range []struct {
			name string
			e    *Epilogue
		}{
			{"fused-bias", &Epilogue{Bias: bias}},
			{"fused-bias-res", &Epilogue{Bias: bias, Res: res, LdRes: c.n}},
			{"fused-bias-act", &Epilogue{Bias: bias, Act: act}},
			{"fused-all", &Epilogue{Bias: bias, Res: res, LdRes: c.n, Act: act}},
		} {
			b.Run(name+"/"+v.name, func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					SgemmPackedBEpi(c.m, a, c.k, pb, out, c.n, v.e)
				}
			})
		}
	}
}
