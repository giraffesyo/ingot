package metal

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// TestGemmNT checks the GPU GEMM (f32 and bf16 weights) against a float64
// oracle, including ragged M/N/K (partial tiles on every edge) and a rerun
// over a dirty C (multiply mode must overwrite, not accumulate).
func TestGemmNT(t *testing.T) {
	d := openDev(t)
	r := rand.New(rand.NewPCG(91, 92))
	for _, bf := range []bool{false, true} {
		for _, s := range []struct{ m, n, k int }{{64, 64, 32}, {128, 192, 256}, {100, 70, 50}, {1, 5, 3}, {257, 129, 33}} {
			esz := 4
			if bf {
				esz = 2
			}
			a, _ := d.NewBuffer(4 * s.m * s.k)
			w, _ := d.NewBuffer(esz * s.n * s.k)
			c, _ := d.NewBuffer(4 * s.m * s.n)
			af := f32s(a.Bytes())
			wv := make([]float64, s.n*s.k)
			for i := range af {
				af[i] = r.Float32()*2 - 1
			}
			wb := w.Bytes()
			for i := range wv {
				x := r.Float32()*2 - 1
				if bf {
					bits := uint16(math.Float32bits(x) >> 16)
					wb[2*i], wb[2*i+1] = byte(bits), byte(bits>>8)
					wv[i] = float64(math.Float32frombits(uint32(bits) << 16))
				} else {
					f32s(wb)[i] = x
					wv[i] = float64(x)
				}
			}
			for i := range f32s(c.Bytes()) {
				f32s(c.Bytes())[i] = 1e9 // dirty: must be overwritten
			}
			if err := d.GemmNT(s.m, s.n, s.k, a, w, c, bf); err != nil {
				t.Fatal(err)
			}
			cf := f32s(c.Bytes())
			for i := range s.m {
				for j := range s.n {
					var want float64
					for p := range s.k {
						want += float64(af[i*s.k+p]) * wv[j*s.k+p]
					}
					if dd := math.Abs(float64(cf[i*s.n+j]) - want); dd > 1e-5*math.Sqrt(float64(s.k))*(1+math.Abs(want)) {
						t.Fatalf("bf16=%v %v: C[%d,%d] = %g, want %g", bf, s, i, j, cf[i*s.n+j], want)
					}
				}
			}
			a.Release()
			w.Release()
			c.Release()
		}
	}
}

// BenchmarkGemmNT reports GPU GEMM throughput at DiT shapes (4096 target
// tokens × a 4096→4096 projection and the 4096→12288 MLP), f32 and bf16
// weights.
func BenchmarkGemmNT(b *testing.B) {
	d := openDev(b)
	for _, bf := range []bool{false, true} {
		for _, s := range []struct{ m, n, k int }{{4096, 4096, 4096}, {4096, 12288, 4096}} {
			a, _ := d.NewBuffer(4 * s.m * s.k)
			w, _ := d.NewBuffer(4 * s.n * s.k)
			c, _ := d.NewBuffer(4 * s.m * s.n)
			b.Run(fmt.Sprintf("w=%s/m=%d/n=%d/k=%d", map[bool]string{false: "f32", true: "bf16"}[bf], s.m, s.n, s.k), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					if err := d.GemmNT(s.m, s.n, s.k, a, w, c, bf); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(2*float64(s.m)*float64(s.n)*float64(s.k)*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOPS")
			})
			a.Release()
			w.Release()
			c.Release()
		}
	}
}
