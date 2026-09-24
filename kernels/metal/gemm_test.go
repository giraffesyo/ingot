package metal

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// TestGemmNT checks the GPU GEMM against a float64 oracle, including ragged
// M/N/K (partial tiles on every edge).
func TestGemmNT(t *testing.T) {
	d := openDev(t)
	r := rand.New(rand.NewPCG(91, 92))
	for _, s := range []struct{ m, n, k int }{{64, 64, 32}, {128, 192, 256}, {100, 70, 50}, {1, 5, 3}, {257, 129, 33}} {
		a, _ := d.NewBuffer(4 * s.m * s.k)
		w, _ := d.NewBuffer(4 * s.n * s.k)
		c, _ := d.NewBuffer(4 * s.m * s.n)
		af, wf := f32s(a.Bytes()), f32s(w.Bytes())
		for i := range af {
			af[i] = r.Float32()*2 - 1
		}
		for i := range wf {
			wf[i] = r.Float32()*2 - 1
		}
		if err := d.GemmNT(s.m, s.n, s.k, a, w, c); err != nil {
			t.Fatal(err)
		}
		cf := f32s(c.Bytes())
		for i := range s.m {
			for j := range s.n {
				var want float64
				for p := range s.k {
					want += float64(af[i*s.k+p]) * float64(wf[j*s.k+p])
				}
				if dd := math.Abs(float64(cf[i*s.n+j]) - want); dd > 1e-5*math.Sqrt(float64(s.k))*(1+math.Abs(want)) {
					t.Fatalf("%v: C[%d,%d] = %g, want %g", s, i, j, cf[i*s.n+j], want)
				}
			}
		}
		a.Release()
		w.Release()
		c.Release()
	}
}

// BenchmarkGemmNT reports GPU GEMM throughput at DiT shapes (4096 target
// tokens × a 4096→4096 projection and the 4096→12288 MLP).
func BenchmarkGemmNT(b *testing.B) {
	d := openDev(b)
	for _, s := range []struct{ m, n, k int }{{4096, 4096, 4096}, {4096, 12288, 4096}} {
		a, _ := d.NewBuffer(4 * s.m * s.k)
		w, _ := d.NewBuffer(4 * s.n * s.k)
		c, _ := d.NewBuffer(4 * s.m * s.n)
		b.Run(fmt.Sprintf("m=%d/n=%d/k=%d", s.m, s.n, s.k), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := d.GemmNT(s.m, s.n, s.k, a, w, c); err != nil {
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
