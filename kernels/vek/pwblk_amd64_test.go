//go:build amd64

package vek

import (
	"testing"

	"golang.org/x/sys/cpu"
)

// TestPwBlk6x16Z checks the ZMM tile against the scalar reference directly
// (the wrapper may route to either kernel depending on the probe).
func TestPwBlk6x16Z(t *testing.T) {
	if !cpu.X86.HasAVX512F {
		t.Skip("no AVX-512F")
	}
	for _, cin := range []int{8, 16, 24, 96, 576} {
		nb := cin / 8
		const pos = 6
		x := make([]float32, nb*pos*8)
		w := make([]float32, cin*16)
		for i := range x {
			x[i] = float32(i%19)*0.05 - 0.4
		}
		for i := range w {
			w[i] = float32(i%23)*0.04 - 0.5
		}
		g0 := make([]float32, pos*8)
		g1 := make([]float32, pos*8)
		w0 := make([]float32, pos*8)
		w1 := make([]float32, pos*8)
		pwblk6x16z_asm(g0, g1, x, w, cin, pos*8*4)
		pwBlkRef(w0, w1, x, w, cin, pos*8*4)
		for i := range w0 {
			if d := g0[i] - w0[i]; d > 1e-3 || d < -1e-3 {
				t.Fatalf("cin=%d dst0[%d]: got %g want %g", cin, i, g0[i], w0[i])
			}
			if d := g1[i] - w1[i]; d > 1e-3 || d < -1e-3 {
				t.Fatalf("cin=%d dst1[%d]: got %g want %g", cin, i, g1[i], w1[i])
			}
		}
	}
}

// TestPwBlk6x16TilesAMD64 checks both looped kernels directly.
func TestPwBlk6x16TilesAMD64(t *testing.T) {
	kernels := []struct {
		name string
		f    func(d0, d1, x, w []float32, cin, xbstride, tiles int)
		ok   bool
	}{
		{"avx2", pwblk6x16t_asm, true},
		{"avx512", pwblk6x16zt_asm, cpu.X86.HasAVX512F},
	}
	for _, k := range kernels {
		if !k.ok {
			continue
		}
		for _, cin := range []int{8, 24, 96} {
			for _, tiles := range []int{1, 2, 5} {
				nb := cin / 8
				P := tiles * 6
				x := make([]float32, nb*P*8)
				w := make([]float32, cin*16)
				for i := range x {
					x[i] = float32(i%19)*0.05 - 0.4
				}
				for i := range w {
					w[i] = float32(i%23)*0.04 - 0.5
				}
				g0 := make([]float32, P*8)
				g1 := make([]float32, P*8)
				w0 := make([]float32, P*8)
				w1 := make([]float32, P*8)
				k.f(g0, g1, x, w, cin, P*8*4, tiles)
				for ti := 0; ti < tiles; ti++ {
					pwBlkRef(w0[ti*48:], w1[ti*48:], x[ti*48:], w, cin, P*8*4)
				}
				for i := range w0 {
					if d := g0[i] - w0[i]; d > 1e-3 || d < -1e-3 {
						t.Fatalf("%s cin=%d tiles=%d dst0[%d]: got %g want %g", k.name, cin, tiles, i, g0[i], w0[i])
					}
					if d := g1[i] - w1[i]; d > 1e-3 || d < -1e-3 {
						t.Fatalf("%s cin=%d tiles=%d dst1[%d]: got %g want %g", k.name, cin, tiles, i, g1[i], w1[i])
					}
				}
			}
		}
	}
}
