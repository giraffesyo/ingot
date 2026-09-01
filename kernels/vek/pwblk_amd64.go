//go:build amd64

package vek

import (
	"os"
	"time"

	"golang.org/x/sys/cpu"
)

//go:noescape
func pwblk6x16_asm(dst0, dst1, x, w []float32, cin, xbstride int)

//go:noescape
func pwblk6x16z_asm(dst0, dst1, x, w []float32, cin, xbstride int)

//go:noescape
func pwblk6x16t_asm(dst0, dst1, x, w []float32, cin, xbstride, tiles int)

//go:noescape
func pwblk6x16zt_asm(dst0, dst1, x, w []float32, cin, xbstride, tiles int)

// usePwBlkZ selects the ZMM pointwise tile. True-512 hardware wins on
// instruction count (each ci: 1 wload + 6 broadcasts + 6 FMAs vs 2+6+12);
// double-pumped AVX-512 parts are decided by an init-time micro-probe on a
// hot tile — relative timing, load-immune (gemm's µkernel-pick policy).
// OCR_PWBLK=avx2|avx512 pins.
var usePwBlkZ = pickPwBlk()

func pickPwBlk() bool {
	if !cpu.X86.HasAVX512F {
		return false
	}
	switch os.Getenv("OCR_PWBLK") {
	case "avx2":
		return false
	case "avx512":
		return true
	}
	const cin = 96
	x := make([]float32, cin/8*6*8)
	w := make([]float32, cin*16)
	d0 := make([]float32, 6*8)
	d1 := make([]float32, 6*8)
	for i := range x {
		x[i] = float32(i%7) * 0.25
	}
	for i := range w {
		w[i] = float32(i%5) * 0.5
	}
	bench := func(f func()) time.Duration {
		f() // warm (page-in, decode)
		best := time.Duration(1 << 62)
		for r := 0; r < 3; r++ {
			t0 := time.Now()
			for i := 0; i < 64; i++ {
				f()
			}
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		return best
	}
	a2 := bench(func() { pwblk6x16_asm(d0, d1, x, w, cin, 6*8*4) })
	a5 := bench(func() { pwblk6x16z_asm(d0, d1, x, w, cin, 6*8*4) })
	return a5*100 < a2*95 // require a >5% win
}

// PwBlk6x16 computes a 6-position × 16-output-channel tile of a blocked
// (nChw8c) 1x1 convolution. See pwblk.go for the reference semantics.
func PwBlk6x16(dst0, dst1, x, w []float32, cin, xbstride int) {
	if usePwBlkZ {
		pwblk6x16z_asm(dst0, dst1, x, w, cin, xbstride)
	} else {
		pwblk6x16_asm(dst0, dst1, x, w, cin, xbstride)
	}
}

// PwBlk6x16Tiles runs `tiles` consecutive 6-position tiles (dst0/dst1/x
// advance 48 floats per tile); the loop lives in asm to shed the per-tile
// call overhead (13% of mv2's CPU profile).
func PwBlk6x16Tiles(dst0, dst1, x, w []float32, cin, xbstride, tiles int) {
	if usePwBlkZ {
		pwblk6x16zt_asm(dst0, dst1, x, w, cin, xbstride, tiles)
	} else {
		pwblk6x16t_asm(dst0, dst1, x, w, cin, xbstride, tiles)
	}
}
