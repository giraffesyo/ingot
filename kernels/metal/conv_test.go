//go:build darwin && arm64

package metal

import (
	"math"
	"math/rand/v2"
	"testing"
	"unsafe"
)

// TestConvKernels: 3x3 conv as im2col (in two pixel chunks) + GEMM + bias,
// SiLU, 2x upsample and the channel-map depth-to-space, each vs a direct
// float64 computation over NHWC.
func TestConvKernels(t *testing.T) {
	d := openDev(t)
	if err := d.PrepareConv(); err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(9, 9))
	const H, W, C, CO = 5, 7, 6, 4
	x, w, bias := buf(t, d, H*W*C), buf(t, d, CO*C*9), buf(t, d, CO)
	cols, y := buf(t, d, H*W*C*9), buf(t, d, H*W*CO)
	xf, wf, bf := fill(r, x), fill(r, w), fill(r, bias)
	x0 := append([]float32(nil), xf...) // x is overwritten by the SiLU at the end
	up := buf(t, d, 4*H*W*C)
	idx := buf(t, d, CO*4)
	iv := unsafe.Slice((*uint32)(unsafe.Pointer(&idx.Bytes()[0])), CO*4)
	for i := range iv {
		iv[i] = uint32((i * 5) % C)
	}
	ds := buf(t, d, 4*H*W*CO)
	split := 17
	if err := d.Run(func(e *Encoder) {
		e.Im2Col3x3(x.At(0), cols.At(0), H, W, C, 0, split)
		e.Im2Col3x3(x.At(0), cols.At(4*split*C*9), H, W, C, split, H*W-split)
		e.Gemm(Gemm{M: H * W, N: CO, K: C * 9, A: cols.At(0), B: w.At(0), C: y.At(0), TransB: true})
		e.AddBias(y.At(0), bias.At(0), H*W, CO, CO)
		e.Upsample2x(x.At(0), up.At(0), H, W, C)
		e.DepthToSpace2Map(x.At(0), ds.At(0), idx.At(0), H, W, C, CO)
		e.SiLU(x.At(0), H*W*C) // last: the checks above read x before this
	}); err != nil {
		t.Fatal(err)
	}
	orig := func(h, ww, c int) float64 { return float64(x0[(h*W+ww)*C+c]) }
	exp := math.Exp
	yf := f32s(y.Bytes())
	for h := range H {
		for ww := range W {
			for co := range CO {
				want := float64(bf[co])
				for ci := range C {
					for kh := range 3 {
						for kw := range 3 {
							hh, w2 := h+kh-1, ww+kw-1
							if hh < 0 || w2 < 0 || hh >= H || w2 >= W {
								continue
							}
							want += orig(hh, w2, ci) * float64(wf[co*C*9+ci*9+kh*3+kw])
						}
					}
				}
				near(t, "conv", yf[(h*W+ww)*CO+co], want, 1e-5)
				for hs := range 2 {
					for ws := range 2 {
						got := f32s(ds.Bytes())[((2*h+hs)*2*W+2*ww+ws)*CO+co]
						if float64(got) != orig(h, ww, int(iv[(co*2+hs)*2+ws])) {
							t.Fatalf("d2s at %d,%d,%d", h, ww, co)
						}
					}
				}
			}
			for c := range C {
				for dy := range 2 {
					for dx := range 2 {
						if f32s(up.Bytes())[((2*h+dy)*2*W+2*ww+dx)*C+c] != float32(orig(h, ww, c)) {
							t.Fatal("upsample")
						}
					}
				}
				v := orig(h, ww, c)
				near(t, "silu", xf[(h*W+ww)*C+c], v/(1+exp(-v)), 1e-6)
			}
		}
	}
}
