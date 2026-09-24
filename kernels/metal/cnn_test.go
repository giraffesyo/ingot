//go:build darwin && arm64

package metal

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
	"unsafe"
)

// convRef is the float64 oracle for NCHW grouped convolution.
func convRef(x, w, bias []float32, g ConvGeom) []float64 {
	Cg, Mg := g.C/g.Group, g.M/g.Group
	out := make([]float64, g.N*g.M*g.OH*g.OW)
	for n := range g.N {
		for m := range g.M {
			gi := m / Mg
			for oy := range g.OH {
				for ox := range g.OW {
					var acc float64
					if bias != nil {
						acc = float64(bias[m])
					}
					for c := range Cg {
						for ky := range g.KH {
							for kx := range g.KW {
								iy, ix := oy*g.SH-g.PT+ky*g.DH, ox*g.SW-g.PL+kx*g.DW
								if iy < 0 || iy >= g.H || ix < 0 || ix >= g.W {
									continue
								}
								acc += float64(x[((n*g.C+gi*Cg+c)*g.H+iy)*g.W+ix]) * float64(w[((m*Cg+c)*g.KH+ky)*g.KW+kx])
							}
						}
					}
					out[((n*g.M+m)*g.OH+oy)*g.OW+ox] = acc
				}
			}
		}
	}
	return out
}

// TestConv: im2col + GEMM (group 1, chunked over pixels) and the direct
// kernel (any group) vs the oracle over strides, dilations, pads, groups.
func TestConv(t *testing.T) {
	d := openDev(t)
	for _, p := range []func() error{d.Prepare, d.PrepareEW, d.PrepareCNN} {
		if err := p(); err != nil {
			t.Fatal(err)
		}
	}
	r := rand.New(rand.NewPCG(5, 5))
	for _, g := range []ConvGeom{
		{N: 1, C: 3, H: 17, W: 13, M: 8, KH: 3, KW: 3, SH: 1, SW: 1, DH: 1, DW: 1, PT: 1, PL: 1, Group: 1},
		{N: 2, C: 5, H: 16, W: 16, M: 12, KH: 3, KW: 5, SH: 2, SW: 1, DH: 1, DW: 2, PT: 0, PL: 3, Group: 1},
		{N: 1, C: 16, H: 9, W: 11, M: 24, KH: 1, KW: 1, SH: 1, SW: 1, DH: 1, DW: 1, Group: 1},
		{N: 2, C: 8, H: 12, W: 10, M: 8, KH: 3, KW: 3, SH: 2, SW: 2, DH: 1, DW: 1, PT: 1, PL: 1, Group: 8},
		{N: 1, C: 12, H: 7, W: 9, M: 6, KH: 5, KW: 3, SH: 1, SW: 2, DH: 2, DW: 1, PT: 2, PL: 0, Group: 3},
		{N: 2, C: 6, H: 13, W: 17, M: 6, KH: 5, KW: 5, SH: 2, SW: 1, DH: 1, DW: 1, PT: 2, PL: 2, Group: 6},
		{N: 1, C: 5, H: 11, W: 23, M: 5, KH: 3, KW: 7, SH: 1, SW: 2, DH: 2, DW: 2, PT: 1, PL: 3, Group: 5},
	} {
		t.Run(fmt.Sprintf("%dx%dx%dx%d_k%dx%d_s%d%d_d%d%d_g%d", g.N, g.C, g.H, g.W, g.KH, g.KW, g.SH, g.SW, g.DH, g.DW, g.Group), func(t *testing.T) {
			// Symmetric pads in these cases: bottom/right = top/left.
			g.OH = (g.H+2*g.PT-(g.DH*(g.KH-1)+1))/g.SH + 1
			g.OW = (g.W+2*g.PL-(g.DW*(g.KW-1)+1))/g.SW + 1
			Cg := g.C / g.Group
			K, P := Cg*g.KH*g.KW, g.OH*g.OW
			x, w, b := buf(t, d, g.N*g.C*g.H*g.W), buf(t, d, g.M*K), buf(t, d, g.M)
			xf, wf, bf := fill(r, x), fill(r, w), fill(r, b)
			want := convRef(xf, wf, bf, g)
			yd, yg, yb := buf(t, d, g.N*g.M*P), buf(t, d, g.N*g.M*P), buf(t, d, g.N*g.M*P)
			// bf16 path: bf16 weights and columns, bf16×bf16 GEMM.
			yh, wh, colsH := buf(t, d, g.N*g.M*P), buf(t, d, (g.M*K+1)/2), buf(t, d, (K*P+1)/2)
			pc := (P + 2) / 3 // three pixel chunks
			cols := buf(t, d, K*pc)
			err := d.Run(func(e *Encoder) {
				e.ConvDirect(x.At(0), w.At(0), b.At(0), yd.At(0), g)
				if DepthwiseOK(g) {
					e.ConvDepthwise(x.At(0), w.At(0), b.At(0), yb.At(0), g)
				}
				if g.Group != 1 {
					return
				}
				e.ConvDirectBlocked(x.At(0), w.At(0), b.At(0), yb.At(0), g)
				for n := range g.N {
					xn, yn := x.At(4*n*g.C*g.H*g.W), n*g.M*P
					for p0 := 0; p0 < P; p0 += pc {
						c := min(pc, P-p0)
						e.Im2ColNCHW(xn, cols.At(0), g, p0, c)
						e.Gemm(Gemm{M: g.M, N: c, K: K, A: w.At(0), B: cols.At(0), LDB: c, C: yg.At(4 * (yn + p0)), LDC: P})
					}
				}
				e.CastBF16(w.At(0), wh.At(0), g.M, K, K, K)
				for n := range g.N {
					e.Im2ColNCHWBF16(x.At(4*n*g.C*g.H*g.W), colsH.At(0), g, 0, P)
					e.Gemm(Gemm{M: g.M, N: P, K: K, A: wh.At(0), B: colsH.At(0), C: yh.At(4 * n * g.M * P), ABF16: true, BF16: true})
				}
				n := g.N * g.M * P
				e.BinaryBcast(OpAdd, yg.At(0), b.At(0), yg.At(0), n, 1, n, P, g.M)
				e.BinaryBcast(OpAdd, yh.At(0), b.At(0), yh.At(0), n, 1, n, P, g.M)
			})
			if err != nil {
				t.Fatal(err)
			}
			tol := 1e-5 * math.Sqrt(float64(K)) * 4
			if g.Group == 1 {
				ax, aw := make([]float32, len(xf)), make([]float32, len(wf))
				for i, v := range xf {
					ax[i] = float32(math.Abs(float64(v)))
				}
				for i, v := range wf {
					aw[i] = float32(math.Abs(float64(v)))
				}
				bound := convRef(ax, aw, nil, g)
				for i, v := range f32s(yh.Bytes()) {
					if math.Abs(float64(v)-want[i]) > bound[i]/128+1e-6 {
						t.Fatalf("bf16 [%d] = %g, want %g (bound %g)", i, v, want[i], bound[i]/128)
					}
				}
			}
			for name, y := range map[string]*Buffer{"direct": yd, "im2col": yg, "blocked": yb} {
				if name == "im2col" && g.Group != 1 || name == "blocked" && g.Group != 1 && !DepthwiseOK(g) {
					continue
				}
				for i, v := range f32s(y.Bytes()) {
					if math.Abs(float64(v)-want[i]) > tol*(1+math.Abs(want[i])) {
						t.Fatalf("%s [%d] = %g, want %g", name, i, v, want[i])
					}
				}
			}
		})
	}
}

// TestPoolAct: max/avg pooling (pads, include-pad) and the activation
// epilogue vs oracles.
func TestPoolAct(t *testing.T) {
	d := openDev(t)
	if err := d.PrepareCNN(); err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(9, 9))
	const planes, H, W, KH, KW, S = 3, 11, 10, 3, 3, 2
	pads := [4]int{1, 1, 1, 1}
	OH, OW := (H+2-KH)/S+1, (W+2-KW)/S+1
	x := buf(t, d, planes*H*W)
	xf := fill(r, x)
	for _, isMax := range []bool{true, false} {
		for _, inc := range []bool{false, true} {
			o := buf(t, d, planes*OH*OW)
			if err := d.Run(func(e *Encoder) {
				e.Pool2D(x.At(0), o.At(0), planes, H, W, OH, OW, KH, KW, S, S, pads, isMax, inc)
			}); err != nil {
				t.Fatal(err)
			}
			for pl := range planes {
				for oy := range OH {
					for ox := range OW {
						m, s, cnt := math.Inf(-1), 0.0, 0
						for ky := range KH {
							for kx := range KW {
								iy, ix := oy*S-1+ky, ox*S-1+kx
								if iy < 0 || iy >= H || ix < 0 || ix >= W {
									continue
								}
								v := float64(xf[(pl*H+iy)*W+ix])
								m, s, cnt = math.Max(m, v), s+v, cnt+1
							}
						}
						if inc {
							hA, hB := max(oy*S-1, -1), min(oy*S-1+KH, H+1)
							wA, wB := max(ox*S-1, -1), min(ox*S-1+KW, W+1)
							cnt = (hB - hA) * (wB - wA)
						}
						want := m
						if !isMax {
							want = s / float64(cnt)
						}
						if got := float64(f32s(o.Bytes())[(pl*OH+oy)*OW+ox]); math.Abs(got-want) > 1e-5 {
							t.Fatalf("pool max=%v inc=%v [%d,%d,%d] = %g, want %g", isMax, inc, pl, oy, ox, got, want)
						}
					}
				}
			}
		}
	}
	const n = 300
	u, uo := buf(t, d, n), buf(t, d, n)
	uf := fill(r, u)
	for i := range uf { // spread over the hard* knees
		uf[i] *= 8
	}
	const alpha, beta, scale, shift = 0.2, 0.5, 1.5, -0.25
	for act, fn := range []func(v float64) float64{
		func(v float64) float64 { return v },
		func(v float64) float64 { return math.Max(v, 0) },
		func(v float64) float64 { return v * math.Min(math.Max(v/6+0.5, 0), 1) },
		func(v float64) float64 { return math.Min(math.Max(alpha*v+beta, 0), 1) },
		func(v float64) float64 { return 1 / (1 + math.Exp(-v)) },
		func(v float64) float64 { return v / (1 + math.Exp(-v)) },
		func(v float64) float64 { return math.Min(math.Max(v, alpha), beta) },
		func(v float64) float64 {
			if v >= 0 {
				return v
			}
			return alpha * v
		},
	} {
		if err := d.Run(func(e *Encoder) { e.Act(act, u.At(0), uo.At(0), n, alpha, beta, scale, shift) }); err != nil {
			t.Fatal(err)
		}
		for i := range n {
			want := fn(float64(uf[i]))*scale + shift
			if got := float64(f32s(uo.Bytes())[i]); math.Abs(got-want) > 2e-6*(1+math.Abs(want)) {
				t.Fatalf("act %d [%d] (x=%g) = %g, want %g", act, i, uf[i], got, want)
			}
		}
	}
}

// TestConvTranspose: the gather-form kernel vs a scatter-add oracle over
// strides, pads, dilation, groups and output padding.
func TestConvTranspose(t *testing.T) {
	d := openDev(t)
	if err := d.PrepareCNN(); err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(7, 7))
	for _, c := range []struct {
		g      ConvGeom
		outPad int
	}{
		{ConvGeom{N: 1, C: 4, H: 5, W: 6, M: 3, KH: 2, KW: 2, SH: 2, SW: 2, DH: 1, DW: 1, Group: 1}, 0},
		{ConvGeom{N: 2, C: 6, H: 4, W: 5, M: 4, KH: 3, KW: 3, SH: 2, SW: 2, DH: 1, DW: 1, PT: 1, PL: 1, Group: 2}, 1},
		{ConvGeom{N: 1, C: 3, H: 6, W: 4, M: 5, KH: 3, KW: 2, SH: 1, SW: 3, DH: 2, DW: 1, PT: 2, PL: 0, Group: 1}, 0},
	} {
		g := c.g
		// Symmetric pads: bottom/right = top/left.
		g.OH = (g.H-1)*g.SH - 2*g.PT + g.DH*(g.KH-1) + c.outPad + 1
		g.OW = (g.W-1)*g.SW - 2*g.PL + g.DW*(g.KW-1) + c.outPad + 1
		CinG, CoutG := g.C/g.Group, g.M/g.Group
		x, w, b := buf(t, d, g.N*g.C*g.H*g.W), buf(t, d, g.C*CoutG*g.KH*g.KW), buf(t, d, g.M)
		xf, wf, bf := fill(r, x), fill(r, w), fill(r, b)
		o, ob := buf(t, d, g.N*g.M*g.OH*g.OW), buf(t, d, g.N*g.M*g.OH*g.OW)
		if err := d.Run(func(e *Encoder) {
			e.ConvTransposeDirect(x.At(0), w.At(0), b.At(0), o.At(0), g)
			e.ConvTransposeBlocked(x.At(0), w.At(0), b.At(0), ob.At(0), g)
		}); err != nil {
			t.Fatal(err)
		}
		want := make([]float64, g.N*g.M*g.OH*g.OW)
		for n := range g.N {
			for oc := range g.M {
				for p := range g.OH * g.OW {
					want[(n*g.M+oc)*g.OH*g.OW+p] = float64(bf[oc])
				}
			}
			for ci := range g.C {
				gi := ci / CinG
				for ocg := range CoutG {
					oc := gi*CoutG + ocg
					for iy := range g.H {
						for ix := range g.W {
							for ky := range g.KH {
								for kx := range g.KW {
									oy, ox := iy*g.SH-g.PT+ky*g.DH, ix*g.SW-g.PL+kx*g.DW
									if oy < 0 || oy >= g.OH || ox < 0 || ox >= g.OW {
										continue
									}
									want[((n*g.M+oc)*g.OH+oy)*g.OW+ox] += float64(xf[((n*g.C+ci)*g.H+iy)*g.W+ix]) *
										float64(wf[((ci*CoutG+ocg)*g.KH+ky)*g.KW+kx])
								}
							}
						}
					}
				}
			}
		}
		for _, y := range []*Buffer{o, ob} {
			for i, v := range f32s(y.Bytes()) {
				if math.Abs(float64(v)-want[i]) > 1e-5*(1+math.Abs(want[i])) {
					t.Fatalf("%+v [%d] = %g, want %g", g, i, v, want[i])
				}
			}
		}
	}
}

// TestResizeTaps: bilinear taps (fractional weights both ways) and a
// nearest map vs direct evaluation.
func TestResizeTaps(t *testing.T) {
	d := openDev(t)
	if err := d.PrepareCNN(); err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(3, 3))
	const planes, H, W, OH, OW = 2, 5, 7, 8, 3
	x, o := buf(t, d, planes*H*W), buf(t, d, planes*OH*OW)
	xf := fill(r, x)
	ti, tw := buf(t, d, 2*OH+2*OW), buf(t, d, OH+OW)
	idx := unsafe.Slice((*int32)(unsafe.Pointer(&ti.Bytes()[0])), 2*OH+2*OW)
	wts := f32s(tw.Bytes())
	for i := range OH {
		idx[i], idx[OH+i] = int32(i*H/OH), int32(min(i*H/OH+1, H-1))
		wts[i] = float32(i%3) / 3
	}
	for j := range OW {
		idx[2*OH+j], idx[2*OH+OW+j] = int32(j*2), int32(j*2+1)
		wts[OH+j] = float32(j) / 4
	}
	if err := d.Run(func(e *Encoder) { e.ResizeTaps(x.At(0), o.At(0), ti.At(0), tw.At(0), planes, H, W, OH, OW) }); err != nil {
		t.Fatal(err)
	}
	for p := range planes {
		for i := range OH {
			for j := range OW {
				at := func(y, xx int32) float32 { return xf[(p*H+int(y))*W+int(xx)] }
				wy, wx := wts[i], wts[OH+j]
				top := at(idx[i], idx[2*OH+j])*(1-wx) + at(idx[i], idx[2*OH+OW+j])*wx
				bot := at(idx[OH+i], idx[2*OH+j])*(1-wx) + at(idx[OH+i], idx[2*OH+OW+j])*wx
				want := top*(1-wy) + bot*wy
				if got := f32s(o.Bytes())[(p*OH+i)*OW+j]; math.Abs(float64(got-want)) > 1e-6 {
					t.Fatalf("[%d %d %d] = %g, want %g", p, i, j, got, want)
				}
			}
		}
	}
}

// BenchmarkConvThin: im2col + GEMM vs the register-blocked direct kernel
// on the OCR detector's thin high-resolution convs.
func BenchmarkConvThin(b *testing.B) {
	d := openDev(b)
	for _, p := range []func() error{d.Prepare, d.PrepareEW, d.PrepareCNN} {
		if err := p(); err != nil {
			b.Fatal(err)
		}
	}
	for _, g := range []ConvGeom{
		{N: 1, C: 96, H: 240, W: 240, M: 24, KH: 3, KW: 3, SH: 1, SW: 1, DH: 1, DW: 1, PT: 1, PL: 1, Group: 1, OH: 240, OW: 240},
		{N: 1, C: 16, H: 480, W: 480, M: 32, KH: 1, KW: 1, SH: 1, SW: 1, DH: 1, DW: 1, Group: 1, OH: 480, OW: 480},
		{N: 1, C: 3, H: 960, W: 960, M: 16, KH: 3, KW: 3, SH: 2, SW: 2, DH: 1, DW: 1, PT: 1, PL: 1, Group: 1, OH: 480, OW: 480},
		{N: 1, C: 192, H: 60, W: 60, M: 192, KH: 1, KW: 1, SH: 1, SW: 1, DH: 1, DW: 1, Group: 1, OH: 60, OW: 60},
	} {
		K, P := g.C*g.KH*g.KW, g.OH*g.OW
		x, w, y := bufB(b, d, g.N*g.C*g.H*g.W), bufB(b, d, g.M*K), bufB(b, d, g.M*P)
		pc := min(P, max(256, (8<<20)/K))
		cols := bufB(b, d, K*pc)
		name := fmt.Sprintf("c%d_m%d_%dx%d_k%d", g.C, g.M, g.H, g.W, g.KH)
		b.Run(name+"/im2col", func(b *testing.B) {
			for b.Loop() {
				d.Run(func(e *Encoder) {
					for p0 := 0; p0 < P; p0 += pc {
						c := min(pc, P-p0)
						if g.KH == 1 && g.SH == 1 {
							e.Gemm(Gemm{M: g.M, N: P, K: K, A: w.At(0), B: x.At(0), C: y.At(0)})
							break
						}
						e.Im2ColNCHW(x.At(0), cols.At(0), g, p0, c)
						e.Gemm(Gemm{M: g.M, N: c, K: K, A: w.At(0), B: cols.At(0), LDB: c, C: y.At(4 * p0), LDC: P})
					}
				})
			}
			b.ReportMetric(2*float64(g.M*K*P)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
		for v, cb := range blockWidths {
			b.Run(fmt.Sprintf("%s/blocked%d", name, cb), func(b *testing.B) {
				for b.Loop() {
					d.Run(func(e *Encoder) { e.convDirectCB(v, x.At(0), w.At(0), Region{}, y.At(0), g) })
				}
				b.ReportMetric(2*float64(g.M*K*P)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			})
		}
	}
}

// BenchmarkConvTranspose: the per-output gather kernel vs the blocked one
// on the OCR detector head's upsampling convs.
func BenchmarkConvTranspose(b *testing.B) {
	d := openDev(b)
	if err := d.PrepareCNN(); err != nil {
		b.Fatal(err)
	}
	for _, g := range []ConvGeom{
		{N: 1, C: 24, H: 240, W: 240, M: 24, KH: 2, KW: 2, SH: 2, SW: 2, DH: 1, DW: 1, Group: 1, OH: 480, OW: 480},
		{N: 1, C: 24, H: 480, W: 480, M: 1, KH: 2, KW: 2, SH: 2, SW: 2, DH: 1, DW: 1, Group: 1, OH: 960, OW: 960},
	} {
		x, w, y := bufB(b, d, g.N*g.C*g.H*g.W), bufB(b, d, g.C*g.M*g.KH*g.KW), bufB(b, d, g.M*g.OH*g.OW)
		name := fmt.Sprintf("c%d_m%d_%dx%d", g.C, g.M, g.H, g.W)
		b.Run(name+"/direct", func(b *testing.B) {
			for b.Loop() {
				d.Run(func(e *Encoder) { e.ConvTransposeDirect(x.At(0), w.At(0), Region{}, y.At(0), g) })
			}
		})
		for v, cb := range blockWidths {
			b.Run(fmt.Sprintf("%s/blocked%d", name, cb), func(b *testing.B) {
				for b.Loop() {
					d.Run(func(e *Encoder) { e.convTCB(v, x.At(0), w.At(0), Region{}, y.At(0), g) })
				}
			})
		}
	}
}

func bufB(b *testing.B, d *Device, floats int) *Buffer {
	buf, err := d.NewBuffer(4 * max(floats, 1))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(buf.Release)
	return buf
}

// BenchmarkConvDepthwise: the direct kernel on depthwise convs (PP-OCR
// recognizer / detector shapes).
func BenchmarkConvDepthwise(b *testing.B) {
	d := openDev(b)
	if err := d.PrepareCNN(); err != nil {
		b.Fatal(err)
	}
	for _, g := range []ConvGeom{
		{N: 8, C: 240, H: 12, W: 134, M: 240, KH: 5, KW: 5, SH: 1, SW: 1, DH: 1, DW: 1, PT: 2, PL: 2, Group: 240, OH: 12, OW: 134},
		{N: 1, C: 192, H: 60, W: 60, M: 192, KH: 5, KW: 5, SH: 1, SW: 1, DH: 1, DW: 1, PT: 2, PL: 2, Group: 192, OH: 60, OW: 60},
	} {
		x, w, y := bufB(b, d, g.N*g.C*g.H*g.W), bufB(b, d, g.M*g.KH*g.KW), bufB(b, d, g.N*g.M*g.OH*g.OW)
		b.Run(fmt.Sprintf("n%d_c%d_%dx%d_k%d", g.N, g.C, g.H, g.W, g.KH), func(b *testing.B) {
			for b.Loop() {
				d.Run(func(e *Encoder) { e.ConvDirect(x.At(0), w.At(0), Region{}, y.At(0), g) })
			}
			b.ReportMetric(float64(g.N*g.M*g.OH*g.OW*g.KH*g.KW*2)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
		b.Run(fmt.Sprintf("n%d_c%d_%dx%d_k%d/dw", g.N, g.C, g.H, g.W, g.KH), func(b *testing.B) {
			for b.Loop() {
				d.Run(func(e *Encoder) { e.ConvDepthwise(x.At(0), w.At(0), Region{}, y.At(0), g) })
			}
			b.ReportMetric(float64(g.N*g.M*g.OH*g.OW*g.KH*g.KW*2)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

// BenchmarkRoundTrip is one empty command buffer: the fixed cost inside
// every Device.Run-based kernel benchmark.
func BenchmarkRoundTrip(b *testing.B) {
	d := openDev(b)
	for b.Loop() {
		d.Run(func(e *Encoder) {})
	}
}

// BenchmarkCopyBandwidth: a plain 2-D copy of 12 MB (the bandwidth
// reference for memory-bound kernels).
func BenchmarkCopyBandwidth(b *testing.B) {
	d := openDev(b)
	if err := d.PrepareEW(); err != nil {
		b.Fatal(err)
	}
	const rows, cols = 1920, 1608
	x, y := bufB(b, d, rows*cols), bufB(b, d, rows*cols)
	for b.Loop() {
		d.Run(func(e *Encoder) { e.Copy2D(x.At(0), y.At(0), rows, cols, cols, cols) })
	}
	b.ReportMetric(2*4*rows*cols*float64(b.N)/b.Elapsed().Seconds()/1e9, "GB/s")
}
