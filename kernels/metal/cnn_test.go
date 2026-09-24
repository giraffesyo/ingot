//go:build darwin && arm64

package metal

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
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
			yd, yg := buf(t, d, g.N*g.M*P), buf(t, d, g.N*g.M*P)
			pc := (P + 2) / 3 // three pixel chunks
			cols := buf(t, d, K*pc)
			err := d.Run(func(e *Encoder) {
				e.ConvDirect(x.At(0), w.At(0), b.At(0), yd.At(0), g)
				if g.Group != 1 {
					return
				}
				for n := range g.N {
					xn, yn := x.At(4*n*g.C*g.H*g.W), n*g.M*P
					for p0 := 0; p0 < P; p0 += pc {
						c := min(pc, P-p0)
						e.Im2ColNCHW(xn, cols.At(0), g, p0, c)
						e.Gemm(Gemm{M: g.M, N: c, K: K, A: w.At(0), B: cols.At(0), LDB: c, C: yg.At(4 * (yn + p0)), LDC: P})
					}
				}
				n := g.N * g.M * P
				e.BinaryBcast(OpAdd, yg.At(0), b.At(0), yg.At(0), n, 1, n, P, g.M)
			})
			if err != nil {
				t.Fatal(err)
			}
			tol := 1e-5 * math.Sqrt(float64(K)) * 4
			for name, y := range map[string]*Buffer{"direct": yd, "im2col": yg} {
				if name == "im2col" && g.Group != 1 {
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
