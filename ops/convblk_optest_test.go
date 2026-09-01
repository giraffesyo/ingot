package ops

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// TestConvDwBlkOracle sweeps blocked depthwise shapes (kernel, stride, small
// and large planes, odd sizes) against a float64 NCHW oracle, routing through
// ToBlk8/FromBlk8. Exercises the row-strip task split across strip edges.
func TestConvDwBlkOracle(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	type cse struct{ N, C, H, W, K, S int }
	var cases []cse
	for _, K := range []int{3, 5} {
		for _, S := range []int{1, 2} {
			for _, hw := range [][2]int{{5, 5}, {7, 9}, {14, 14}, {20, 33}, {56, 56}, {3, 40}} {
				if hw[0] < K || hw[1] < K {
					continue
				}
				cases = append(cases, cse{1, 8, hw[0], hw[1], K, S})
				cases = append(cases, cse{2, 32, hw[0], hw[1], K, S})
			}
		}
	}
	for _, c := range cases {
		pad := c.K / 2
		x := tensor.New(tensor.F32, c.N, c.C, c.H, c.W)
		xf := x.F32()
		for i := range xf {
			xf[i] = r.Float32()*2 - 1
		}
		w := tensor.New(tensor.F32, c.C, 1, c.K, c.K)
		wf := w.F32()
		for i := range wf {
			wf[i] = r.Float32()*2 - 1
		}
		bias := tensor.New(tensor.F32, c.C)
		bf := bias.F32()
		for i := range bf {
			bf[i] = r.Float32()
		}
		ctx := &Ctx{}
		to := &toBlk8Op{}
		xb, err := to.Run(ctx, []*tensor.Tensor{x})
		if err != nil {
			t.Fatal(err)
		}
		xbT := xb[0].Clone()
		dw := &convDwBlkOp{k: c.K, s: c.S, pads: [4]int{pad, pad, pad, pad}}
		yb, err := dw.Run(ctx, []*tensor.Tensor{xbT, w, bias})
		if err != nil {
			t.Fatal(err)
		}
		ybT := yb[0].Clone()
		from := &fromBlk8Op{}
		y, err := from.Run(ctx, []*tensor.Tensor{ybT})
		if err != nil {
			t.Fatal(err)
		}
		yf := y[0].F32()
		OH := (c.H+2*pad-c.K)/c.S + 1
		OW := (c.W+2*pad-c.K)/c.S + 1
		for n := 0; n < c.N; n++ {
			for ch := 0; ch < c.C; ch++ {
				for oi := 0; oi < OH; oi++ {
					for oj := 0; oj < OW; oj++ {
						acc := float64(bf[ch])
						for ki := 0; ki < c.K; ki++ {
							for kj := 0; kj < c.K; kj++ {
								ii, jj := oi*c.S+ki-pad, oj*c.S+kj-pad
								if ii < 0 || ii >= c.H || jj < 0 || jj >= c.W {
									continue
								}
								acc += float64(wf[(ch*c.K+ki)*c.K+kj]) * float64(xf[((n*c.C+ch)*c.H+ii)*c.W+jj])
							}
						}
						got := yf[((n*c.C+ch)*OH+oi)*OW+oj]
						if d := math.Abs(float64(got) - acc); d > 1e-5*(1+math.Abs(acc)) {
							t.Fatalf("%+v out[%d,%d,%d,%d]: got %g want %g", c, n, ch, oi, oj, got, acc)
						}
					}
				}
			}
		}
	}
}

// TestConvPwBlkOracle sweeps blocked pointwise shapes against a float64
// oracle, with bias and a NON-idempotent epilogue (post scale/shift): a
// position post-processed twice, or rewritten raw after post-processing
// (the ragged-tile overlap race), fails loudly. P sweeps all tile residues
// and chunk-fold geometries; odd M/8 exercises the discarded upper half.
func TestConvPwBlkOracle(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	Ps := []int{1, 2, 5, 6, 7, 11, 12, 13, 30, 35, 36, 37, 49, 100, 196, 1201, 12544 / 8}
	for _, C := range []int{8, 16} {
		for _, M := range []int{16, 24} {
			for _, P := range Ps {
				x := tensor.New(tensor.F32, 1, C/8, P, 1, 8)
				xf := x.F32()
				for i := range xf {
					xf[i] = r.Float32()*2 - 1
				}
				w := tensor.New(tensor.F32, M, C, 1, 1)
				wf := w.F32()
				for i := range wf {
					wf[i] = r.Float32()*2 - 1
				}
				bias := tensor.New(tensor.F32, M)
				bf := bias.F32()
				for i := range bf {
					bf[i] = r.Float32()
				}
				op := &convPwBlkOp{epi: epilogue{act: "relu", post: true, scale: 1.5, shift: -0.25}}
				got, err := op.Run(&Ctx{}, []*tensor.Tensor{x, w, bias})
				if err != nil {
					t.Fatal(err)
				}
				gf := got[0].Clone().F32()
				for m := 0; m < M; m++ {
					for p := 0; p < P; p++ {
						acc := float64(bf[m])
						for ci := 0; ci < C; ci++ {
							acc += float64(wf[m*C+ci]) * float64(xf[((ci/8)*P+p)*8+ci%8])
						}
						acc = math.Max(acc, 0)*1.5 - 0.25
						g := gf[((m/8)*P+p)*8+m%8]
						if d := math.Abs(float64(g) - acc); d > 1e-5*(1+math.Abs(acc)) {
							t.Fatalf("C=%d M=%d P=%d out[%d,%d]: got %g want %g", C, M, P, m, p, g, acc)
						}
					}
				}
			}
		}
	}
}
