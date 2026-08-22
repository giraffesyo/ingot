package ops

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// TestConvEpilogue checks the fused epilogue (ingot_act + post scale/shift)
// against convRef followed by a scalar reference of the same epilogue, over
// the conv paths (im2col, pointwise, depthwise s1 3x3/5x5, generic depthwise).
func TestConvEpilogue(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	rnd := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = r.Float32()*2 - 1
		}
		return s
	}
	acts := []struct {
		name        string
		alpha, beta float32
		ref         func(x float32) float32
	}{
		{"relu", 0, 0, func(x float32) float32 { return max(x, 0) }},
		{"hardswish", 0, 0, func(x float32) float32 { return x * min(max(x+3, 0), 6) / 6 }},
		{"hardsigmoid", 0.2, 0.5, func(x float32) float32 { return min(max(0.2*x+0.5, 0), 1) }},
		{"sigmoid", 0, 0, func(x float32) float32 { return float32(1 / (1 + math.Exp(float64(-x)))) }},
		{"clip", -0.25, 0.5, func(x float32) float32 { return min(max(x, -0.25), 0.5) }},
		{"leakyrelu", 0.1, 0, func(x float32) float32 {
			if x < 0 {
				return 0.1 * x
			}
			return x
		}},
	}
	type cfg struct {
		N, C, H, W, M, KH, KW, G, sh, sw, pt, pl int
	}
	cfgs := []cfg{
		{1, 4, 10, 12, 6, 3, 3, 1, 1, 1, 1, 1}, // im2col (tiled)
		{2, 3, 20, 20, 5, 3, 3, 1, 2, 2, 1, 1}, // im2col stride 2
		{1, 4, 6, 6, 4, 1, 1, 1, 1, 1, 0, 0},   // pointwise
		{1, 8, 9, 9, 8, 3, 3, 8, 1, 1, 1, 1},   // depthwise s1 3x3 (NEON path)
		{1, 6, 9, 11, 6, 5, 5, 6, 1, 1, 2, 2},  // depthwise s1 5x5
		{1, 6, 9, 9, 6, 3, 3, 6, 2, 2, 1, 1},   // depthwise s2 (generic)
	}
	for _, c := range cfgs {
		for _, a := range acts {
			x := rnd(c.N * c.C * c.H * c.W)
			w := rnd(c.M * (c.C / c.G) * c.KH * c.KW)
			bias := rnd(c.M)
			want, OH, OW := convRef(x, w, bias, c.N, c.C, c.H, c.W, c.M, c.KH, c.KW, c.G, c.sh, c.sw, 1, 1, c.pt, c.pl)
			const scale, shift = 0.75, -0.125
			for i := range want {
				want[i] = scale*a.ref(want[i]) + shift
			}
			op := mkOp(t, "Conv", 1, Attrs{
				"group":            {Kind: KindInt, I: int64(c.G)},
				"strides":          {Kind: KindInts, Ints: []int64{int64(c.sh), int64(c.sw)}},
				"pads":             {Kind: KindInts, Ints: []int64{int64(c.pt), int64(c.pl), int64(c.pt), int64(c.pl)}},
				"kernel_shape":     {Kind: KindInts, Ints: []int64{int64(c.KH), int64(c.KW)}},
				"ingot_act":        {Kind: KindString, S: a.name},
				"ingot_act_alpha":  {Kind: KindFloat, F: a.alpha},
				"ingot_act_beta":   {Kind: KindFloat, F: a.beta},
				"ingot_post_scale": {Kind: KindFloat, F: scale},
				"ingot_post_shift": {Kind: KindFloat, F: shift},
			}, 3, 1)
			out := run(t, op,
				tensor.FromF32(x, c.N, c.C, c.H, c.W),
				tensor.FromF32(w, c.M, c.C/c.G, c.KH, c.KW),
				tensor.FromF32(bias, c.M))[0]
			eqF32(t, "conv+"+a.name, out, []int{c.N, c.M, OH, OW}, want)
		}
	}
}

// TestConvTransposeEpilogue: ConvTranspose (k2s2 non-overlap and k3s2 overlap)
// with a fused sigmoid + post affine, against a naive scatter reference.
func TestConvTransposeEpilogue(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	rnd := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = r.Float32()*2 - 1
		}
		return s
	}
	for _, c := range []struct{ N, Cin, H, W, Cout, K, s, pad, outPad int }{
		{1, 3, 7, 9, 2, 2, 2, 0, 0},
		{2, 4, 5, 6, 1, 2, 2, 0, 0},
		{1, 3, 6, 6, 2, 3, 2, 1, 1},
		{1, 2, 5, 5, 3, 1, 2, 0, 0}, // KH<stride: uncovered rows must be written
		{1, 2, 5, 5, 3, 2, 2, 0, 1}, // output_padding
	} {
		x := rnd(c.N * c.Cin * c.H * c.W)
		w := rnd(c.Cin * c.Cout * c.K * c.K)
		bias := rnd(c.Cout)
		OH := (c.H-1)*c.s - 2*c.pad + c.K + c.outPad
		OW := (c.W-1)*c.s - 2*c.pad + c.K + c.outPad
		want := make([]float32, c.N*c.Cout*OH*OW)
		for n := 0; n < c.N; n++ {
			for oc := 0; oc < c.Cout; oc++ {
				plane := want[(n*c.Cout+oc)*OH*OW:]
				for i := 0; i < OH*OW; i++ {
					plane[i] = bias[oc]
				}
				for ic := 0; ic < c.Cin; ic++ {
					for ih := 0; ih < c.H; ih++ {
						for iw := 0; iw < c.W; iw++ {
							v := x[((n*c.Cin+ic)*c.H+ih)*c.W+iw]
							for kh := 0; kh < c.K; kh++ {
								for kw := 0; kw < c.K; kw++ {
									oh, ow := ih*c.s-c.pad+kh, iw*c.s-c.pad+kw
									if oh < 0 || oh >= OH || ow < 0 || ow >= OW {
										continue
									}
									plane[oh*OW+ow] += v * w[((ic*c.Cout+oc)*c.K+kh)*c.K+kw]
								}
							}
						}
					}
				}
			}
		}
		for i := range want {
			want[i] = 2*float32(1/(1+math.Exp(float64(-want[i])))) - 1
		}
		op := mkOp(t, "ConvTranspose", 1, Attrs{
			"strides":          {Kind: KindInts, Ints: []int64{int64(c.s), int64(c.s)}},
			"pads":             {Kind: KindInts, Ints: []int64{int64(c.pad), int64(c.pad), int64(c.pad), int64(c.pad)}},
			"output_padding":   {Kind: KindInts, Ints: []int64{int64(c.outPad), int64(c.outPad)}},
			"kernel_shape":     {Kind: KindInts, Ints: []int64{int64(c.K), int64(c.K)}},
			"ingot_act":        {Kind: KindString, S: "sigmoid"},
			"ingot_post_scale": {Kind: KindFloat, F: 2},
			"ingot_post_shift": {Kind: KindFloat, F: -1},
		}, 3, 1)
		out := run(t, op,
			tensor.FromF32(x, c.N, c.Cin, c.H, c.W),
			tensor.FromF32(w, c.Cin, c.Cout, c.K, c.K),
			tensor.FromF32(bias, c.Cout))[0]
		eqF32(t, "convtranspose", out, []int{c.N, c.Cout, OH, OW}, want)
	}
}
