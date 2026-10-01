package ops

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// conv1dRef is an independent NCW conv oracle with asymmetric pads (float64
// accumulation).
func conv1dRef(x, w, bias []float32, N, C, T, M, K, G, s, d, pl, pr int) ([]float32, int) {
	Cg, Mg := C/G, M/G
	OT := (T+pl+pr-(d*(K-1)+1))/s + 1
	out := make([]float32, N*M*OT)
	for n := range N {
		for m := range M {
			g := m / Mg
			for o := range OT {
				acc := float64(bias[m])
				for cg := range Cg {
					c := g*Cg + cg
					for k := range K {
						if i := o*s + k*d - pl; i >= 0 && i < T {
							acc += float64(x[(n*C+c)*T+i]) * float64(w[(m*Cg+cg)*K+k])
						}
					}
				}
				out[(n*M+m)*OT+o] = float32(acc)
			}
		}
	}
	return out, OT
}

// convTranspose1dRef scatters each input sample through the kernel:
// out[oc][i·s + k·d − pl] += x[ic][i]·w[ic][oc][k] (groups 1).
func convTranspose1dRef(x, w, bias []float32, N, Cin, T, Cout, K, s, d, pl, pr int) ([]float32, int) {
	OT := (T-1)*s - pl - pr + d*(K-1) + 1
	acc := make([]float64, N*Cout*OT)
	for n := range N {
		for oc := range Cout {
			for o := range OT {
				acc[(n*Cout+oc)*OT+o] = float64(bias[oc])
			}
			for ic := range Cin {
				for i := range T {
					for k := range K {
						if o := i*s + k*d - pl; o >= 0 && o < OT {
							acc[(n*Cout+oc)*OT+o] += float64(x[(n*Cin+ic)*T+i]) * float64(w[(ic*Cout+oc)*K+k])
						}
					}
				}
			}
		}
	}
	out := make([]float32, len(acc))
	for i, v := range acc {
		out[i] = float32(v)
	}
	return out, OT
}

// TestConv1D: rank-3 Conv (1-D attributes) against the oracle over the
// branches the 2-D lowering takes — causal left-only padding, dilation,
// depthwise, grouped, strided, pointwise.
func TestConv1D(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	rnd := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = r.Float32()*2 - 1
		}
		return s
	}
	for _, c := range []struct{ N, C, T, M, K, G, s, d, pl, pr int }{
		{1, 8, 37, 12, 7, 1, 1, 1, 6, 0},   // causal k7
		{1, 8, 40, 8, 7, 1, 1, 3, 18, 0},   // causal dilated (d3)
		{2, 16, 29, 16, 7, 16, 1, 1, 6, 0}, // depthwise causal
		{1, 12, 33, 6, 3, 3, 1, 1, 1, 1},   // grouped, symmetric
		{1, 4, 50, 8, 4, 1, 2, 1, 2, 1},    // stride 2
		{1, 16, 21, 24, 1, 1, 1, 1, 0, 0},  // pointwise
	} {
		t.Run(fmt.Sprintf("%+v", c), func(t *testing.T) {
			x, w, b := rnd(c.N*c.C*c.T), rnd(c.M*(c.C/c.G)*c.K), rnd(c.M)
			want, OT := conv1dRef(x, w, b, c.N, c.C, c.T, c.M, c.K, c.G, c.s, c.d, c.pl, c.pr)
			op := mkOp(t, "Conv", 11, Attrs{
				"group":     {Kind: KindInt, I: int64(c.G)},
				"strides":   {Kind: KindInts, Ints: []int64{int64(c.s)}},
				"dilations": {Kind: KindInts, Ints: []int64{int64(c.d)}},
				"pads":      {Kind: KindInts, Ints: []int64{int64(c.pl), int64(c.pr)}},
			}, 3, 1)
			out := run(t, op, tensor.FromF32(x, c.N, c.C, c.T), tensor.FromF32(w, c.M, c.C/c.G, c.K), tensor.FromF32(b, c.M))[0]
			eqF32(t, "conv1d", out, []int{c.N, c.M, OT}, want)
		})
	}
}

// TestConvTranspose1D: rank-3 ConvTranspose against the scatter oracle —
// the upsampling shapes vocoders use (kernel = 2·stride, and kernel =
// stride, non-overlapping) plus padding and dilation.
func TestConvTranspose1D(t *testing.T) {
	r := rand.New(rand.NewPCG(13, 14))
	rnd := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = r.Float32()*2 - 1
		}
		return s
	}
	for _, c := range []struct{ N, Cin, T, Cout, K, s, d, pl, pr int }{
		{1, 8, 13, 4, 16, 8, 1, 0, 0}, // k = 2s (overlapping)
		{1, 6, 11, 6, 2, 2, 1, 0, 0},  // k = s (each output written once)
		{2, 4, 9, 5, 6, 3, 1, 0, 0},
		{1, 4, 10, 3, 5, 2, 2, 1, 2}, // dilation + pads
	} {
		t.Run(fmt.Sprintf("%+v", c), func(t *testing.T) {
			x, w, b := rnd(c.N*c.Cin*c.T), rnd(c.Cin*c.Cout*c.K), rnd(c.Cout)
			want, OT := convTranspose1dRef(x, w, b, c.N, c.Cin, c.T, c.Cout, c.K, c.s, c.d, c.pl, c.pr)
			op := mkOp(t, "ConvTranspose", 11, Attrs{
				"strides":   {Kind: KindInts, Ints: []int64{int64(c.s)}},
				"dilations": {Kind: KindInts, Ints: []int64{int64(c.d)}},
				"pads":      {Kind: KindInts, Ints: []int64{int64(c.pl), int64(c.pr)}},
			}, 3, 1)
			out := run(t, op, tensor.FromF32(x, c.N, c.Cin, c.T), tensor.FromF32(w, c.Cin, c.Cout, c.K), tensor.FromF32(b, c.Cout))[0]
			eqF32(t, "convtranspose1d", out, []int{c.N, c.Cout, OT}, want)
		})
	}
}
