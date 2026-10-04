package ops

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// conv3dRef is an independent NCDHW conv oracle (float64 accumulation):
// s, d per axis; p = front, top, left, back, bottom, right.
func conv3dRef(x, w, bias []float32, N, C int, in [3]int, M int, k [3]int, G int, s, d [3]int, p [6]int) ([]float32, [3]int) {
	Cg, Mg := C/G, M/G
	var o [3]int
	for a := range 3 {
		o[a] = (in[a]+p[a]+p[a+3]-(d[a]*(k[a]-1)+1))/s[a] + 1
	}
	V, KV, OV := in[0]*in[1]*in[2], k[0]*k[1]*k[2], o[0]*o[1]*o[2]
	out := make([]float32, N*M*OV)
	for n := range N {
		for m := range M {
			g := m / Mg
			for od := range o[0] {
				for oh := range o[1] {
					for ow := range o[2] {
						acc := float64(bias[m])
						for cg := range Cg {
							c := g*Cg + cg
							for kd := range k[0] {
								for kh := range k[1] {
									for kw := range k[2] {
										id, ih, iw := od*s[0]+kd*d[0]-p[0], oh*s[1]+kh*d[1]-p[1], ow*s[2]+kw*d[2]-p[2]
										if id < 0 || id >= in[0] || ih < 0 || ih >= in[1] || iw < 0 || iw >= in[2] {
											continue
										}
										acc += float64(x[(n*C+c)*V+(id*in[1]+ih)*in[2]+iw]) *
											float64(w[(m*Cg+cg)*KV+(kd*k[1]+kh)*k[2]+kw])
									}
								}
							}
						}
						out[(n*M+m)*OV+(od*o[1]+oh)*o[2]+ow] = float32(acc)
					}
				}
			}
		}
	}
	return out, o
}

func conv3dAttrs(G int, s, d [3]int, p [6]int) Attrs {
	return Attrs{
		"group":     {Kind: KindInt, I: int64(G)},
		"strides":   {Kind: KindInts, Ints: []int64{int64(s[0]), int64(s[1]), int64(s[2])}},
		"dilations": {Kind: KindInts, Ints: []int64{int64(d[0]), int64(d[1]), int64(d[2])}},
		"pads":      {Kind: KindInts, Ints: []int64{int64(p[0]), int64(p[1]), int64(p[2]), int64(p[3]), int64(p[4]), int64(p[5])}},
	}
}

// TestConv3D: rank-5 Conv against the oracle over the branches vol2col
// takes — same-padded 3³, the 2³ stride-2 downsample, anisotropic kernels
// and strides, dilation, asymmetric pads, groups, and the 1×1×1 path that
// skips the column matrix.
func TestConv3D(t *testing.T) {
	r := rand.New(rand.NewPCG(31, 32))
	rnd := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = r.Float32()*2 - 1
		}
		return s
	}
	one := [3]int{1, 1, 1}
	for _, c := range []struct {
		N, C int
		in   [3]int
		M    int
		k    [3]int
		G    int
		s, d [3]int
		p    [6]int
	}{
		{1, 8, [3]int{6, 7, 5}, 12, [3]int{3, 3, 3}, 1, one, one, [6]int{1, 1, 1, 1, 1, 1}},        // same-padded 3³
		{2, 4, [3]int{8, 6, 10}, 6, [3]int{2, 2, 2}, 1, [3]int{2, 2, 2}, one, [6]int{}},            // 2³ stride-2
		{1, 3, [3]int{9, 8, 7}, 5, [3]int{1, 3, 2}, 1, [3]int{1, 2, 3}, one, [6]int{0, 1, 0, 0}},   // anisotropic
		{1, 4, [3]int{11, 9, 10}, 4, [3]int{3, 3, 3}, 1, one, [3]int{2, 1, 3}, [6]int{2, 0, 3, 1}}, // dilated, asymmetric pads
		{1, 12, [3]int{5, 6, 4}, 6, [3]int{3, 3, 3}, 3, one, one, [6]int{1, 1, 1, 1, 1, 1}},        // grouped
		{2, 16, [3]int{4, 5, 6}, 24, one, 1, one, one, [6]int{}},                                   // pointwise
		{1, 64, [3]int{12, 12, 12}, 32, [3]int{3, 3, 3}, 1, one, one, [6]int{1, 1, 1, 1, 1, 1}},    // multi-tile
	} {
		t.Run(fmt.Sprintf("%+v", c), func(t *testing.T) {
			V, KV := c.in[0]*c.in[1]*c.in[2], c.k[0]*c.k[1]*c.k[2]
			x, w, b := rnd(c.N*c.C*V), rnd(c.M*(c.C/c.G)*KV), rnd(c.M)
			want, o := conv3dRef(x, w, b, c.N, c.C, c.in, c.M, c.k, c.G, c.s, c.d, c.p)
			op := mkOp(t, "Conv", 11, conv3dAttrs(c.G, c.s, c.d, c.p), 3, 1)
			out := run(t, op, tensor.FromF32(x, c.N, c.C, c.in[0], c.in[1], c.in[2]),
				tensor.FromF32(w, c.M, c.C/c.G, c.k[0], c.k[1], c.k[2]), tensor.FromF32(b, c.M))[0]
			eqF32(t, "conv3d", out, []int{c.N, c.M, o[0], o[1], o[2]}, want)
		})
	}
	// Rank-5 operands with every attribute defaulted (unit stride, no pad).
	x, w, b := rnd(2*3*4*4*4), rnd(5*3*2*2*2), rnd(5)
	want, o := conv3dRef(x, w, b, 2, 3, [3]int{4, 4, 4}, 5, [3]int{2, 2, 2}, 1, one, one, [6]int{})
	out := run(t, mkOp(t, "Conv", 11, nil, 3, 1), tensor.FromF32(x, 2, 3, 4, 4, 4), tensor.FromF32(w, 5, 3, 2, 2, 2), tensor.FromF32(b, 5))[0]
	eqF32(t, "conv3d defaults", out, []int{2, 5, o[0], o[1], o[2]}, want)
}

// BenchmarkConv3D: same-padded 3³ convs at the shapes a volumetric decoder
// runs (wide and coarse, then narrow and fine).
func BenchmarkConv3D(b *testing.B) {
	r := rand.New(rand.NewPCG(33, 34))
	for _, c := range []struct{ C, M, S int }{
		{512, 512, 16},
		{128, 128, 32},
		{32, 32, 64},
	} {
		x := tensor.New(tensor.F32, 1, c.C, c.S, c.S, c.S)
		for i := range x.F32() {
			x.F32()[i] = r.Float32()*2 - 1
		}
		w := tensor.New(tensor.F32, c.M, c.C, 3, 3, 3)
		for i := range w.F32() {
			w.F32()[i] = r.Float32()*2 - 1
		}
		bias := tensor.New(tensor.F32, c.M)
		op := mkOpB(b, "Conv", 11, conv3dAttrs(1, [3]int{1, 1, 1}, [3]int{1, 1, 1}, [6]int{1, 1, 1, 1, 1, 1}), 3, 1)
		ctx := &Ctx{Pool: tensor.NewPool()}
		in := []*tensor.Tensor{x, w, bias}
		b.Run(fmt.Sprintf("shape=%dx%d^3->%d", c.C, c.S, c.M), func(b *testing.B) {
			flops := 2 * float64(c.M) * float64(c.C) * 27 * float64(c.S*c.S*c.S)
			b.ReportAllocs()
			for b.Loop() {
				out, err := op.Run(ctx, in)
				if err != nil {
					b.Fatal(err)
				}
				ctx.Pool.Put(out[0])
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}
