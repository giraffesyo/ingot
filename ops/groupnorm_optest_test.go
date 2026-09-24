package ops

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// naiveGroupNorm is the float64 oracle: per (n, g) block mean and biased
// variance, then the affine with per-group (opset 18) or per-channel
// (opset 21) scale/bias.
func naiveGroupNorm(x, sc, b []float32, shape []int, G int, eps float64, perChannel bool) []float64 {
	N, C := shape[0], shape[1]
	P := 1
	for _, d := range shape[2:] {
		P *= d
	}
	cpg, L := C/G, C/G*P
	out := make([]float64, len(x))
	for n := range N {
		for g := range G {
			blk := x[(n*G+g)*L : (n*G+g+1)*L]
			var mean float64
			for _, v := range blk {
				mean += float64(v)
			}
			mean /= float64(L)
			var vs float64
			for _, v := range blk {
				d := float64(v) - mean
				vs += d * d
			}
			inv := 1 / math.Sqrt(vs/float64(L)+eps)
			for i, v := range blk {
				s, t := float64(sc[g]), float64(b[g])
				if perChannel {
					c := g*cpg + i/P
					s, t = float64(sc[c]), float64(b[c])
				}
				out[(n*G+g)*L+i] = (float64(v)-mean)*inv*s + t
			}
		}
	}
	return out
}

func TestGroupNormalization(t *testing.T) {
	rng := rand.New(rand.NewPCG(41, 42))
	cases := []struct {
		shape  []int
		G      int
		offset float32 // shifts X to exercise the centred variance
	}{
		{[]int{1, 4, 3, 3}, 2, 0},
		{[]int{2, 6, 5, 7}, 3, 0},
		{[]int{1, 8, 4, 4}, 1, 0},      // G=1: LayerNorm over C·H·W
		{[]int{2, 8, 4, 4}, 8, 0},      // G=C: InstanceNorm
		{[]int{1, 64, 32, 32}, 32, 0},  // VAE-like, L=2048
		{[]int{1, 32, 64, 64}, 4, 100}, // L=32768 > normChunk, large mean
		{[]int{2, 6, 10}, 2, 0},        // rank 3
		{[]int{1, 4, 2, 3, 5}, 2, 0},   // rank 5
		{[]int{3, 6}, 3, 0},            // no spatial dims
	}
	for _, perChannel := range []bool{false, true} {
		ver := 18
		if perChannel {
			ver = 21
		}
		for _, c := range cases {
			t.Run(fmt.Sprintf("opset=%d/shape=%v/G=%d", ver, c.shape, c.G), func(t *testing.T) {
				x := randT(rng, c.shape...)
				for i := range x.F32() {
					x.F32()[i] += c.offset
				}
				n := c.G
				if perChannel {
					n = c.shape[1]
				}
				sc, b := randT(rng, n), randT(rng, n)
				op := mkOp(t, "GroupNormalization", ver, Attrs{
					"num_groups": {Kind: KindInt, I: int64(c.G)},
					"epsilon":    {Kind: KindFloat, F: 1e-5},
				}, 3, 1)
				got := run(t, op, x, sc, b)[0]
				if !got.Shape().Equal(tensor.Shape(c.shape)) {
					t.Fatalf("shape %v, want %v", got.Shape(), c.shape)
				}
				want := naiveGroupNorm(x.F32(), sc.F32(), b.F32(), c.shape, c.G, 1e-5, perChannel)
				L := len(x.F32()) / (c.shape[0] * c.G)
				tol := 1e-5 * math.Sqrt(float64(L))
				for i, w := range want {
					if d := math.Abs(float64(got.F32()[i]) - w); d > tol*(1+math.Abs(w)) {
						t.Fatalf("[%d] = %g, want %g (|Δ| %g > %g)", i, got.F32()[i], w, d, tol*(1+math.Abs(w)))
					}
				}
			})
		}
	}
}

func TestGroupNormalizationErrors(t *testing.T) {
	b, err := Lookup("", "GroupNormalization", 21)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b(NodeInfo{Name: "gn", OpType: "GroupNormalization", Version: 21, NumIn: 3, NumOut: 1}); err == nil {
		t.Fatal("missing num_groups: want build error")
	}
	op := mkOp(t, "GroupNormalization", 21, Attrs{"num_groups": {Kind: KindInt, I: 3}}, 3, 1)
	x := tensor.New(tensor.F32, 1, 4, 2, 2)
	s := tensor.New(tensor.F32, 4)
	if _, err := op.Run(&Ctx{Pool: tensor.NewPool()}, []*tensor.Tensor{x, s, s}); err == nil {
		t.Fatal("C=4, G=3: want run error")
	}
	op = mkOp(t, "GroupNormalization", 21, Attrs{"num_groups": {Kind: KindInt, I: 2}}, 3, 1)
	g := tensor.New(tensor.F32, 2) // opset-18-shaped scale under opset 21
	if _, err := op.Run(&Ctx{Pool: tensor.NewPool()}, []*tensor.Tensor{x, g, g}); err == nil {
		t.Fatal("per-group scale at opset 21: want run error")
	}
}

// TestSinCos checks the trig ops against float64 math, including large
// arguments (timestep embeddings reach |x| ~ 1e3) and exact zeros.
func TestSinCos(t *testing.T) {
	rng := rand.New(rand.NewPCG(43, 44))
	x := tensor.New(tensor.F32, 4, 257)
	xf := x.F32()
	for i := range xf {
		xf[i] = (rng.Float32()*2 - 1) * float32([]float64{1, 10, 1e3, 1e4}[i/257])
	}
	xf[0] = 0
	for _, c := range []struct {
		name string
		fn   func(float64) float64
	}{{"Sin", math.Sin}, {"Cos", math.Cos}} {
		got := run(t, mkOp(t, c.name, 7, nil, 1, 1), x)[0].F32()
		for i, v := range xf {
			w := c.fn(float64(v))
			if d := math.Abs(float64(got[i]) - w); d > 1e-6 {
				t.Fatalf("%s(%g) = %g, want %g", c.name, v, got[i], w)
			}
		}
	}
}
