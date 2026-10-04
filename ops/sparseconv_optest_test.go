package ops

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// sparseConvRef is the definition, accumulated in float64:
// out[i][co] = bias[co] + Σ_v Σ_ci W[co][v][ci]·X[nbr[i][v]][ci].
func sparseConvRef(x []float32, nbr []int32, w, bias []float32, N, Ci, V, Co int) []float32 {
	out := make([]float32, N*Co)
	for i := range N {
		for co := range Co {
			acc := 0.0
			if bias != nil {
				acc = float64(bias[co])
			}
			for v := range V {
				j := nbr[i*V+v]
				if j < 0 {
					continue
				}
				for ci := range Ci {
					acc += float64(w[(co*V+v)*Ci+ci]) * float64(x[int(j)*Ci+ci])
				}
			}
			out[i*Co+co] = float32(acc)
		}
	}
	return out
}

func mkIngotOp(t testing.TB, name string, attrs Attrs, numIn int) Op {
	t.Helper()
	b, err := Lookup("ingot", name, 1)
	if err != nil {
		t.Fatalf("Lookup ingot.%s: %v", name, err)
	}
	op, err := b(NodeInfo{Name: name, OpType: name, Domain: "ingot", Version: 1, Attrs: attrs, NumIn: numIn, NumOut: 1})
	if err != nil {
		t.Fatalf("build ingot.%s: %v", name, err)
	}
	return op
}

// sparseConvCase builds random features, weights and a neighbour table in
// which roughly `fill` of the taps are occupied (the centre always).
func sparseConvCase(r *rand.Rand, N, Ci, V, Co int, fill float64) (x []float32, nbr []int32, w, bias []float32) {
	rnd := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = r.Float32()*2 - 1
		}
		return s
	}
	nbr = make([]int32, N*V)
	for i := range N {
		for v := range V {
			switch {
			case v == V/2:
				nbr[i*V+v] = int32(i)
			case r.Float64() < fill:
				nbr[i*V+v] = int32(r.IntN(N))
			default:
				nbr[i*V+v] = -1
			}
		}
	}
	return rnd(N * Ci), nbr, rnd(Co * V * Ci), rnd(Co)
}

// TestSparseConv: against the oracle over single- and multi-tile row
// counts (the tile budget is lowered to reach the tiled path at test
// sizes), with and without bias, a 1-tap kernel, and an empty voxel set.
func TestSparseConv(t *testing.T) {
	r := rand.New(rand.NewPCG(51, 52))
	for _, c := range []struct {
		N, Ci, V, Co int
		fill         float64
		bias         bool
	}{
		{37, 8, 27, 12, 0.4, true},
		{500, 16, 27, 8, 0.3, true},  // several tiles
		{129, 5, 27, 33, 0.9, false}, // ragged channels, no bias
		{64, 24, 1, 24, 1, true},     // 1-tap kernel
		{200, 32, 125, 4, 0.2, true}, // 5³ kernel
		{1, 7, 27, 3, 0, true},       // lone voxel
	} {
		t.Run(fmt.Sprintf("%+v", c), func(t *testing.T) {
			x, nbr, w, bias := sparseConvCase(r, c.N, c.Ci, c.V, c.Co, c.fill)
			in := []*tensor.Tensor{tensor.FromF32(x, c.N, c.Ci), tensor.FromI32(nbr, c.N, c.V), tensor.FromF32(w, c.Co, c.V, c.Ci)}
			if c.bias {
				in = append(in, tensor.FromF32(bias, c.Co))
			} else {
				bias = nil
			}
			want := sparseConvRef(x, nbr, w, bias, c.N, c.Ci, c.V, c.Co)
			out := run(t, mkIngotOp(t, "SparseConv", nil, len(in)), in...)[0]
			eqF32(t, "sparseconv", out, []int{c.N, c.Co}, want)
		})
	}
	op := mkIngotOp(t, "SparseConv", nil, 3)
	out := run(t, op, tensor.New(tensor.F32, 0, 4), tensor.New(tensor.I32, 0, 27), tensor.New(tensor.F32, 6, 27, 4))[0]
	if !out.Shape().Equal(tensor.Shape{0, 6}) {
		t.Fatalf("empty set: shape %v", out.Shape())
	}
	// A neighbour row past the set is an error, not a read out of bounds.
	x, nbr, w, _ := sparseConvCase(r, 10, 4, 27, 4, 0.5)
	nbr[5] = 10
	if _, err := op.Run(&Ctx{Pool: tensor.NewPool()}, []*tensor.Tensor{tensor.FromF32(x, 10, 4), tensor.FromI32(nbr, 10, 27), tensor.FromF32(w, 4, 27, 4)}); err == nil {
		t.Fatal("out-of-range neighbour accepted")
	}
}

// BenchmarkSparseConv: 3³ submanifold convs at the shapes a sparse voxel
// decoder runs — wide over few cells, narrow over many — with a third of
// the taps occupied, as on a surface.
func BenchmarkSparseConv(b *testing.B) {
	r := rand.New(rand.NewPCG(53, 54))
	for _, c := range []struct{ N, Ci, Co int }{
		{4000, 1024, 1024},
		{50000, 256, 256},
		{200000, 128, 128},
		{200000, 128, 512},
	} {
		x, nbr, w, bias := sparseConvCase(r, c.N, c.Ci, 27, c.Co, 0.33)
		in := []*tensor.Tensor{tensor.FromF32(x, c.N, c.Ci), tensor.FromI32(nbr, c.N, 27), tensor.FromF32(w, c.Co, 27, c.Ci), tensor.FromF32(bias, c.Co)}
		op := mkIngotOp(b, "SparseConv", nil, 4)
		ctx := &Ctx{Pool: tensor.NewPool()}
		b.Run(fmt.Sprintf("shape=%dx%d->%d", c.N, c.Ci, c.Co), func(b *testing.B) {
			flops := 2 * float64(c.N) * 27 * float64(c.Ci) * float64(c.Co)
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
