package ops

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"
	"time"
	"weak"

	"github.com/giraffesyo/ingot/tensor"
)

func bf16Tensor(r *rand.Rand, shape ...int) (*tensor.Tensor, []float64) {
	t := tensor.New(tensor.BF16, shape...)
	wide := make([]float64, t.Numel())
	for i := range t.BF16() {
		t.BF16()[i] = uint16(math.Float32bits(r.Float32()*2-1) >> 16)
		wide[i] = float64(math.Float32frombits(uint32(t.BF16()[i]) << 16))
	}
	return t, wide
}

// TestGemmBF16Weight: Gemm with a bf16 B against the float64 oracle over
// the widened weights, both storage orders, with bias and alpha/beta.
func TestGemmBF16Weight(t *testing.T) {
	r := rand.New(rand.NewPCG(81, 82))
	const M, K, N = 37, 300, 29
	for _, transB := range []int64{0, 1} {
		a := randT(r, M, K)
		shape := []int{K, N}
		if transB == 1 {
			shape = []int{N, K}
		}
		b, wide := bf16Tensor(r, shape...)
		c := randT(r, N)
		op := mkOp(t, "Gemm", 13, Attrs{"transB": {Kind: KindInt, I: transB}, "alpha": {Kind: KindFloat, F: 0.5},
			"beta": {Kind: KindFloat, F: 2}}, 3, 1)
		got := run(t, op, a, b, c)[0].F32()
		for i := range M {
			for j := range N {
				want := 2 * float64(c.F32()[j])
				for k := range K {
					w := wide[k*N+j]
					if transB == 1 {
						w = wide[j*K+k]
					}
					want += 0.5 * float64(a.F32()[i*K+k]) * w
				}
				if d := math.Abs(float64(got[i*N+j]) - want); d > 1e-5*math.Sqrt(K)*(1+math.Abs(want)) {
					t.Fatalf("transB=%d [%d,%d] = %g, want %g", transB, i, j, got[i*N+j], want)
				}
			}
		}
	}
}

// TestBF16PackShared: two ops over one weight share its pack, and the
// cache entry goes when the weight is collected.
func TestBF16PackShared(t *testing.T) {
	r := rand.New(rand.NewPCG(83, 84))
	var key bf16PackKey
	func() {
		b, _ := bf16Tensor(r, 64, 48)
		p1 := packedBF16(b, true, 48, 64)
		p2 := packedBF16(b, true, 48, 64)
		if p1 != p2 {
			t.Fatal("same weight packed twice")
		}
		if p3 := packedBF16(b, false, 64, 48); p3 == p1 {
			t.Fatal("transB must be part of the key")
		}
		key = bf16PackKey{weak.Make(b), true}
		if _, ok := bf16Packs.Load(key); !ok {
			t.Fatal("no cache entry for a packed weight")
		}
		runtime.KeepAlive(b)
	}()
	for i := 0; i < 50; i++ {
		if _, ok := bf16Packs.Load(key); !ok {
			return
		}
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("cache entry outlived its weight")
}
