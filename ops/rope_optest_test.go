package ops

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

func mkRoPE(t testing.TB, layout int64) Op {
	t.Helper()
	b, err := Lookup("ingot", "RoPE", 1)
	if err != nil {
		t.Fatal(err)
	}
	op, err := b(NodeInfo{Name: "rope", OpType: "RoPE", Domain: "ingot", Version: 1, NumIn: 3, NumOut: 1,
		Attrs: Attrs{"layout": {Kind: KindInt, I: layout}}})
	if err != nil {
		t.Fatal(err)
	}
	return op
}

// TestRoPE: both layouts vs a float64 oracle, heads sharing their token's
// angles, a head dim that is not a power of two.
func TestRoPE(t *testing.T) {
	r := rand.New(rand.NewPCG(71, 71))
	const T, H, dh = 5, 3, 72
	x, c, s := randT(r, T, H, dh), randT(r, T, dh/2), randT(r, T, dh/2)
	for layout := range int64(2) {
		got := run(t, mkRoPE(t, layout), x, c, s)[0].F32()
		for tt := range T {
			for h := range H {
				base := (tt*H + h) * dh
				for j := range dh / 2 {
					ct, st := float64(c.F32()[tt*dh/2+j]), float64(s.F32()[tt*dh/2+j])
					ia, ib := 2*j, 2*j+1
					if layout == 1 {
						ia, ib = j, j+dh/2
					}
					a, b := float64(x.F32()[base+ia]), float64(x.F32()[base+ib])
					wa, wb := a*ct-b*st, b*ct+a*st
					if math.Abs(float64(got[base+ia])-wa) > 1e-5 || math.Abs(float64(got[base+ib])-wb) > 1e-5 {
						t.Fatalf("layout %d t=%d h=%d j=%d: (%g,%g) want (%g,%g)", layout, tt, h, j, got[base+ia], got[base+ib], wa, wb)
					}
				}
			}
		}
	}
}

// BenchmarkRoPE times a DiT q/k rotation ([T, 32 heads, 128]).
func BenchmarkRoPE(b *testing.B) {
	for _, T := range []int{256, 4096} {
		x, c, s := tensor.New(tensor.F32, T, 32, 128), tensor.New(tensor.F32, T, 64), tensor.New(tensor.F32, T, 64)
		op := mkRoPE(b, 0)
		ctx := &Ctx{Pool: tensor.NewPool()}
		b.Run(fmt.Sprintf("T=%d", T), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(8 * x.Numel()))
			for i := 0; i < b.N; i++ {
				out, err := op.Run(ctx, []*tensor.Tensor{x, c, s})
				if err != nil {
					b.Fatal(err)
				}
				ctx.Pool.Put(out[0])
			}
		})
	}
}
