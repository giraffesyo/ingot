package ops

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// naiveSDPA is the float64 oracle: softmax(scale·q·kᵀ + mask)·v per (b, h),
// with q [B,H,T,dh], k [B,H,Tk,dh] (row-major, not transposed), v [B,H,Tk,dh].
func naiveSDPA(q, k, v, mask []float32, B, H, T, Tk, dh int, scale float64) []float64 {
	out := make([]float64, B*H*T*dh)
	s := make([]float64, Tk)
	for bh := range B * H {
		for t := range T {
			mx := math.Inf(-1)
			for j := range Tk {
				var d float64
				for c := range dh {
					d += float64(q[(bh*T+t)*dh+c]) * float64(k[(bh*Tk+j)*dh+c])
				}
				s[j] = d * scale
				if mask != nil {
					s[j] += float64(mask[t*Tk+j])
				}
				mx = max(mx, s[j])
			}
			var sum float64
			for j := range Tk {
				s[j] = math.Exp(s[j] - mx)
				sum += s[j]
			}
			for c := range dh {
				var acc float64
				for j := range Tk {
					acc += s[j] * float64(v[(bh*Tk+j)*dh+c])
				}
				out[(bh*T+t)*dh+c] = acc / sum
			}
		}
	}
	return out
}

// TestSDPAOracle covers the materialised and both flash paths (masked with
// block skipping; unmasked with wide key blocks) against the float64 oracle,
// including ragged T/Tk that leave partial row tiles and key blocks, and both
// K layouts ([B,H,dh,Tk] default and [B,H,Tk,dh]).
func TestSDPAOracle(t *testing.T) {
	defer func(v int) { sdpaFlashTile = v }(sdpaFlashTile)
	rng := rand.New(rand.NewPCG(51, 52))
	type mk int
	const (
		none mk = iota
		causal
		keyPad // last Tk/5 keys masked for every query: mixed blocks
	)
	cases := []struct {
		B, H, T, Tk, dh int
		mask            mk
		forceFlash      bool
	}{
		{1, 2, 64, 64, 32, none, false},       // materialised
		{2, 3, 97, 150, 16, none, false},      // materialised, ragged
		{1, 3, 300, 1100, 64, none, true},     // unmasked flash, Tk % 512 != 0
		{2, 2, 129, 2048, 48, none, true},     // unmasked flash, partial row tile
		{1, 1, 40, 700, 1152 / 8, none, true}, // wide head
		{1, 2, 600, 600, 32, causal, false},   // masked flash, block skip
		{1, 2, 257, 900, 32, keyPad, false},   // masked flash, mixed blocks
	}
	for _, c := range cases {
		for _, bLay := range []int64{0, 2} {
			name := fmt.Sprintf("B=%d/H=%d/T=%d/Tk=%d/dh=%d/mask=%d/flash=%v/bLay=%d", c.B, c.H, c.T, c.Tk, c.dh, c.mask, c.forceFlash, bLay)
			t.Run(name, func(t *testing.T) {
				sdpaFlashTile = 1 << 20
				if c.forceFlash {
					sdpaFlashTile = 1
				}
				q := randT(rng, c.B, c.H, c.T, c.dh)
				k := randT(rng, c.B, c.H, c.Tk, c.dh) // [B,H,Tk,dh]
				v := randT(rng, c.B, c.H, c.Tk, c.dh)
				kIn := k
				if bLay == 0 { // [B,H,dh,Tk]
					kIn = tensor.New(tensor.F32, c.B, c.H, c.dh, c.Tk)
					for bh := range c.B * c.H {
						for j := range c.Tk {
							for d := range c.dh {
								kIn.F32()[(bh*c.dh+d)*c.Tk+j] = k.F32()[(bh*c.Tk+j)*c.dh+d]
							}
						}
					}
				}
				in := []*tensor.Tensor{q, kIn, v}
				var mask []float32
				if c.mask != none {
					m := tensor.New(tensor.F32, c.T, c.Tk)
					for i := range c.T {
						for j := range c.Tk {
							if (c.mask == causal && j > i) || (c.mask == keyPad && j >= c.Tk-c.Tk/5) {
								m.F32()[i*c.Tk+j] = float32(math.Inf(-1))
							}
						}
					}
					mask = m.F32()
					in = append(in, m)
				}
				scale := 1 / math.Sqrt(float64(c.dh))
				bld, err := Lookup("ingot", "SDPA", 1)
				if err != nil {
					t.Fatal(err)
				}
				op, err := bld(NodeInfo{Name: "sdpa", OpType: "SDPA", Domain: "ingot", Version: 1,
					Attrs: Attrs{"scale": {Kind: KindFloat, F: float32(scale)}, "b_layout": {Kind: KindInt, I: bLay}},
					NumIn: len(in), NumOut: 1})
				if err != nil {
					t.Fatal(err)
				}
				got := run(t, op, in...)[0].F32()
				want := naiveSDPA(q.F32(), k.F32(), v.F32(), mask, c.B, c.H, c.T, c.Tk, c.dh, scale)
				// Outputs are convex combinations of v (|v| ≤ 2); the
				// reductions run over Tk scores and dh products.
				tol := 2e-5 * math.Sqrt(float64(max(c.Tk, c.dh)))
				for i, w := range want {
					if d := math.Abs(float64(got[i]) - w); d > tol {
						t.Fatalf("[%d] = %g, want %g (|Δ| %.2e > %.2e)", i, got[i], w, d, tol)
					}
				}
			})
		}
	}
}
