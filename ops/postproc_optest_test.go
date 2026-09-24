package ops

import (
	"math/rand/v2"
	"sort"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// TestTopKOracle: TopK (selection path for small k, sort path above it)
// vs a naive stable-sort oracle over random rows with many ties, both
// largest and smallest, every k, on a middle axis.
func TestTopKOracle(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	for _, n := range []int{1, 7, 40, 70} {
		for _, largest := range []int64{0, 1} {
			for _, k := range []int{0, 1, 3, n / 2, n} {
				if k > n {
					continue
				}
				const outer, inner = 2, 3
				x := tensor.New(tensor.F32, outer, n, inner)
				for i := range x.F32() {
					x.F32()[i] = float32(r.IntN(9)) // ties everywhere
				}
				kt := tensor.New(tensor.I64, 1)
				kt.I64()[0] = int64(k)
				op := mkOp(t, "TopK", 11, Attrs{"axis": {Kind: KindInt, I: 1}, "largest": {Kind: KindInt, I: largest}}, 2, 2)
				outs, err := op.Run(&Ctx{}, []*tensor.Tensor{x, kt})
				if err != nil {
					t.Fatal(err)
				}
				vf, ifc := outs[0].F32(), outs[1].I64()
				for oi := range outer {
					for ii := range inner {
						ord := make([]int, n)
						for j := range ord {
							ord[j] = j
						}
						at := func(j int) float32 { return x.F32()[(oi*n+j)*inner+ii] }
						sort.SliceStable(ord, func(a, b int) bool {
							if largest == 1 {
								return at(ord[a]) > at(ord[b])
							}
							return at(ord[a]) < at(ord[b])
						})
						for j := range k {
							got := (oi*k+j)*inner + ii
							if ifc[got] != int64(ord[j]) || vf[got] != at(ord[j]) {
								t.Fatalf("n=%d k=%d largest=%d [%d,%d,%d]: (%g, %d), want (%g, %d)",
									n, k, largest, oi, j, ii, vf[got], ifc[got], at(ord[j]), ord[j])
							}
						}
					}
				}
			}
		}
	}
}
