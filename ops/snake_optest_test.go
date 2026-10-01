package ops

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestSnake: ingot.Snake against a float64 oracle, rank 3 (N, C, T) and
// rank 4 (per-channel parameters broadcast over every trailing position).
func TestSnake(t *testing.T) {
	r := rand.New(rand.NewPCG(21, 22))
	for _, shape := range [][]int{{2, 5, 37}, {1, 3, 4, 6}, {1, 7, 1}} {
		x := randT(r, shape...)
		C := shape[1]
		fr, sc := randT(r, C), randT(r, C)
		build, err := Lookup("ingot", "Snake", 1)
		if err != nil {
			t.Fatal(err)
		}
		inst, err := build(NodeInfo{Name: "snake", OpType: "Snake", Domain: "ingot", NumIn: 3, NumOut: 1})
		if err != nil {
			t.Fatal(err)
		}
		got := run(t, inst, x, fr, sc)[0]
		inner := x.Numel() / (shape[0] * C)
		want := make([]float32, x.Numel())
		for i, v := range x.F32() {
			c := (i / inner) % C
			s := math.Sin(float64(v) * float64(fr.F32()[c]))
			want[i] = float32(float64(v) + float64(sc.F32()[c])*s*s)
		}
		eqF32(t, "snake", got, shape, want)
	}
}
