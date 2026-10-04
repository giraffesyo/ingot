package sparse

import (
	"math/rand/v2"
	"testing"
)

// randomCells returns n distinct cells of a side³ grid.
func randomCells(r *rand.Rand, n, side int) []Coord {
	seen := map[Coord]bool{}
	out := make([]Coord, 0, n)
	for len(out) < n {
		c := Coord{int32(r.IntN(side)), int32(r.IntN(side)), int32(r.IntN(side))}
		if !seen[c] {
			seen[c] = true
			out = append(out, c)
		}
	}
	return out
}

// TestNeighbors checks the table against a map-based oracle, over kernel
// sizes and dilations, on a grid dense enough that most cells have
// neighbours and sparse enough that many do not.
func TestNeighbors(t *testing.T) {
	r := rand.New(rand.NewPCG(41, 42))
	for _, c := range []struct{ n, side, k, dil int }{
		{500, 12, 3, 1}, {3000, 20, 3, 1}, {400, 10, 3, 2}, {300, 9, 5, 1}, {1, 4, 3, 1},
	} {
		coords := randomCells(r, c.n, c.side)
		got, err := Neighbors(coords, c.k, c.dil)
		if err != nil {
			t.Fatal(err)
		}
		row := map[Coord]int32{}
		for i, p := range coords {
			row[p] = int32(i)
		}
		V, h := c.k*c.k*c.k, int32(c.k/2*c.dil)
		for i, p := range coords {
			v := 0
			for a := range int32(c.k) {
				for b := range int32(c.k) {
					for e := range int32(c.k) {
						want, ok := row[Coord{p[0] - h + a*int32(c.dil), p[1] - h + b*int32(c.dil), p[2] - h + e*int32(c.dil)}]
						if !ok {
							want = -1
						}
						if got[i*V+v] != want {
							t.Fatalf("%+v: cell %d tap %d = %d, want %d", c, i, v, got[i*V+v], want)
						}
						v++
					}
				}
			}
		}
	}
}

func TestIndexErrors(t *testing.T) {
	if _, err := NewIndex([]Coord{{1, 2, 3}, {4, 5, 6}, {1, 2, 3}}); err == nil {
		t.Fatal("duplicate coordinate accepted")
	}
	if _, err := NewIndex([]Coord{{-1, 0, 0}}); err == nil {
		t.Fatal("negative coordinate accepted")
	}
	ix, err := NewIndex([]Coord{{0, 0, 0}, {7, 8, 9}})
	if err != nil {
		t.Fatal(err)
	}
	if ix.Lookup(Coord{7, 8, 9}) != 1 || ix.Lookup(Coord{7, 8, 8}) != -1 || ix.Lookup(Coord{-1, 0, 0}) != -1 {
		t.Fatal("lookup mismatch")
	}
}

// TestSubdivide: children land at 2·parent + (dx, dy, dz) in parent order,
// and src addresses [parent, sub-cell].
func TestSubdivide(t *testing.T) {
	coords := []Coord{{1, 2, 3}, {0, 0, 5}}
	mask := make([]bool, 16)
	mask[0*8+0], mask[0*8+5], mask[1*8+2], mask[1*8+7] = true, true, true, true
	ch, src, err := Subdivide(coords, mask)
	if err != nil {
		t.Fatal(err)
	}
	wantC := []Coord{{2, 4, 6}, {3, 4, 7}, {0, 1, 10}, {1, 1, 11}}
	wantS := []int32{0, 5, 10, 15}
	for i := range wantC {
		if ch[i] != wantC[i] || src[i] != wantS[i] {
			t.Fatalf("child %d = %v src %d, want %v src %d", i, ch[i], src[i], wantC[i], wantS[i])
		}
	}
	if _, _, err := Subdivide(coords, mask[:8]); err == nil {
		t.Fatal("short mask accepted")
	}
}

func BenchmarkNeighbors(b *testing.B) {
	r := rand.New(rand.NewPCG(43, 44))
	coords := randomCells(r, 200000, 128)
	b.Run("shape=200000x27", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := Neighbors(coords, 3, 1); err != nil {
				b.Fatal(err)
			}
		}
	})
}
