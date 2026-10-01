package generate

import (
	"math"
	"testing"
)

// TestProcessors pins each HF logits processor on hand-computed rows.
func TestProcessors(t *testing.T) {
	ninf := float32(math.Inf(-1))
	for _, c := range []struct {
		name    string
		s       Sampler
		in      []float32
		history []int64
		want    []float32
	}{
		{"repetition", Sampler{RepetitionPenalty: 2}, []float32{4, -4, 1, 3}, []int64{0, 1, 1}, []float32{2, -8, 1, 3}},
		{"suppress", Sampler{Suppress: []int{1, 3}}, []float32{1, 2, 3, 4}, nil, []float32{1, ninf, 3, ninf}},
		{"min-new masks eos", Sampler{EOS: 2, MinNew: 2}, []float32{1, 2, 3}, []int64{0}, []float32{1, 2, ninf}},
		{"min-new met", Sampler{EOS: 2, MinNew: 2}, []float32{1, 2, 3}, []int64{0, 1}, []float32{1, 2, 3}},
	} {
		got := append([]float32(nil), c.in...)
		c.s.process(got, c.history)
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("%s: got %v want %v", c.name, got, c.want)
				break
			}
		}
	}
}

// TestWarpers: temperature, top-k (ties kept), top-p (smallest set reaching
// p, top token always kept).
func TestWarpers(t *testing.T) {
	ninf := float32(math.Inf(-1))
	s := Sampler{Temperature: 2, TopK: 2}
	got := []float32{2, 6, 4, 6}
	s.warp(got)
	if want := []float32{ninf, 3, ninf, 3}; !eq(got, want) {
		t.Errorf("top-k ties: got %v want %v", got, want)
	}
	// probs ∝ 1,2,4,8 (/15): p=0.7 drops the tail whose cumulative mass is <= 0.3 (1/15, 3/15).
	s = Sampler{TopP: 0.7}
	got = []float32{0, float32(math.Log(2)), float32(math.Log(4)), float32(math.Log(8))}
	s.warp(got)
	if !math.IsInf(float64(got[0]), -1) || !math.IsInf(float64(got[1]), -1) || math.IsInf(float64(got[2]), -1) || math.IsInf(float64(got[3]), -1) {
		t.Errorf("top-p: got %v", got)
	}
}

// TestGreedyAndDraw: argmax takes the first maximum; seeded sampling is
// reproducible and respects masked tokens.
func TestGreedyAndDraw(t *testing.T) {
	var g Sampler
	if got := g.Next([]float32{1, 3, 3, 2}, nil); got != 1 {
		t.Errorf("argmax tie-break: got %d want 1", got)
	}
	draw := func() []int {
		s := Sampler{DoSample: true, Seed: 7, Suppress: []int{0}}
		var out []int
		for range 50 {
			out = append(out, s.Next([]float32{9, 0, 0.5, 1}, nil))
		}
		return out
	}
	a, b := draw(), draw()
	for i := range a {
		if a[i] != b[i] {
			t.Fatal("seeded sampling not reproducible")
		}
		if a[i] == 0 {
			t.Fatal("drew a suppressed token")
		}
	}
}

func eq(a, b []float32) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return len(a) == len(b)
}
