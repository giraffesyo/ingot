package ocr

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// ctcLabelProb is the exact CTC probability of label under [T,C] posteriors
// (forward algorithm over the blank-interleaved label), the oracle for the
// beam search.
func ctcLabelProb(p []float32, T, C int, label []int32) float64 {
	L := 2*len(label) + 1
	ext := make([]int32, L)
	for i := range label {
		ext[2*i+1] = label[i]
	}
	alpha := make([]float64, L)
	alpha[0] = float64(p[0])
	if L > 1 {
		alpha[1] = float64(p[ext[1]])
	}
	for t := 1; t < T; t++ {
		row := p[t*C : (t+1)*C]
		nxt := make([]float64, L)
		for s := 0; s < L; s++ {
			a := alpha[s]
			if s > 0 {
				a += alpha[s-1]
			}
			if s > 1 && ext[s] != 0 && ext[s] != ext[s-2] {
				a += alpha[s-2]
			}
			nxt[s] = a * float64(row[ext[s]])
		}
		alpha = nxt
	}
	pr := alpha[L-1]
	if L > 1 {
		pr += alpha[L-2]
	}
	return pr
}

func greedyLabel(p []float32, T, C int) []int32 {
	var out []int32
	prev := -1
	for t := 0; t < T; t++ {
		row := p[t*C : (t+1)*C]
		best := 0
		for c := 1; c < C; c++ {
			if row[c] > row[best] {
				best = c
			}
		}
		if best != 0 && best != prev {
			out = append(out, int32(best))
		}
		prev = best
	}
	return out
}

// The textbook case: blank is the per-frame argmax everywhere, yet "a" (all
// alignments a-, -a, aa) carries more mass than the empty label.
func TestCTCBeamBeatsGreedy(t *testing.T) {
	p := []float32{0.6, 0.4, 0.6, 0.4}
	label, prob := ctcBeamDecode(p, 2, 2, 4, beamCharMin)
	if len(label) != 1 || label[0] != 1 {
		t.Fatalf("beam label %v, want [1]", label)
	}
	if math.Abs(prob-0.64) > 1e-6 { // float32 posteriors
		t.Fatalf("prob %v, want 0.64", prob)
	}
	if g := greedyLabel(p, 2, 2); len(g) != 0 {
		t.Fatalf("greedy label %v, want empty", g)
	}
}

// Brute force over all labels of a tiny alphabet: with a wide beam the
// search must return the maximum-probability label with its exact
// probability; a narrow beam may prune the best label (and can even lose to
// greedy), and the mass it reports is a lower bound on its label's exact
// probability (alignments through pruned prefixes are lost, never invented).
func TestCTCBeamOracle(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	const T, C = 5, 3
	// Every label over {1,2} of length ≤ T.
	var labels [][]int32
	var gen func(prefix []int32)
	gen = func(prefix []int32) {
		labels = append(labels, append([]int32{}, prefix...))
		if len(prefix) == T {
			return
		}
		for c := int32(1); c < C; c++ {
			gen(append(prefix, c))
		}
	}
	gen(nil)
	for trial := 0; trial < 300; trial++ {
		p := make([]float32, T*C)
		for t := 0; t < T; t++ {
			var sum float32
			for c := 0; c < C; c++ {
				v := float32(math.Pow(rng.Float64(), 2)) // skewed, some near-one-hot rows
				p[t*C+c] = v
				sum += v
			}
			for c := 0; c < C; c++ {
				p[t*C+c] /= sum
			}
		}
		var bestLabel []int32
		bestP := -1.0
		for _, l := range labels {
			if pr := ctcLabelProb(p, T, C, l); pr > bestP+1e-12 {
				bestP, bestLabel = pr, l
			}
		}
		got, gotP := ctcBeamDecode(p, T, C, 64, 0)
		if math.Abs(gotP-ctcLabelProb(p, T, C, got)) > 1e-9 {
			t.Fatalf("trial %d: beam prob %v ≠ oracle prob %v for %v", trial, gotP, ctcLabelProb(p, T, C, got), got)
		}
		if math.Abs(gotP-bestP) > 1e-9 {
			t.Fatalf("trial %d: beam picked %v (%v), oracle %v (%v)", trial, got, gotP, bestLabel, bestP)
		}
		narrow, narrowP := ctcBeamDecode(p, T, C, 2, 0)
		if np := ctcLabelProb(p, T, C, narrow); narrowP > np+1e-9 || narrowP <= 0 {
			t.Fatalf("trial %d: width-2 beam reports %v for %v, exact %v", trial, narrowP, narrow, np)
		}
	}
}

// Wide-alphabet path: only a handful of classes clear beamCharMin per frame.
func TestCTCBeamWideAlphabet(t *testing.T) {
	const T, C = 6, 300
	p := make([]float32, T*C)
	for t := 0; t < T; t++ {
		p[t*C+0] = 0.05
		p[t*C+7] = 0.9
		p[t*C+9] = 0.05
	}
	label, _ := ctcBeamDecode(p, T, C, 8, beamCharMin)
	if len(label) != 1 || label[0] != 7 {
		t.Fatalf("label %v, want [7]", label)
	}
}

// BenchmarkCTCDecode: greedy vs beam on a PP-OCR-shaped posterior (T=40,
// C=6625, near-one-hot with a runner-up per frame) — the per-crop decode
// cost on top of the ~3-4 ms forward.
func BenchmarkCTCDecode(b *testing.B) {
	const T, C = 40, 6625
	rng := rand.New(rand.NewPCG(5, 6))
	p := make([]float32, T*C)
	for t := 0; t < T; t++ {
		row := p[t*C : (t+1)*C]
		for c := range row {
			row[c] = 1e-6
		}
		best := rng.IntN(C)
		if t%3 == 0 {
			best = 0
		}
		row[best] = 0.9
		row[rng.IntN(C)] += 0.08
	}
	r := &Recognizer{dict: make([]string, C)}
	for i := range r.dict {
		r.dict[i] = string(rune('a' + i%26))
	}
	b.Run("greedy", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			r.ctcDecode(p, T, C)
		}
	})
	for _, w := range []int{4, 16} {
		b.Run(fmt.Sprintf("beam=%d", w), func(b *testing.B) {
			r.BeamWidth = w
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				r.beamDecode(p, T, C)
			}
		})
	}
}
