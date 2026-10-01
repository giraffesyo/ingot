// Package generate holds the token-selection half of autoregressive
// decoding, independent of any model: Hugging Face generate()'s logits
// pipeline (repetition penalty, suppressed / minimum-length-masked tokens,
// temperature, top-k, top-p) followed by argmax or a seeded categorical
// draw. Model drivers own the forward passes; they hand each step's logits
// and the tokens generated so far to a Sampler.
package generate

import (
	"math"
	"math/rand/v2"
	"slices"
)

// Sampler selects the next token from a logits row. The zero value is
// greedy argmax with no processing.
type Sampler struct {
	// DoSample draws from the processed distribution; false takes argmax.
	DoSample bool
	// Temperature divides the logits when sampling (0 or 1: unchanged).
	Temperature float32
	// TopK keeps the K highest logits when sampling (0: all). Ties at the
	// K-th value are kept, as in HF TopKLogitsWarper.
	TopK int
	// TopP keeps the smallest set of tokens whose probability mass reaches
	// TopP when sampling (0 or >= 1: all).
	TopP float32
	// RepetitionPenalty (HF CTRL form, 0 or 1: off) divides positive and
	// multiplies negative logits of every token already generated.
	RepetitionPenalty float32
	// Suppress lists token ids that may never be chosen.
	Suppress []int
	// EOS, with MinNew > 0, is masked until MinNew tokens exist.
	EOS    int
	MinNew int

	// RNG drives sampling; nil uses a PCG seeded with Seed.
	RNG  *rand.Rand
	Seed uint64

	probs []float64 // scratch
	order []int
	vals  []float32
}

// Next processes logits in place and returns the chosen token. history is
// every token generated so far in this sequence (the repetition-penalty
// and minimum-length context).
func (s *Sampler) Next(logits []float32, history []int64) int {
	s.process(logits, history)
	if !s.DoSample {
		return argmax(logits)
	}
	s.warp(logits)
	return s.draw(logits)
}

// process applies the logits processors (they run for greedy decoding too).
func (s *Sampler) process(logits []float32, history []int64) {
	if p := s.RepetitionPenalty; p != 0 && p != 1 {
		// Each distinct token is penalised once (HF gathers then scatters).
		for i, t := range history {
			if t < 0 || int(t) >= len(logits) || slices.Contains(history[:i], t) {
				continue
			}
			if v := logits[t]; v < 0 {
				logits[t] = v * p
			} else {
				logits[t] = v / p
			}
		}
	}
	ninf := float32(math.Inf(-1))
	if s.MinNew > 0 && len(history) < s.MinNew && s.EOS >= 0 && s.EOS < len(logits) {
		logits[s.EOS] = ninf
	}
	for _, t := range s.Suppress {
		if t >= 0 && t < len(logits) {
			logits[t] = ninf
		}
	}
}

// warp applies temperature, top-k, top-p (HF warper order).
func (s *Sampler) warp(logits []float32) {
	ninf := float32(math.Inf(-1))
	if t := s.Temperature; t != 0 && t != 1 {
		for i := range logits {
			logits[i] /= t
		}
	}
	if k := s.TopK; k > 0 && k < len(logits) {
		s.vals = append(s.vals[:0], logits...)
		slices.Sort(s.vals)
		kth := s.vals[len(s.vals)-k]
		for i, v := range logits {
			if v < kth {
				logits[i] = ninf
			}
		}
	}
	if p := s.TopP; p > 0 && p < 1 {
		// HF TopPLogitsWarper: sort ascending, drop the low tail whose
		// cumulative probability is <= 1-p (always keeping the top token).
		s.softmax(logits)
		order := s.sortedAsc(logits)
		var cum float64
		for _, idx := range order[:len(order)-1] {
			if cum += s.probs[idx]; cum > 1-float64(p) {
				break
			}
			logits[idx] = ninf
		}
	}
}

// Uniform returns the next uniform in [0, 1) of the sampler's stream — the
// draw Next would make — for a sampler running elsewhere (e.g. on a GPU).
func (s *Sampler) Uniform() float64 {
	if s.RNG == nil {
		s.RNG = rand.New(rand.NewPCG(s.Seed, s.Seed^0x9e3779b97f4a7c15))
	}
	return s.RNG.Float64()
}

// draw samples from softmax(logits).
func (s *Sampler) draw(logits []float32) int {
	s.softmax(logits)
	u := s.Uniform()
	var cum float64
	last := 0
	for i, p := range s.probs {
		if p == 0 {
			continue
		}
		cum += p
		last = i
		if u < cum {
			return i
		}
	}
	return last
}

func (s *Sampler) softmax(logits []float32) {
	if cap(s.probs) < len(logits) {
		s.probs = make([]float64, len(logits))
	}
	s.probs = s.probs[:len(logits)]
	m := math.Inf(-1)
	for _, v := range logits {
		m = math.Max(m, float64(v))
	}
	var sum float64
	for i, v := range logits {
		e := math.Exp(float64(v) - m)
		s.probs[i] = e
		sum += e
	}
	for i := range s.probs {
		s.probs[i] /= sum
	}
}

func (s *Sampler) sortedAsc(logits []float32) []int {
	s.order = s.order[:0]
	for i := range logits {
		s.order = append(s.order, i)
	}
	slices.SortStableFunc(s.order, func(a, b int) int {
		switch {
		case logits[a] < logits[b]:
			return -1
		case logits[a] > logits[b]:
			return 1
		}
		return 0
	})
	return s.order
}

// argmax returns the first index of the maximum (torch.argmax tie-break).
func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}
