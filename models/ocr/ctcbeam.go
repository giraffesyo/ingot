package ocr

import (
	"math"
	"sort"
	"strings"
)

// CTC prefix beam search (Hannun et al. 2014): keeps the `width` most
// probable label prefixes per step, each with the total probability of the
// alignment paths ending in blank (pb) and in a non-blank (pnb). Paths that
// collapse to the same label are merged, so the search maximises label
// probability rather than path probability — the case where it beats greedy
// decoding is a label whose mass is spread over several alignments.
//
// Per step only characters with probability ≥ beamCharMin are expanded (at
// most beamCharMax of them, best first): PP-OCR's 6625-way softmax is
// near-one-hot at every frame, and expanding the tail buys nothing.
const (
	beamCharMin = 1e-3
	beamCharMax = 16
)

type ctcBeam struct {
	prefix  []int32 // label ids (never blank)
	pb, pnb float64
}

// ctcBeamDecode decodes [T,C] probabilities (blank = class 0) with beam width
// w (≥ 2; callers route w ≤ 1 to greedy), expanding only characters whose
// frame probability is ≥ charMin. Returns the label and its probability
// summed over the alignments the beam kept (exact when nothing was pruned).
func ctcBeamDecode(p []float32, T, C, w int, charMin float64) ([]int32, float64) {
	beams := []ctcBeam{{prefix: nil, pb: 1, pnb: 0}}
	next := make(map[string]*ctcBeam, 4*w)
	var cands []int32
	var candP []float64
	var keyBuf []byte
	key := func(prefix []int32) string {
		keyBuf = keyBuf[:0]
		for _, c := range prefix {
			keyBuf = append(keyBuf, byte(c), byte(c>>8), byte(c>>16))
		}
		return string(keyBuf)
	}
	get := func(prefix []int32) *ctcBeam {
		k := key(prefix)
		b := next[k]
		if b == nil {
			b = &ctcBeam{prefix: prefix}
			next[k] = b
		}
		return b
	}
	for t := 0; t < T; t++ {
		row := p[t*C : (t+1)*C]
		// Candidate characters for this frame, best first.
		cands, candP = cands[:0], candP[:0]
		for c := 1; c < C; c++ {
			v := float64(row[c])
			if v < charMin || v == 0 {
				continue
			}
			// Insertion into a short sorted list (descending).
			i := len(cands)
			if i == beamCharMax {
				if v <= candP[i-1] {
					continue
				}
				i--
			} else {
				cands = append(cands, 0)
				candP = append(candP, 0)
			}
			for i > 0 && candP[i-1] < v {
				cands[i], candP[i] = cands[i-1], candP[i-1]
				i--
			}
			cands[i], candP[i] = int32(c), v
		}
		pBlank := float64(row[0])
		for k := range next {
			delete(next, k)
		}
		for _, b := range beams {
			ptot := b.pb + b.pnb
			// Extend with blank: label unchanged.
			nb := get(b.prefix)
			nb.pb += ptot * pBlank
			last := int32(-1)
			if len(b.prefix) > 0 {
				last = b.prefix[len(b.prefix)-1]
			}
			for i, c := range cands {
				pc := candP[i]
				if c == last {
					// Repeated char: collapses unless a blank separated it.
					nb := get(b.prefix)
					nb.pnb += b.pnb * pc
					if b.pb > 0 {
						ext := get(appendLabel(b.prefix, c))
						ext.pnb += b.pb * pc
					}
				} else {
					ext := get(appendLabel(b.prefix, c))
					ext.pnb += ptot * pc
				}
			}
		}
		beams = beams[:0]
		for _, b := range next {
			beams = append(beams, *b)
		}
		sort.Slice(beams, func(i, j int) bool {
			pi, pj := beams[i].pb+beams[i].pnb, beams[j].pb+beams[j].pnb
			if pi != pj {
				return pi > pj
			}
			// Deterministic tie-break: shorter, then lexicographic.
			return lessPrefix(beams[i].prefix, beams[j].prefix)
		})
		if len(beams) > w {
			beams = beams[:w]
		}
	}
	best := beams[0]
	return best.prefix, best.pb + best.pnb
}

func appendLabel(prefix []int32, c int32) []int32 {
	out := make([]int32, len(prefix)+1)
	copy(out, prefix)
	out[len(prefix)] = c
	return out
}

func lessPrefix(a, b []int32) bool {
	if len(a) != len(b) {
		return len(a) < len(b)
	}
	for i := range a {
		if a[i] != b[i] {
			return a[i] < b[i]
		}
	}
	return false
}

// beamDecode runs the prefix beam search and renders the label through the
// dictionary. Confidence is the label probability's geometric mean per
// character (a probability in [0,1] on the same scale as the greedy path's
// mean max-probability; equal for a one-hot posterior).
func (r *Recognizer) beamDecode(p []float32, T, C int) (string, float64) {
	label, prob := ctcBeamDecode(p, T, C, r.BeamWidth, beamCharMin)
	var sb strings.Builder
	for _, c := range label {
		if int(c) < len(r.dict) {
			sb.WriteString(r.dict[c])
		}
	}
	conf := math.Pow(prob, 1/math.Max(1, float64(len(label))))
	return sb.String(), conf
}
