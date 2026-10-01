package align

import (
	"fmt"
	"strings"

	"github.com/giraffesyo/ingot/audio"
)

// ParseStress strips *word* stress markup from text: it returns the
// clean text and the indices (among its whitespace-separated words) of
// the marked ones. A marked word keeps any punctuation outside the
// asterisks ("*never*," → "never,").
func ParseStress(text string) (clean string, marked []int) {
	fields := strings.Fields(text)
	out := make([]string, len(fields))
	for i, f := range fields {
		a := strings.Index(f, "*")
		b := strings.LastIndex(f, "*")
		if a >= 0 && b > a+1 {
			out[i] = f[:a] + f[a+1:b] + f[b+1:]
			marked = append(marked, i)
			continue
		}
		out[i] = f
	}
	return strings.Join(out, " "), marked
}

// StressMarked aligns wav against the clean text and stresses the words
// at the given indices (as ParseStress returns them) with e, last first
// so earlier spans keep their times. It returns the new waveform.
func (a *Aligner) StressMarked(wav []float32, rate int, clean string, marked []int, e audio.Emphasis) ([]float32, error) {
	if len(marked) == 0 {
		return wav, nil
	}
	words, err := a.Align(wav, rate, clean)
	if err != nil {
		return nil, err
	}
	// Map text word index → aligned word: words with no alignable
	// character (digits, symbols) are skipped by the aligner.
	_, alignable := a.Tokens(clean)
	fields := strings.Fields(clean)
	index := make([]int, len(fields))
	j := 0
	for i, f := range fields {
		index[i] = -1
		if j < len(alignable) && alignable[j] == f {
			if j < len(words) {
				index[i] = j
			}
			j++
		}
	}
	for k := len(marked) - 1; k >= 0; k-- {
		i := marked[k]
		if i < 0 || i >= len(index) || index[i] < 0 {
			return nil, fmt.Errorf("align: marked word %d (%q) has no alignable letters", i, fields[min(max(i, 0), len(fields)-1)])
		}
		w := words[index[i]]
		wav = audio.Emphasize(wav, rate, w.Start, w.End, e)
	}
	return wav, nil
}
