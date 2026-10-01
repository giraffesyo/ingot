package align

import (
	"math"
	"testing"
)

// synthWords renders "words" as harmonic tones separated by short
// silences: each word's f0, amplitude and length are chosen, so its
// expected cues follow from the construction.
func synthWords(specs []struct{ f0, amp, sec float64 }) ([]float32, []Word) {
	const gap = 0.12
	var x []float32
	var words []Word
	t := gap
	x = append(x, make([]float32, int(gap*SampleRate))...)
	for i, s := range specs {
		n := int(s.sec * SampleRate)
		for k := range n {
			ph := 2 * math.Pi * s.f0 * float64(k) / SampleRate
			x = append(x, float32(s.amp*(math.Sin(ph)+0.4*math.Sin(2*ph)+0.2*math.Sin(3*ph))))
		}
		words = append(words, Word{Text: [...]string{"abcd", "efgh", "ijkl", "mnop"}[i], Start: t, End: t + s.sec})
		t += s.sec
		x = append(x, make([]float32, int(gap*SampleRate))...)
		t += gap
	}
	return x, words
}

// TestMeasureSynthetic: three equal words around a fourth with known
// differences — pitch in semitones, level in dB and length — come back
// as constructed.
func TestMeasureSynthetic(t *testing.T) {
	base := struct{ f0, amp, sec float64 }{150, 0.2, 0.30}
	stressed := struct{ f0, amp, sec float64 }{150 * math.Pow(2, 3.0/12), 0.2 * math.Pow(10, 4.0/20), 0.45}
	x, words := synthWords([]struct{ f0, amp, sec float64 }{base, stressed, base, base})
	m := Measure(x, SampleRate, words)
	// Utterance references include the stressed word itself.
	if got := m[1].PitchST; math.Abs(got-3) > 0.15 {
		t.Errorf("pitch: %.3f st, want 3", got)
	}
	if got := m[0].PitchST; math.Abs(got) > 0.15 {
		t.Errorf("plain word pitch %.3f st, want 0", got)
	}
	// Mean power over word frames: (3·0.3·1 + 0.45·10^0.4)/(0.9+0.45) of the
	// plain level.
	g := math.Pow(10, 0.4)
	utt := (0.9 + 0.45*g) / 1.35
	if want, got := 10*math.Log10(g/utt), m[1].LoudDB; math.Abs(got-want) > 0.3 {
		t.Errorf("loudness %.3f dB, want %.3f", got, want)
	}
	if want, got := 0.45/(1.35/4), m[1].DurRatio; math.Abs(got-want) > 0.02 {
		t.Errorf("duration ratio %.3f, want %.3f", got, want)
	}
	// Contrast with the plain rendering of the same "line".
	px, pw := synthWords([]struct{ f0, amp, sec float64 }{base, base, base, base})
	plain := Measure(px, SampleRate, pw)
	c := Compare(plain[1], m[1], DefaultThresholds)
	if !c.Stressed || c.Cues != 3 {
		t.Errorf("contrast %+v: want all three cues", c)
	}
	if c := Compare(plain[0], m[0], DefaultThresholds); c.Stressed {
		t.Errorf("unchanged word judged stressed: %+v", c)
	}
}

func TestFindWord(t *testing.T) {
	w := []Word{{Text: "I"}, {Text: "never,"}, {Text: "said"}, {Text: "that."}}
	if FindWord(w, "Never") != 1 || FindWord(w, "that") != 3 || FindWord(w, "nope") != -1 {
		t.Error("FindWord")
	}
}
