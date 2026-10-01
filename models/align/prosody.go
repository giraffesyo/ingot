package align

import (
	"encoding/json"
	"math"
	"sort"
	"strings"
	"unicode"

	"github.com/giraffesyo/ingot/audio"
)

// Prosody analysis frames: 10 ms hop; YIN over 40 ms (two periods of a
// 60 Hz voice — longer windows blur speech's moving pitch and read voiced
// syllables as aperiodic), RMS over 25 ms. Pitch is searched in [60, 500]
// Hz.
const (
	prosodyHop      = 160 // samples at 16 kHz
	pitchFrame      = 640
	rmsFrame        = 400
	pitchMin        = 60.0
	pitchMax        = 500.0
	voicedAperiodic = 0.25 // YIN aperiodicity below which a frame is voiced
	silenceDB       = 35.0 // frames this far below the loudest are silence
)

// WordProsody is one word's prominence relative to its utterance. NaN
// cues (an unvoiced word's pitch) marshal to JSON null.
type WordProsody struct {
	Word
	// PitchST is the word's peak voiced f0 (90th percentile of its voiced
	// frames) in semitones above the utterance's median voiced f0; NaN when
	// the word has no voiced frame.
	PitchST float64
	// LoudDB is the word's energy (mean power over its frames) in dB above
	// the utterance's speech frames.
	LoudDB float64
	// DurRatio is the word's duration over the utterance's mean duration
	// for a word of its letter count (1 = the line's own pace).
	DurRatio float64
	// Voiced is the fraction of the word's frames that are voiced.
	Voiced float64
}

// Measure computes each aligned word's pitch, loudness and duration cues
// in wav (any rate; analysed at 16 kHz).
func Measure(wav []float32, rate int, words []Word) []WordProsody {
	x := wav
	if rate != SampleRate {
		x = audio.Resample(wav, rate, SampleRate)
	}
	pitch := audio.Yin(x, SampleRate, pitchMin, pitchMax, pitchFrame, prosodyHop)
	rms := audio.FrameRMS(x, rmsFrame, prosodyHop)
	n := min(len(pitch), len(rms))
	db := make([]float64, n)
	maxDB := math.Inf(-1)
	for i := range n {
		db[i] = 20 * math.Log10(rms[i]+1e-12)
		maxDB = math.Max(maxDB, db[i])
	}
	voiced := func(i int) bool {
		return i < n && pitch[i].Aperiodicity < voicedAperiodic && db[i] > maxDB-silenceDB
	}
	// Analysis frames of a word: its span in 10 ms frames.
	span := func(w Word) (int, int) {
		a := int(math.Round(w.Start * SampleRate / prosodyHop))
		b := int(math.Round(w.End * SampleRate / prosodyHop))
		return max(0, min(a, n)), max(0, min(b, n))
	}
	// Utterance references: median voiced f0, mean power over word
	// frames, seconds per letter.
	var f0s []float64
	var pow float64
	var frames, letters int
	var secs float64
	for _, w := range words {
		a, b := span(w)
		for i := a; i < b; i++ {
			if voiced(i) {
				f0s = append(f0s, pitch[i].F0)
			}
			pow += rms[i] * rms[i]
			frames++
		}
		letters += letterCount(w.Text)
		secs += w.End - w.Start
	}
	medF0 := median(f0s)
	uttDB := 10 * math.Log10(pow/math.Max(1, float64(frames))+1e-24)
	secPerLetter := secs / math.Max(1, float64(letters))

	out := make([]WordProsody, len(words))
	for k, w := range words {
		a, b := span(w)
		var vf []float64
		var p float64
		for i := a; i < b; i++ {
			if voiced(i) {
				vf = append(vf, pitch[i].F0)
			}
			p += rms[i] * rms[i]
		}
		wp := WordProsody{Word: w, PitchST: math.NaN()}
		if len(vf) > 0 && medF0 > 0 {
			wp.PitchST = 12 * math.Log2(percentile(vf, 0.9)/medF0)
		}
		if b > a {
			wp.LoudDB = 10*math.Log10(p/float64(b-a)+1e-24) - uttDB
			wp.Voiced = float64(len(vf)) / float64(b-a)
		}
		if l := letterCount(w.Text); l > 0 && secPerLetter > 0 {
			wp.DurRatio = (w.End - w.Start) / (float64(l) * secPerLetter)
		}
		out[k] = wp
	}
	return out
}

func (w WordProsody) MarshalJSON() ([]byte, error) {
	type plain WordProsody
	return json.Marshal(struct {
		plain
		PitchST *float64
	}{plain(w), nanNil(w.PitchST)})
}

func (c Contrast) MarshalJSON() ([]byte, error) {
	type plain Contrast
	return json.Marshal(struct {
		plain
		PitchST *float64
	}{plain(c), nanNil(c.PitchST)})
}

func nanNil(v float64) *float64 {
	if math.IsNaN(v) {
		return nil
	}
	return &v
}

// Thresholds decide whether a rendering stressed a word compared with a
// plain rendering of the same line.
type Thresholds struct {
	PitchST  float64 // rise in PitchST
	LoudDB   float64 // rise in LoudDB
	DurRatio float64 // growth factor of DurRatio
	Need     int     // how many of the three cues must clear
}

// DefaultThresholds: +1 semitone, +1.5 dB, ×1.15 — two of three. A
// heuristic for "audibly stressed", to be tuned against listening.
var DefaultThresholds = Thresholds{PitchST: 1, LoudDB: 1.5, DurRatio: 1.15, Need: 2}

// Contrast is a word's change from a plain rendering to a stressed one.
type Contrast struct {
	PitchST, LoudDB, DurRatio float64 // deltas (DurRatio: a factor)
	Cues                      int     // how many cleared the thresholds
	Stressed                  bool
}

// Compare contrasts word i's prosody in a stressed rendering with the
// same word in a plain rendering.
func Compare(plain, stressed WordProsody, th Thresholds) Contrast {
	c := Contrast{PitchST: stressed.PitchST - plain.PitchST, LoudDB: stressed.LoudDB - plain.LoudDB}
	if plain.DurRatio > 0 {
		c.DurRatio = stressed.DurRatio / plain.DurRatio
	}
	if c.PitchST >= th.PitchST { // NaN (unvoiced) never clears
		c.Cues++
	}
	if c.LoudDB >= th.LoudDB {
		c.Cues++
	}
	if c.DurRatio >= th.DurRatio {
		c.Cues++
	}
	c.Stressed = c.Cues >= th.Need
	return c
}

func letterCount(s string) int {
	n := 0
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			n++
		}
	}
	return n
}

func median(v []float64) float64 { return percentile(v, 0.5) }

// percentile is the q-quantile of v (linear interpolation), 0 if empty.
func percentile(v []float64, q float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	pos := q * float64(len(s)-1)
	i := int(pos)
	if i+1 >= len(s) {
		return s[len(s)-1]
	}
	return s[i] + (pos-float64(i))*(s[i+1]-s[i])
}

// FindWord returns the index of the first aligned word whose letters
// match target (case and punctuation ignored), or -1.
func FindWord(words []Word, target string) int {
	norm := func(s string) string {
		var b strings.Builder
		for _, r := range strings.ToLower(s) {
			if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '\'' {
				b.WriteRune(r)
			}
		}
		return b.String()
	}
	t := norm(target)
	for i, w := range words {
		if norm(w.Text) == t {
			return i
		}
	}
	return -1
}
