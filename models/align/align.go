// Package align finds where each word of a known transcript sits in a
// recording: a wav2vec2 character-CTC model (an ONNX export run on the
// generic graph path; tools/export/align_ref.py) gives per-frame character
// log-probabilities, and CTC forced alignment (Viterbi over the transcript,
// as torchaudio.functional.forced_align) assigns frames to characters and
// so words. English (the 960h checkpoint's alphabet); 20 ms frames.
package align

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"unicode"

	"github.com/giraffesyo/ingot/audio"
	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
)

// FrameSeconds is the CTC frame period (the feature encoder's 320-sample
// stride at 16 kHz).
const FrameSeconds = 0.02

// SampleRate is the model's input rate.
const SampleRate = 16000

// Aligner holds the compiled CTC model and its alphabet.
type Aligner struct {
	sess  *graph.Session
	vocab map[string]int
	blank int
	sep   int // word separator "|"
}

// Load compiles the ONNX export and reads the checkpoint's vocab.json.
func Load(onnxPath, vocabPath string) (*Aligner, error) {
	m, err := onnx.DecodeFile(onnxPath)
	if err != nil {
		return nil, fmt.Errorf("align: %w", err)
	}
	g, err := graph.FromONNX(m)
	if err != nil {
		return nil, fmt.Errorf("align: %w", err)
	}
	s, err := graph.Compile(g)
	if err != nil {
		return nil, fmt.Errorf("align: %w", err)
	}
	raw, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("align: %w", err)
	}
	a := &Aligner{sess: s}
	if err := json.Unmarshal(raw, &a.vocab); err != nil {
		return nil, fmt.Errorf("align: %s: %w", vocabPath, err)
	}
	var ok1, ok2 bool
	a.blank, ok1 = a.vocab["<pad>"]
	a.sep, ok2 = a.vocab["|"]
	if !ok1 || !ok2 {
		return nil, fmt.Errorf("align: %s lacks <pad> or |", vocabPath)
	}
	return a, nil
}

// LogProbs returns the per-frame log-probabilities [frames][vocab] of wav
// (mono, any rate; resampled to 16 kHz and normalised to zero mean, unit
// variance as Wav2Vec2FeatureExtractor does).
func (a *Aligner) LogProbs(wav []float32, rate int) ([][]float32, error) {
	var x []float32
	if rate != SampleRate {
		x = audio.Resample(wav, rate, SampleRate)
	} else {
		x = append([]float32(nil), wav...)
	}
	if len(x) < 400 {
		return nil, fmt.Errorf("align: %d samples is too short", len(x))
	}
	var mean, sq float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(len(x))
	for _, v := range x {
		d := float64(v) - mean
		sq += d * d
	}
	inv := 1 / math.Sqrt(sq/float64(len(x))+1e-7)
	for i, v := range x {
		x[i] = float32((float64(v) - mean) * inv)
	}
	out, err := a.sess.Run(map[string]*tensor.Tensor{"input_values": tensor.FromF32(x, 1, len(x))})
	if err != nil {
		return nil, fmt.Errorf("align: %w", err)
	}
	lg := out["logits"]
	F, V := lg.Shape()[1], lg.Shape()[2]
	rows := make([][]float32, F)
	for f := range F {
		row := lg.F32()[f*V : (f+1)*V]
		m := float32(math.Inf(-1))
		for _, v := range row {
			m = max(m, v)
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v - m))
		}
		lse := float64(m) + math.Log(sum)
		rows[f] = make([]float32, V)
		for i, v := range row {
			rows[f][i] = float32(float64(v) - lse)
		}
	}
	a.sess.Release(out)
	return rows, nil
}

// Word is one transcript word's span in the recording.
type Word struct {
	Text string // as written in the transcript
	// FrameStart, FrameEnd are the CTC frames of the word's characters
	// ([first char's first frame, last char's last frame + 1]; torchaudio's
	// token spans) — CTC is peaky, so this is shorter than the word sounds.
	FrameStart, FrameEnd int
	// Start, End (seconds) are the word's acoustic span: from its first
	// character to the start of the next word's separator (or the next
	// word, or its last character at the end of the line) — the blanks
	// after its letters belong to it.
	Start, End float64
}

// Tokens maps a transcript to CTC targets: letters upper-cased, spaces to
// the word separator, characters outside the alphabet dropped. It also
// returns the words (original spelling) that kept at least one character.
func (a *Aligner) Tokens(text string) (targets []int, words []string) {
	for _, w := range strings.Fields(text) {
		var ids []int
		for _, r := range strings.ToUpper(w) {
			if id, ok := a.vocab[string(r)]; ok && !unicode.IsSpace(r) {
				ids = append(ids, id)
			}
		}
		if len(ids) == 0 {
			continue
		}
		if len(targets) > 0 {
			targets = append(targets, a.sep)
		}
		targets = append(targets, ids...)
		words = append(words, w)
	}
	return targets, words
}

// Align finds each word of text in wav.
func (a *Aligner) Align(wav []float32, rate int, text string) ([]Word, error) {
	lp, err := a.LogProbs(wav, rate)
	if err != nil {
		return nil, err
	}
	targets, words := a.Tokens(text)
	if len(targets) == 0 {
		return nil, fmt.Errorf("align: no alignable characters in %q", text)
	}
	path, err := ForcedAlign(lp, targets, a.blank)
	if err != nil {
		return nil, err
	}
	return wordSpans(path, targets, words, a.sep), nil
}

// ForcedAlign is CTC Viterbi alignment (torchaudio.functional.forced_align
// semantics): the most likely frame labelling that reads targets, blanks
// optional between labels and required between repeats. It returns, per
// frame, the index into targets that frame emits, or -1 for a blank.
func ForcedAlign(logp [][]float32, targets []int, blank int) ([]int, error) {
	T, L := len(logp), len(targets)
	S := 2*L + 1 // blank, t0, blank, t1, …, blank
	repeats := 0
	for i := 1; i < L; i++ {
		if targets[i] == targets[i-1] {
			repeats++
		}
	}
	if T < L+repeats {
		return nil, fmt.Errorf("align: %d frames cannot hold %d labels", T, L+repeats)
	}
	label := func(s int) int {
		if s%2 == 0 {
			return blank
		}
		return targets[s/2]
	}
	ninf := math.Inf(-1)
	prev, cur := make([]float64, S), make([]float64, S)
	back := make([][]int8, T) // 0: stay, 1: from s-1, 2: from s-2
	for s := range prev {
		prev[s] = ninf
	}
	prev[0] = float64(logp[0][blank])
	prev[1] = float64(logp[0][label(1)])
	back[0] = make([]int8, S)
	for t := 1; t < T; t++ {
		back[t] = make([]int8, S)
		for s := range S {
			best, from := prev[s], int8(0)
			if s >= 1 && prev[s-1] > best {
				best, from = prev[s-1], 1
			}
			if s >= 2 && s%2 == 1 && label(s) != label(s-2) && prev[s-2] > best {
				best, from = prev[s-2], 2
			}
			if math.IsInf(best, -1) {
				cur[s] = ninf
				continue
			}
			cur[s] = best + float64(logp[t][label(s)])
			back[t][s] = from
		}
		prev, cur = cur, prev
	}
	s := S - 1
	if S >= 2 && prev[S-2] > prev[S-1] {
		s = S - 2
	}
	if math.IsInf(prev[s], -1) {
		return nil, fmt.Errorf("align: no alignment")
	}
	path := make([]int, T)
	for t := T - 1; t >= 0; t-- {
		if s%2 == 1 {
			path[t] = s / 2
		} else {
			path[t] = -1
		}
		s -= int(back[t][s])
	}
	return path, nil
}

// wordSpans turns a frame path (target indices or -1) into word spans
// (see Word).
func wordSpans(path, targets []int, words []string, sep int) []Word {
	first, last := make([]int, len(targets)), make([]int, len(targets))
	for i := range first {
		first[i], last[i] = -1, -1
	}
	for t, ti := range path {
		if ti < 0 {
			continue
		}
		if first[ti] < 0 {
			first[ti] = t
		}
		last[ti] = t
	}
	// Word w covers targets [lo[w], hi[w]); the separator after it, if any,
	// is target hi[w].
	var out []Word
	w, lo := 0, 0
	for i := 0; i <= len(targets); i++ {
		if i < len(targets) && targets[i] != sep {
			continue
		}
		start, end := -1, -1
		for j := lo; j < i; j++ {
			if first[j] >= 0 {
				if start < 0 {
					start = first[j]
				}
				end = last[j]
			}
		}
		if w < len(words) && start >= 0 {
			acoustic := end + 1
			if i < len(targets) { // up to the separator, else the next word
				if first[i] >= 0 {
					acoustic = first[i]
				} else if i+1 < len(targets) && first[i+1] >= 0 {
					acoustic = first[i+1]
				}
			}
			out = append(out, Word{Text: words[w], FrameStart: start, FrameEnd: end + 1,
				Start: float64(start) * FrameSeconds, End: float64(max(acoustic, end+1)) * FrameSeconds})
		}
		w++
		lo = i + 1
	}
	return out
}
