// Command prosody measures how each word of a known transcript is said:
// pitch (semitones above the line's median), loudness (dB above the line)
// and duration (relative to the line's pace), from a wav2vec2 forced
// alignment. With -baseline (a plain rendering of the same line) it reports
// each word's change and whether -word came out stressed.
//
//	prosody -wav take.wav -text "I never said that"
//	prosody -wav stressed.wav -baseline plain.wav -text "I never said that" -word never
//
// The aligner is the ONNX export of facebook/wav2vec2-base-960h
// (tools/export/align_ref.py writes testdata/align/wav2vec2-base-960h.onnx);
// English only.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/giraffesyo/ingot/audio"
	"github.com/giraffesyo/ingot/models/align"
)

func main() {
	wavPath := flag.String("wav", "", "recording to measure (WAV)")
	text := flag.String("text", "", "its transcript")
	baseline := flag.String("baseline", "", "a plain rendering of the same line to compare against (WAV)")
	word := flag.String("word", "", "the word that should be stressed (with -baseline)")
	model := flag.String("model", "testdata/align/wav2vec2-base-960h.onnx", "aligner ONNX (tools/export/align_ref.py)")
	vocab := flag.String("vocab", "", "aligner vocab.json (default: facebook/wav2vec2-base-960h in the HF cache)")
	asJSON := flag.Bool("json", false, "print JSON instead of a table")
	th := align.DefaultThresholds
	flag.Float64Var(&th.PitchST, "min-pitch", th.PitchST, "stressed: pitch rise in semitones")
	flag.Float64Var(&th.LoudDB, "min-loud", th.LoudDB, "stressed: loudness rise in dB")
	flag.Float64Var(&th.DurRatio, "min-dur", th.DurRatio, "stressed: duration growth factor")
	flag.IntVar(&th.Need, "need", th.Need, "stressed: how many of the three cues must clear")
	flag.Parse()
	if *wavPath == "" || *text == "" {
		fmt.Fprintln(os.Stderr, "prosody: -wav and -text are required")
		flag.Usage()
		os.Exit(2)
	}
	if *vocab == "" {
		home, _ := os.UserHomeDir()
		m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--facebook--wav2vec2-base-960h/snapshots/*/vocab.json"))
		if len(m) == 0 {
			fail(fmt.Errorf("facebook/wav2vec2-base-960h not in the HF cache; pass -vocab"))
		}
		*vocab = m[0]
	}
	a, err := align.Load(*model, *vocab)
	if err != nil {
		fail(err)
	}
	measure := func(path string) []align.WordProsody {
		f, err := os.Open(path)
		if err != nil {
			fail(err)
		}
		wav, rate, err := audio.ReadWAV(f)
		f.Close()
		if err != nil {
			fail(fmt.Errorf("%s: %w", path, err))
		}
		words, err := a.Align(wav, rate, *text)
		if err != nil {
			fail(fmt.Errorf("%s: %w", path, err))
		}
		return align.Measure(wav, rate, words)
	}
	m := measure(*wavPath)
	var base []align.WordProsody
	if *baseline != "" {
		base = measure(*baseline)
		if len(base) != len(m) {
			fail(fmt.Errorf("aligned %d words in -wav but %d in -baseline", len(m), len(base)))
		}
	}
	target := -1
	if *word != "" {
		words := make([]align.Word, len(m))
		for i, w := range m {
			words[i] = w.Word
		}
		if target = align.FindWord(words, *word); target < 0 {
			fail(fmt.Errorf("%q is not a word of the transcript", *word))
		}
	}

	if *asJSON {
		// A named field: an embedded WordProsody would promote its
		// MarshalJSON over the whole row.
		type row struct {
			Word     align.WordProsody
			Contrast *align.Contrast `json:",omitempty"`
			Target   bool            `json:",omitempty"`
		}
		var rows []row
		for i, w := range m {
			r := row{Word: w, Target: i == target}
			if base != nil {
				c := align.Compare(base[i], w, th)
				r.Contrast = &c
			}
			rows = append(rows, r)
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", " ")
		if err := enc.Encode(rows); err != nil {
			fail(err)
		}
		return
	}
	fmt.Printf("%-14s %13s %8s %8s %6s %6s", "word", "time (s)", "pitch", "loud", "dur", "voiced")
	if base != nil {
		fmt.Printf("   %8s %8s %6s  %s", "Δpitch", "Δloud", "×dur", "cues")
	}
	fmt.Println()
	for i, w := range m {
		mark := " "
		if i == target {
			mark = "*"
		}
		fmt.Printf("%s%-13s %5.2f-%-6.2f %+7.2fst %+6.1fdB %6.2f %5.0f%%", mark, trim(w.Text, 13), w.Start, w.End,
			nanZero(w.PitchST), w.LoudDB, w.DurRatio, 100*w.Voiced)
		if base != nil {
			c := align.Compare(base[i], w, th)
			fmt.Printf("   %+7.2fst %+6.1fdB %6.2f  %d/3", nanZero(c.PitchST), c.LoudDB, c.DurRatio, c.Cues)
		}
		fmt.Println()
	}
	if base != nil && target >= 0 {
		c := align.Compare(base[target], m[target], th)
		verdict := "NOT stressed"
		if c.Stressed {
			verdict = "stressed"
		}
		fmt.Printf("\n%q: %s (%d of 3 cues; need %d: pitch ≥ +%.1f st, loudness ≥ +%.1f dB, duration ≥ ×%.2f)\n",
			m[target].Text, verdict, c.Cues, th.Need, th.PitchST, th.LoudDB, th.DurRatio)
	}
}

func nanZero(v float64) float64 {
	if math.IsNaN(v) {
		return 0
	}
	return v
}

func trim(s string, n int) string {
	if len([]rune(s)) <= n {
		return s
	}
	return string([]rune(s)[:n-1]) + "…"
}

func fail(err error) {
	msg := err.Error()
	if !strings.HasPrefix(msg, "prosody: ") {
		msg = "prosody: " + msg
	}
	fmt.Fprintln(os.Stderr, msg)
	os.Exit(1)
}
