// Package prosodycmd is ingot prosody: how each word of a known transcript
// is said — pitch (semitones above the line's median), loudness (dB above
// the line) and duration (relative to the line's pace), from a wav2vec2
// forced alignment. With --baseline (a plain rendering of the same line) it
// reports each word's change and whether --word came out stressed.
//
//	ingot prosody --wav take.wav --text "I never said that"
//	ingot prosody --wav stressed.wav --baseline plain.wav --text "I never said that" --word never
//
// The aligner is the ONNX export of facebook/wav2vec2-base-960h
// (tools/export/align_ref.py writes testdata/align/wav2vec2-base-960h.onnx);
// English only.
package prosodycmd

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/giraffesyo/ingot/audio"
	"github.com/giraffesyo/ingot/models/align"
)

type options struct {
	wav, text, baseline, word, model, vocab string
	json                                    bool
	th                                      align.Thresholds
}

// Command returns the prosody subcommand.
func Command() *cobra.Command {
	o := options{th: align.DefaultThresholds}
	cmd := &cobra.Command{
		Use:   "prosody",
		Short: "Measure per-word pitch, loudness and duration of a recording",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return run(o)
		},
	}
	f := cmd.Flags()
	f.StringVar(&o.wav, "wav", "", "recording to measure (WAV; required)")
	f.StringVar(&o.text, "text", "", "its transcript (required)")
	f.StringVar(&o.baseline, "baseline", "", "a plain rendering of the same line to compare against (WAV)")
	f.StringVar(&o.word, "word", "", "the word that should be stressed (with --baseline)")
	f.StringVar(&o.model, "model", "testdata/align/wav2vec2-base-960h.onnx", "aligner ONNX (tools/export/align_ref.py)")
	f.StringVar(&o.vocab, "vocab", "", "aligner vocab.json (default: facebook/wav2vec2-base-960h in the HF cache)")
	f.BoolVar(&o.json, "json", false, "print JSON instead of a table")
	f.Float64Var(&o.th.PitchST, "min-pitch", o.th.PitchST, "stressed: pitch rise in semitones")
	f.Float64Var(&o.th.LoudDB, "min-loud", o.th.LoudDB, "stressed: loudness rise in dB")
	f.Float64Var(&o.th.DurRatio, "min-dur", o.th.DurRatio, "stressed: duration growth factor")
	f.IntVar(&o.th.Need, "need", o.th.Need, "stressed: how many of the three cues must clear")
	_ = cmd.MarkFlagRequired("wav")
	_ = cmd.MarkFlagRequired("text")
	return cmd
}

func run(o options) error {
	th := o.th
	if o.vocab == "" {
		home, _ := os.UserHomeDir()
		m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--facebook--wav2vec2-base-960h/snapshots/*/vocab.json"))
		if len(m) == 0 {
			return fmt.Errorf("facebook/wav2vec2-base-960h not in the HF cache; pass --vocab")
		}
		o.vocab = m[0]
	}
	a, err := align.Load(o.model, o.vocab)
	if err != nil {
		return err
	}
	measure := func(path string) ([]align.WordProsody, error) {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		wav, rate, err := audio.ReadWAV(f)
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		words, err := a.Align(wav, rate, o.text)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		return align.Measure(wav, rate, words), nil
	}
	m, err := measure(o.wav)
	if err != nil {
		return err
	}
	var base []align.WordProsody
	if o.baseline != "" {
		if base, err = measure(o.baseline); err != nil {
			return err
		}
		if len(base) != len(m) {
			return fmt.Errorf("aligned %d words in --wav but %d in --baseline", len(m), len(base))
		}
	}
	target := -1
	if o.word != "" {
		words := make([]align.Word, len(m))
		for i, w := range m {
			words[i] = w.Word
		}
		if target = align.FindWord(words, o.word); target < 0 {
			return fmt.Errorf("%q is not a word of the transcript", o.word)
		}
	}

	if o.json {
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
		return enc.Encode(rows)
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
	return nil
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
