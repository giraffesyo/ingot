// Package stableaudiocmd is ingot stableaudio: a sound from a text prompt
// with Stable Audio 3, running entirely on the ingot runtime (pure Go, CPU).
//
//	ingot stableaudio --prompt "TrackType: SFX, a wooden door creaking open" --seconds 3 --seed 7 --out door.wav
//
// The graphs come from stabilityai/stable-audio-3-optimized in the Hugging
// Face cache (--model for another snapshot directory). The weights are under
// the Stability AI Community License.
package stableaudiocmd

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/spf13/cobra"

	"github.com/giraffesyo/ingot/models/stableaudio"
)

// Command returns the stableaudio subcommand.
func Command() *cobra.Command {
	var (
		prompt, variant, model, out string
		seconds                     float64
		seed                        uint64
		steps                       int
	)
	cmd := &cobra.Command{
		Use:   "stableaudio",
		Short: "Generate a sound from a text prompt (Stable Audio 3)",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return run(prompt, variant, model, out, stableaudio.Options{Seconds: seconds, Steps: steps, Seed: seed})
		},
	}
	f := cmd.Flags()
	f.StringVar(&prompt, "prompt", "", "what to generate (required)")
	f.Float64Var(&seconds, "seconds", 5, "length in seconds")
	f.Uint64Var(&seed, "seed", 0, "noise seed")
	f.IntVar(&steps, "steps", 8, "sampler steps")
	f.StringVar(&variant, "variant", "sm-sfx", "sm-sfx or sm-music")
	f.StringVar(&model, "model", "", "snapshot directory (default: the Hugging Face cache)")
	f.StringVar(&out, "out", "out.wav", "output WAV (44.1 kHz stereo, 16-bit)")
	_ = cmd.MarkFlagRequired("prompt")
	return cmd
}

func run(prompt, variant, dir, out string, o stableaudio.Options) error {
	if dir == "" {
		var err error
		if dir, err = stableaudio.Snapshot(); err != nil {
			return err
		}
	}
	t0 := time.Now()
	m, err := stableaudio.Load(dir, variant)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "stableaudio: loaded %s in %.1f s\n", variant, time.Since(t0).Seconds())
	t0 = time.Now()
	l, r, err := m.Generate(prompt, o)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "stableaudio: %.1f s of audio in %.2f s\n", o.Seconds, time.Since(t0).Seconds())
	return writeStereo(out, l, r)
}

// writeStereo writes 16-bit stereo PCM.
func writeStereo(path string, l, r []float32) error {
	b := make([]byte, 44, 44+4*len(l))
	copy(b, "RIFF")
	binary.LittleEndian.PutUint32(b[4:], uint32(36+4*len(l)))
	copy(b[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(b[16:], 16)
	binary.LittleEndian.PutUint16(b[20:], 1)
	binary.LittleEndian.PutUint16(b[22:], 2)
	binary.LittleEndian.PutUint32(b[24:], stableaudio.SampleRate)
	binary.LittleEndian.PutUint32(b[28:], stableaudio.SampleRate*4)
	binary.LittleEndian.PutUint16(b[32:], 4)
	binary.LittleEndian.PutUint16(b[34:], 16)
	copy(b[36:], "data")
	binary.LittleEndian.PutUint32(b[40:], uint32(4*len(l)))
	q := func(v float32) uint16 {
		return uint16(int16(math.Round(float64(max(-1, min(1, v))) * 32767)))
	}
	for i := range l {
		b = binary.LittleEndian.AppendUint16(b, q(l[i]))
		b = binary.LittleEndian.AppendUint16(b, q(r[i]))
	}
	return os.WriteFile(path, b, 0o644)
}
