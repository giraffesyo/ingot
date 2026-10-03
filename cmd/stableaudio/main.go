// Command stableaudio makes a sound from a text prompt with Stable Audio 3,
// running entirely on the ingot runtime (pure Go, CPU).
//
//	stableaudio -prompt "TrackType: SFX, a wooden door creaking open" -seconds 3 -seed 7 -out door.wav
//
// The graphs come from stabilityai/stable-audio-3-optimized in the Hugging
// Face cache (-model for another snapshot directory). The weights are under
// the Stability AI Community License.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/giraffesyo/ingot/models/stableaudio"
)

func main() {
	prompt := flag.String("prompt", "", "what to generate")
	seconds := flag.Float64("seconds", 5, "length in seconds")
	seed := flag.Uint64("seed", 0, "noise seed")
	steps := flag.Int("steps", 8, "sampler steps")
	variant := flag.String("variant", "sm-sfx", "sm-sfx or sm-music")
	model := flag.String("model", "", "snapshot directory (default: the Hugging Face cache)")
	out := flag.String("out", "out.wav", "output WAV (44.1 kHz stereo, 16-bit)")
	flag.Parse()
	if *prompt == "" {
		fmt.Fprintln(os.Stderr, "stableaudio: -prompt is required")
		os.Exit(2)
	}
	dir := *model
	if dir == "" {
		var err error
		if dir, err = stableaudio.Snapshot(); err != nil {
			fail(err)
		}
	}
	t0 := time.Now()
	m, err := stableaudio.Load(dir, *variant)
	if err != nil {
		fail(err)
	}
	fmt.Fprintf(os.Stderr, "stableaudio: loaded %s in %.1f s\n", *variant, time.Since(t0).Seconds())
	t0 = time.Now()
	l, r, err := m.Generate(*prompt, stableaudio.Options{Seconds: *seconds, Steps: *steps, Seed: *seed})
	if err != nil {
		fail(err)
	}
	fmt.Fprintf(os.Stderr, "stableaudio: %.1f s of audio in %.2f s\n", *seconds, time.Since(t0).Seconds())
	if err := writeStereo(*out, l, r); err != nil {
		fail(err)
	}
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

func fail(err error) {
	fmt.Fprintln(os.Stderr, "stableaudio:", err)
	os.Exit(1)
}
