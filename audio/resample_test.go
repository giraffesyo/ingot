package audio

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func tone(n, rate int, f, amp float64) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(amp * math.Sin(2*math.Pi*f*float64(i)/float64(rate)))
	}
	return x
}

// rms over the middle of y (away from the zero-padded edges).
func midRMS(y []float32) float64 {
	a, b := len(y)/4, 3*len(y)/4
	var s float64
	for _, v := range y[a:b] {
		s += float64(v) * float64(v)
	}
	return math.Sqrt(s / float64(b-a))
}

// TestResampleFilter: lengths match librosa's ceil(n·to/from); passband
// tones keep unit gain (within 0.02 dB); a tone above the output Nyquist
// is rejected by ≥ 90 dB; equal rates copy.
func TestResampleFilter(t *testing.T) {
	for _, c := range []struct{ from, to int }{{44100, 24000}, {48000, 24000}, {16000, 24000}, {22050, 24000}, {24000, 16000}} {
		t.Run(fmt.Sprint(c.from, "->", c.to), func(t *testing.T) {
			n := c.from / 2
			if got, want := len(Resample(make([]float32, n+3), c.from, c.to)), (int64(n+3)*int64(c.to)+int64(c.from)-1)/int64(c.from); int64(got) != want {
				t.Fatalf("length %d, want %d", got, want)
			}
			ny := 0.5 * float64(min(c.from, c.to))
			for _, f := range []float64{100, 1000, 0.8 * ny} {
				y := Resample(tone(n, c.from, f, 0.5), c.from, c.to)
				if db := 20 * math.Log10(midRMS(y)/(0.5/math.Sqrt2)); math.Abs(db) > 0.02 {
					t.Errorf("passband %g Hz: gain %.4f dB", f, db)
				}
			}
			if c.from > c.to { // a tone the output cannot represent
				y := Resample(tone(n, c.from, 1.1*ny, 0.5), c.from, c.to)
				if db := 20 * math.Log10(midRMS(y)/(0.5/math.Sqrt2)); db > -90 {
					t.Errorf("stopband %g Hz: %.1f dB", 1.1*ny, db)
				}
			}
		})
	}
	x := tone(100, 24000, 440, 1)
	if y := Resample(x, 24000, 24000); len(y) != 100 || y[37] != x[37] {
		t.Error("equal rates must copy")
	}
}

// TestResampleLibrosa: against librosa.resample (soxr_hq) on the speech-
// band test signal of tools/export/qwen3tts_ref.py --resample, at 16, 22.05,
// 44.1 and 48 kHz → 24 kHz: a different filter design, so the bar is SNR,
// measured away from the edges.
func TestResampleLibrosa(t *testing.T) {
	for _, sr := range []int{16000, 22050, 44100, 48000} {
		raw, err := os.ReadFile(filepath.Join("../testdata/qwen3tts", fmt.Sprintf("resample_%d.json", sr)))
		if err != nil {
			t.Skip("references not generated (tools/export/qwen3tts_ref.py --resample)")
		}
		var man struct {
			Inputs, Outputs []struct{ File string }
		}
		if err := json.Unmarshal(raw, &man); err != nil {
			t.Fatal(err)
		}
		load := func(f string) []float32 {
			b, err := os.ReadFile(filepath.Join("../testdata/qwen3tts", f))
			if err != nil {
				t.Fatal(err)
			}
			v := make([]float32, len(b)/4)
			for i := range v {
				v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
			}
			return v
		}
		x, want := load(man.Inputs[0].File), load(man.Outputs[0].File)
		got := Resample(x, sr, 24000)
		if len(got) != len(want) {
			t.Fatalf("%d Hz: %d samples, librosa %d", sr, len(got), len(want))
		}
		var sig, noise float64
		for i := len(want) / 20; i < len(want)-len(want)/20; i++ {
			d := float64(got[i] - want[i])
			sig += float64(want[i]) * float64(want[i])
			noise += d * d
		}
		snr := 10 * math.Log10(sig/noise)
		t.Logf("%d Hz -> 24 kHz: SNR vs librosa %.1f dB", sr, snr)
		if snr < 50 {
			t.Errorf("%d Hz: SNR %.1f dB < 50", sr, snr)
		}
	}
}

// BenchmarkResample/from=...: 10 s of audio to 24 kHz.
func BenchmarkResample(b *testing.B) {
	for _, from := range []int{16000, 44100, 48000} {
		x := tone(10*from, from, 440, 0.5)
		b.Run(fmt.Sprintf("from=%d", from), func(b *testing.B) {
			for b.Loop() {
				Resample(x, from, 24000)
			}
		})
	}
}
