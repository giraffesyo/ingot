package audio

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestYinTones: tones with a harmonic come back at their frequency within
// 0.1% with near-zero aperiodicity, at the 128 ms frames low voices need.
func TestYinTones(t *testing.T) {
	const rate = 16000
	for _, f := range []float64{82, 150, 220, 440} {
		x := tone(rate, rate, f, 0.5)
		h := tone(rate, rate, 2*f, 0.2)
		for i := range x {
			x[i] += h[i]
		}
		p := Yin(x, rate, 60, 500, 2048, 160)
		mid := p[len(p)/4 : 3*len(p)/4]
		for _, q := range mid {
			if math.Abs(q.F0-f)/f > 1e-3 || q.Aperiodicity > 0.05 {
				t.Fatalf("%g Hz tone: f0 %.3f, aperiodicity %.3f", f, q.F0, q.Aperiodicity)
			}
		}
	}
}

// TestYinLibrosa: against librosa.yin on the aligner's speech clip
// (tools/export/align_ref.py) — a sanity bar, not parity: our difference
// function is the exact overlap sum where librosa's shortcut biases long
// lags, and librosa computes in float32. Clearly periodic frames
// (aperiodicity < 0.05) agree within 1%, 95% of voiced frames (< 0.2)
// within 2%; unvoiced frames have no meaningful f0 to compare.
func TestYinLibrosa(t *testing.T) {
	raw, err := os.ReadFile("../testdata/align/yin.json")
	if err != nil {
		t.Skip("YIN reference not generated (tools/export/align_ref.py)")
	}
	var man struct {
		Meta struct {
			Fmin, Fmax  float64
			SR          int `json:"sr"`
			FrameLength int `json:"frame_length"`
			HopLength   int `json:"hop_length"`
		} `json:"meta"`
		Tensors []struct{ Name, File string } `json:"tensors"`
	}
	if err := json.Unmarshal(raw, &man); err != nil {
		t.Fatal(err)
	}
	load := func(name string) []byte {
		for _, tn := range man.Tensors {
			if tn.Name == name {
				b, err := os.ReadFile(filepath.Join("../testdata/align", tn.File))
				if err != nil {
					t.Fatal(err)
				}
				return b
			}
		}
		t.Fatalf("no %s", name)
		return nil
	}
	xb, fb := load("x"), load("f0")
	x := make([]float32, len(xb)/4)
	for i := range x {
		x[i] = math.Float32frombits(binary.LittleEndian.Uint32(xb[4*i:]))
	}
	want := make([]float64, len(fb)/8)
	for i := range want {
		want[i] = math.Float64frombits(binary.LittleEndian.Uint64(fb[8*i:]))
	}
	m := man.Meta
	got := Yin(x, m.SR, m.Fmin, m.Fmax, m.FrameLength, m.HopLength)
	if len(got) != len(want) {
		t.Fatalf("%d frames, librosa %d", len(got), len(want))
	}
	close5, voiced, periodic := 0, 0, 0
	for i, p := range got {
		rel := math.Abs(p.F0-want[i]) / want[i]
		if p.Aperiodicity < 0.2 {
			periodic++
			if rel < 2e-2 {
				close5++
			}
		}
		if p.Aperiodicity < 0.05 {
			voiced++
			if rel > 1e-2 {
				t.Errorf("frame %d (aperiodicity %.3f): f0 %.3f, librosa %.3f", i, p.Aperiodicity, p.F0, want[i])
			}
		}
	}
	t.Logf("%d/%d voiced frames within 2%%; %d clearly voiced", close5, periodic, voiced)
	if periodic == 0 || float64(close5) < 0.95*float64(periodic) {
		t.Errorf("only %d/%d voiced frames within 2%%", close5, periodic)
	}
}
