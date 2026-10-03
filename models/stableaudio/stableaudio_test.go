package stableaudio

import (
	"math"
	"os"
	"testing"
)

func TestSchedule(t *testing.T) {
	s := Schedule(8)
	if len(s) != 9 || s[0] != 1 || s[8] != 0 {
		t.Fatalf("schedule %v", s)
	}
	for i := 1; i < len(s); i++ {
		if !(s[i] < s[i-1]) {
			t.Errorf("schedule is not decreasing at %d: %v", i, s)
		}
	}
	// sigma = sigmoid(-(end - t*(end-anchor))): at t = 0.5 the logSNR is
	// 2 - 4.1 = -2.1.
	if want := 1 / (1 + math.Exp(-2.1)); math.Abs(float64(s[4])-want) > 1e-6 {
		t.Errorf("mid sigma %v, want %v", s[4], want)
	}
	// Just under the top of the grid the warp is far from 1: the first
	// step is pinned there.
	if s[1] > 0.999 || s[1] < 0.99 {
		t.Errorf("second sigma %v", s[1])
	}
}

func TestLatents(t *testing.T) {
	for _, c := range []struct {
		sec  float64
		want int
	}{{0.01, 2}, {1, 12}, {3, 34}, {5, 54}, {4096.0 * 2 / 44100, 2}} {
		if got := Latents(c.sec); got != c.want {
			t.Errorf("Latents(%v) = %d, want %d", c.sec, got, c.want)
		}
	}
}

// TestGenerate runs the whole pipeline when the weights are in the Hugging
// Face cache: the same seed gives the same audio, another seed another, and
// two prompts at opposite ends of the spectrum land there.
func TestGenerate(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	dir, err := Snapshot()
	if err != nil {
		t.Skip(err)
	}
	if _, err := os.Stat(dir + "/onnx/sa3-sm-sfx/dit.onnx"); err != nil {
		t.Skip("sm-sfx graphs not in the cache")
	}
	m, err := Load(dir, "sm-sfx")
	if err != nil {
		t.Fatal(err)
	}
	if n := len(m.Tokens("a dog barking")); n < 3 || n > 6 {
		t.Errorf("%d tokens for a three-word prompt", n)
	}
	gen := func(prompt string, seed uint64) []float32 {
		l, r, err := m.Generate(prompt, Options{Seconds: 1.5, Seed: seed})
		if err != nil {
			t.Fatal(err)
		}
		if len(l) != int(1.5*SampleRate) || len(r) != len(l) {
			t.Fatalf("%d and %d samples", len(l), len(r))
		}
		return l
	}
	// brightness is an energy-weighted frequency in Hz: the energy of the
	// first difference over the signal's, which is (2·sin(πf/rate))² for a
	// sine at f. Quiet tails do not move it.
	brightness := func(x []float32) float64 {
		var d, e float64
		for i := 1; i < len(x); i++ {
			d += float64(x[i]-x[i-1]) * float64(x[i]-x[i-1])
			e += float64(x[i]) * float64(x[i])
		}
		return math.Asin(min(1, math.Sqrt(d/e)/2)) * SampleRate / math.Pi
	}
	rms := func(x []float32) float64 {
		var s float64
		for _, v := range x {
			s += float64(v) * float64(v)
		}
		return math.Sqrt(s / float64(len(x)))
	}
	low := gen("TrackType: SFX, deep thunder rumble", 1)
	again := gen("TrackType: SFX, deep thunder rumble", 1)
	for i := range low {
		if low[i] != again[i] {
			t.Fatalf("the same seed gave different audio at sample %d", i)
		}
	}
	other := gen("TrackType: SFX, deep thunder rumble", 2)
	same := true
	for i := range low {
		same = same && low[i] == other[i]
	}
	if same {
		t.Error("another seed gave the same audio")
	}
	high := gen("TrackType: SFX, a small glass bell struck once, bright ringing", 1)
	if rms(low) < 0.003 || rms(high) < 0.0005 {
		t.Errorf("near silence: rms %v and %v", rms(low), rms(high))
	}
	if bl, bh := brightness(low), brightness(high); !(bl < 400 && bh > 2000) {
		t.Errorf("thunder sits at %.0f Hz and the bell at %.0f Hz: the prompt is not steering the sound", bl, bh)
	} else {
		t.Logf("thunder %.0f Hz, bell %.0f Hz", bl, bh)
	}
}
