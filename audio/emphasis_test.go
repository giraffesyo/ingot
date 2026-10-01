package audio

import (
	"math"
	"testing"
)

// vowel is a harmonic signal with a slow pitch glide (speech-like), f0
// from f0a to f0b over n samples.
func vowel(n, rate int, f0a, f0b, amp float64) []float32 {
	x := make([]float32, n)
	ph := 0.0
	for i := range x {
		f := f0a + (f0b-f0a)*float64(i)/float64(n)
		ph += 2 * math.Pi * f / float64(rate)
		x[i] = float32(amp * (math.Sin(ph) + 0.5*math.Sin(2*ph) + 0.25*math.Sin(3*ph)))
	}
	return x
}

func snr(ref, got []float32) float64 {
	var s, e float64
	for i := range ref {
		d := float64(got[i] - ref[i])
		s += float64(ref[i]) * float64(ref[i])
		e += d * d
	}
	return 10 * math.Log10(s/e)
}

func rmsOf(x []float32) float64 {
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	return math.Sqrt(s / float64(len(x)))
}

func medianF0(x []float32, rate int) float64 {
	var f []float64
	for _, p := range Yin(x, rate, 60, 500, rate/25, rate/200) {
		if p.Aperiodicity < 0.2 {
			f = append(f, p.F0)
		}
	}
	if len(f) == 0 {
		return 0
	}
	for i := range f {
		for j := i + 1; j < len(f); j++ {
			if f[j] < f[i] {
				f[i], f[j] = f[j], f[i]
			}
		}
	}
	return f[len(f)/2]
}

// TestEmphasizeIdentity: with no effect the resynthesised region
// reconstructs the input (the overlap-add is weight-normalised).
func TestEmphasizeIdentity(t *testing.T) {
	const rate = 24000
	x := vowel(rate, rate, 140, 180, 0.3)
	y := Emphasize(x, rate, 0.3, 0.7, Emphasis{Stretch: 1})
	if len(y) != len(x) {
		t.Fatalf("length %d, want %d", len(y), len(x))
	}
	if s := snr(x[rate/10:len(x)-rate/10], y[rate/10:len(y)-rate/10]); s < 25 {
		t.Errorf("identity SNR %.1f dB < 25", s)
	}
}

// TestEmphasizeSynthetic: pitch, length and level changes come out as
// asked on a steady vowel, and audio outside the region is untouched.
func TestEmphasizeSynthetic(t *testing.T) {
	const rate = 24000
	x := vowel(rate, rate, 150, 150, 0.3)
	// Pitch: a flat +4 st over the span's middle (the accent is a half
	// sine: its middle third sits within ~0.6 st of the peak).
	e := Emphasis{PitchST: 4, GainDB: 6, Stretch: 1.5}
	start, end := 0.3, 0.7
	y := Emphasize(x, rate, start, end, e)
	grow := len(y) - len(x)
	// The stretch plateau covers the span; its raised-cosine 25 ms ramps
	// count half each.
	wantGrow := (end - start - 0.025) * rate * (e.Stretch - 1)
	if math.Abs(float64(grow)-wantGrow) > 0.03*rate*(end-start) {
		t.Errorf("grew %d samples, want ≈ %.0f", grow, wantGrow)
	}
	// Middle of the stressed span in the output.
	mid := int((start+(end-start)/2)*rate) + grow/2
	seg := y[mid-rate/30 : mid+rate/30]
	if f := medianF0(seg, rate); math.Abs(12*math.Log2(f/150)-4) > 0.6 {
		t.Errorf("pitch at the accent peak %.1f Hz = %+.2f st, want +4", f, 12*math.Log2(f/150))
	}
	if g := 20 * math.Log10(rmsOf(seg)/rmsOf(x[mid-rate/30:mid+rate/30])); math.Abs(g-6) > 1 {
		t.Errorf("level %+.2f dB, want +6", g)
	}
	// Untouched before the region and (shifted) after it.
	r0 := int((start - 0.06) * rate)
	for i := range r0 {
		if y[i] != x[i] {
			t.Fatalf("sample %d before the region changed", i)
		}
	}
	r1 := int((end + 0.06) * rate)
	for i := r1; i < len(x); i++ {
		if y[i+grow] != x[i] {
			t.Fatalf("sample %d after the region changed", i)
		}
	}
	// And the region's own seams: the f0 just outside the span is the
	// original 150 Hz.
	pre := y[int((start-0.05)*rate):int((start-0.01)*rate)]
	if f := medianF0(pre, rate); math.Abs(f-150) > 2 {
		t.Errorf("f0 before the span %.1f Hz, want 150", f)
	}
}
