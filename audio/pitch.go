package audio

import (
	"math"

	"github.com/giraffesyo/ingot/kernels/par"
)

// Pitch is one frame's YIN estimate.
type Pitch struct {
	F0 float64 // Hz
	// Aperiodicity is the cumulative-mean-normalised difference at the
	// chosen period: near 0 for clean periodic (voiced) frames, near 1 for
	// noise. Callers threshold it (≈ 0.1-0.25) to decide voicing.
	Aperiodicity float64
}

// Yin estimates the fundamental frequency per frame: librosa.yin's
// framing (center=True with zero padding, frames of frameLength samples
// hop apart), trough threshold 0.1 and parabolic refinement, with the
// aperiodicity librosa computes but drops. f0 is searched in [fmin, fmax].
//
// One deliberate difference: the difference function is the exact
// overlap sum d(k) = Σ_{m<N-k} (y[m] - y[m+k])². librosa's shortcut
// 2(acf(0) - acf(k)) - Σ_{m<k} y[m]² keeps the full-frame energy where
// the overlap's belongs, which biases long lags: an 82 Hz tone reads
// 82.21 Hz at 128 ms frames (here: exact). Use frames of at least two
// periods of fmin (128 ms at 16 kHz for 60 Hz voices).
func Yin(x []float32, rate int, fmin, fmax float64, frameLength, hop int) []Pitch {
	minP := int(math.Floor(float64(rate) / fmax))
	maxP := min(int(math.Ceil(float64(rate)/fmin)), frameLength-1)
	pad := frameLength / 2
	padded := make([]float64, len(x)+2*pad)
	for i, v := range x {
		padded[pad+i] = float64(v)
	}
	n := 1 + (len(padded)-frameLength)/hop
	if len(padded) < frameLength {
		return nil
	}
	out := make([]Pitch, n)
	par.For(n, 4, func(f, _ int) {
		y := padded[f*hop : f*hop+frameLength]
		out[f] = yinFrame(y, rate, minP, maxP)
	})
	return out
}

// yinFrame is one frame of librosa's yin.
func yinFrame(y []float64, rate, minP, maxP int) Pitch {
	N := len(y)
	acf := make([]float64, maxP+1)
	for k := 0; k <= maxP; k++ {
		var s float64
		for m := 0; m+k < N; m++ {
			s += y[m] * y[m+k]
		}
		acf[k] = s
	}
	// d[k] = Σ_{m<N-k} (y[m] - y[m+k])² = head(N-k) + tail(k) - 2·acf[k],
	// head(n) = Σ_{m<n} y², tail(k) = Σ_{m≥k} y².
	cum := make([]float64, N+1)
	for i, v := range y {
		cum[i+1] = cum[i] + v*v
	}
	d := make([]float64, maxP+1)
	for k := 1; k <= maxP; k++ {
		d[k] = cum[N-k] + (cum[N] - cum[k]) - 2*acf[k]
	}
	// Cumulative mean normalisation over k = 1..maxP.
	yv := make([]float64, maxP-minP+1)
	var run float64
	for k := 1; k <= maxP; k++ {
		run += d[k]
		if k >= minP {
			yv[k-minP] = d[k] / (run/float64(k) + math.SmallestNonzeroFloat64)
		}
	}
	P := len(yv)
	shift := func(i int) float64 {
		if i <= 0 || i >= P-1 {
			return 0
		}
		a := yv[i+1] + yv[i-1] - 2*yv[i]
		b := (yv[i+1] - yv[i-1]) / 2
		if math.Abs(b) >= math.Abs(a) {
			return 0
		}
		return -b / a
	}
	trough := func(i int) bool {
		switch {
		case P == 1:
			return false
		case i == 0:
			return yv[0] < yv[1]
		case i == P-1:
			return yv[i] < yv[i-1]
		}
		return yv[i] < yv[i-1] && yv[i] <= yv[i+1]
	}
	idx := -1
	for i := range P {
		if trough(i) && yv[i] < 0.1 {
			idx = i
			break
		}
	}
	if idx < 0 {
		idx = 0
		for i := range P {
			if yv[i] < yv[idx] {
				idx = i
			}
		}
	}
	period := float64(minP+idx) + shift(idx)
	return Pitch{F0: float64(rate) / period, Aperiodicity: yv[idx]}
}

// FrameRMS is each centered frame's root-mean-square level (the same
// framing as Yin: zero padding of frameLength/2, hop samples apart).
func FrameRMS(x []float32, frameLength, hop int) []float64 {
	pad := frameLength / 2
	total := len(x) + 2*pad
	if total < frameLength {
		return nil
	}
	n := 1 + (total-frameLength)/hop
	out := make([]float64, n)
	for f := range out {
		var s float64
		for i := f*hop - pad; i < f*hop-pad+frameLength; i++ {
			if i >= 0 && i < len(x) {
				s += float64(x[i]) * float64(x[i])
			}
		}
		out[f] = math.Sqrt(s / float64(frameLength))
	}
	return out
}
