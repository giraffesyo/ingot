package audio

import (
	"math"
)

// Emphasis is how to stress a span of speech: a pitch accent (PitchST
// semitones at its peak, rising and falling over the span), a loudness
// lift (GainDB on a plateau) and a lengthening (Stretch, ≥ 1 slows the
// span's core down).
type Emphasis struct {
	PitchST float64
	GainDB  float64
	Stretch float64
}

// DefaultEmphasis clears the stress meter's default thresholds (+1 st,
// +1.5 dB, ×1.15) with margin while staying within what overlap-add
// resynthesis does cleanly.
var DefaultEmphasis = Emphasis{PitchST: 3, GainDB: 3, Stretch: 1.2}

// emphasis analysis constants: YIN over 40 ms frames every 5 ms; grains
// in unvoiced stretches every 10 ms; 60 ms of untouched margin around the
// span; 25 ms ramps into the plateau effects; 10 ms crossfades back into
// the original.
const (
	emphPitchFrameSec = 0.040
	emphHopSec        = 0.005
	emphUnvoicedSec   = 0.010
	emphMarginSec     = 0.060
	emphRampSec       = 0.025
	emphFadeSec       = 0.010
	emphAperiodic     = 0.25
)

// Emphasize returns x with the span [start, end) seconds stressed by e,
// using TD-PSOLA (pitch-synchronous overlap-add) on the span plus a
// margin: grains around pitch marks (one period apart, snapped to the
// waveform's peaks; fixed 10 ms grains where unvoiced) are re-spaced at
// period/pitch-ratio and re-timed by the stretch, Hann-windowed and
// overlap-added with the window sum divided out — so with no effect the
// region reconstructs the input. The region is spliced back with short
// crossfades; outside it x is unchanged. The result is longer than x by
// the stretch.
func Emphasize(x []float32, rate int, start, end float64, e Emphasis) []float32 {
	if e.Stretch <= 0 {
		e.Stretch = 1
	}
	sr := float64(rate)
	r0 := max(0, int((start-emphMarginSec)*sr))
	r1 := min(len(x), int((end+emphMarginSec)*sr))
	if r1-r0 < int(0.03*sr) || end <= start {
		return append([]float32(nil), x...)
	}
	region := x[r0:r1]
	s0, s1 := start*sr-float64(r0), end*sr-float64(r0) // span in region samples

	// Pitch track over the region (5 ms hop): period in samples, 0 when
	// unvoiced.
	hop := max(1, int(emphHopSec*sr))
	frame := int(emphPitchFrameSec * sr)
	pt := Yin(region, rate, 60, 500, frame, hop)
	period := func(t float64) float64 {
		i := int(math.Round(t / float64(hop)))
		if i < 0 || i >= len(pt) || pt[i].Aperiodicity >= emphAperiodic {
			return 0
		}
		return sr / pt[i].F0
	}
	unv := emphUnvoicedSec * sr

	// Effect envelopes over region time t: the plateau ramps over 25 ms
	// inside the span; the pitch accent is a half-sine over the span.
	ramp := emphRampSec * sr
	plateau := func(t float64) float64 {
		if t <= s0 || t >= s1 {
			return 0
		}
		r := math.Min(1, math.Min(t-s0, s1-t)/ramp)
		return 0.5 - 0.5*math.Cos(math.Pi*r)
	}
	accent := func(t float64) float64 {
		if t <= s0 || t >= s1 {
			return 0
		}
		return math.Sin(math.Pi * (t - s0) / (s1 - s0))
	}
	alpha := func(t float64) float64 { return math.Pow(2, e.PitchST*accent(t)/12) }
	beta := func(t float64) float64 { return 1 + (e.Stretch-1)*plateau(t) }
	gain := func(t float64) float64 { return math.Pow(10, e.GainDB*plateau(t)/20) }

	// Analysis marks: one period apart, each snapped to the largest
	// magnitude within a quarter period.
	var marks []int
	for t := 0.0; t < float64(len(region)); {
		p := period(t)
		m := int(t)
		if p > 0 {
			lo, hi := max(0, int(t-p/4)), min(len(region)-1, int(t+p/4))
			for i := lo; i <= hi; i++ {
				if math.Abs(float64(region[i])) > math.Abs(float64(region[m])) {
					m = i
				}
			}
		}
		if len(marks) == 0 || m > marks[len(marks)-1] {
			marks = append(marks, m)
		}
		step := p
		if step <= 0 {
			step = unv
		}
		t = math.Max(t+step, float64(m)+step/2)
	}

	// Synthesis: output time u and analysis time tau start at the first
	// mark. Each step places the grain of the analysis mark nearest tau at
	// u, then advances u by that mark's own spacing over the pitch ratio,
	// and tau by the same over the stretch — with no effect tau visits
	// every mark and each grain lands back where it came from.
	outLen := 0.0
	for t := 0.0; t < float64(len(region)); t++ {
		outLen += beta(t)
	}
	n := int(math.Ceil(outLen)) + 1
	acc := make([]float64, n+1)
	wsum := make([]float64, n+1)
	nearest := func(tau float64) int {
		lo, hi := 0, len(marks)-1
		for lo < hi {
			mid := (lo + hi) / 2
			if float64(marks[mid]) < tau {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		if lo > 0 && tau-float64(marks[lo-1]) <= float64(marks[lo])-tau {
			lo--
		}
		return lo
	}
	spacingOf := func(k int) float64 {
		switch {
		case len(marks) == 1:
			return unv
		case k+1 < len(marks):
			return float64(marks[k+1] - marks[k])
		}
		return float64(marks[k] - marks[k-1])
	}
	u, tau := float64(marks[0]), float64(marks[0])
	prevSpacing := 0.0
	for tau < float64(len(region)) && u < float64(n) {
		k := nearest(tau)
		a := marks[k]
		pk := spacingOf(k)
		voiced := period(float64(a)) > 0
		spacing := pk
		if voiced {
			spacing = pk / alpha(float64(a))
		}
		// Grain half-width: the synthesis spacing (the larger of this and
		// the previous step's), so each output sample sits under about two
		// grains whatever the pitch ratio — wider grains at a raised pitch
		// would overlap out of phase and cancel.
		h := int(math.Ceil(math.Max(spacing, prevSpacing)))
		prevSpacing = spacing
		g := gain(float64(a))
		uc := int(math.Round(u))
		for j := -h; j <= h; j++ {
			src, dst := a+j, uc+j
			if src < 0 || src >= len(region) || dst < 0 || dst > n {
				continue
			}
			w := 0.5 + 0.5*math.Cos(math.Pi*float64(j)/float64(h+1))
			acc[dst] += w * g * float64(region[src])
			wsum[dst] += w
		}
		u += spacing
		tau += spacing / beta(float64(a))
	}
	n = min(n, int(math.Round(outLen)))
	out := make([]float32, n)
	for i := range out {
		if wsum[i] > 1e-6 {
			out[i] = float32(acc[i] / wsum[i])
		}
	}

	// Splice: original before the region, crossfade in, the resynthesised
	// region, crossfade out, original after. At the region's edges the
	// effects are off, so the two signals agree there.
	fade := min(int(emphFadeSec*sr), len(out)/4)
	y := make([]float32, 0, len(x)+n-len(region))
	y = append(y, x[:r0]...)
	for i := range out {
		v := out[i]
		switch {
		case i < fade:
			w := float32(i) / float32(fade)
			v = w*v + (1-w)*region[i]
		case i >= len(out)-fade:
			j := i - (len(out) - fade) // 0..fade-1 into the tail
			w := float32(fade-j) / float32(fade)
			v = w*v + (1-w)*region[len(region)-fade+j]
		}
		y = append(y, v)
	}
	y = append(y, x[r1:]...)
	return y
}
