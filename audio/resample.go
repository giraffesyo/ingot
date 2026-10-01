package audio

import (
	"math"

	"github.com/giraffesyo/ingot/kernels/vek"
)

// Resampling: a Kaiser-windowed sinc low-pass evaluated polyphase for the
// rational ratio to/from (reduced by their gcd). The cutoff sits at
// resampleRolloff of the lower Nyquist rate, with resampleZeros zero
// crossings each side and a Kaiser window of beta resampleBeta (≈100 dB
// stopband). Rolloff and length were swept against librosa.resample's
// default (soxr "high quality", what qwen-tts runs on reference audio):
// 0.955 / 96 matched best, 52-56 dB SNR at 16, 22.05, 44.1 and 48 kHz →
// 24 kHz (0.96 / 48: 44-48; 0.955 / 192: 44-48) — a different filter, so
// agreement is to a measured SNR, not bit for bit.
const (
	resampleRolloff = 0.955
	resampleZeros   = 96
	resampleBeta    = 10.0
	// maxPhases bounds the precomputed filter table (phases × taps); ratios
	// with more phases evaluate the kernel directly.
	maxPhases = 4096
)

// Resample converts x from rate from to rate to and returns
// ceil(len(x)·to/from) samples (librosa's length). Samples outside x are
// zero.
func Resample(x []float32, from, to int) []float32 {
	if from <= 0 || to <= 0 {
		panic("audio: Resample: non-positive rate")
	}
	if from == to {
		return append([]float32(nil), x...)
	}
	g := gcd(from, to)
	up, down := to/g, from/g
	nOut := int((int64(len(x))*int64(to) + int64(from) - 1) / int64(from))
	out := make([]float32, nOut)
	// Cutoff relative to the input rate's Nyquist; half-width in input
	// samples.
	r := resampleRolloff * math.Min(1, float64(to)/float64(from))
	half := int(math.Ceil(resampleZeros / r))
	taps := 2*half + 1
	kernel := func(tau float64) float64 { // input-sample offset tau
		if math.Abs(tau) >= float64(half) {
			return 0
		}
		w := kaiser(tau/float64(half), resampleBeta)
		return r * sinc(r*tau) * w
	}
	// Output n sits at input time t = n·down/up = base + phase/up.
	var table []float32
	if up <= maxPhases {
		table = make([]float32, up*taps)
		for p := range up {
			frac := float64(p) / float64(up)
			for j := range taps {
				// tap j reads input base - half + j: offset t - k = frac + half - j
				table[p*taps+j] = float32(kernel(frac + float64(half-j)))
			}
		}
	}
	// A zero-padded copy so every window reads in bounds.
	padded := make([]float32, half+len(x)+half+1)
	copy(padded[half:], x)
	for n := range out {
		num := int64(n) * int64(down)
		base, p := int(num/int64(up)), int(num%int64(up))
		win := padded[base : base+taps] // inputs base-half .. base+half
		if table != nil {
			out[n] = vek.Dot(win, table[p*taps:(p+1)*taps])
			continue
		}
		frac := float64(p) / float64(up)
		var acc float64
		for j, v := range win {
			acc += float64(v) * kernel(frac+float64(half-j))
		}
		out[n] = float32(acc)
	}
	return out
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	return math.Sin(math.Pi*x) / (math.Pi * x)
}

// kaiser is the Kaiser window at u in [-1, 1].
func kaiser(u, beta float64) float64 {
	return bessI0(beta*math.Sqrt(max(0, 1-u*u))) / bessI0(beta)
}

// bessI0 is the zeroth-order modified Bessel function of the first kind
// (power series; converges quickly for the window's arguments).
func bessI0(x float64) float64 {
	sum, term := 1.0, 1.0
	for k := 1; k < 64; k++ {
		term *= (x / (2 * float64(k))) * (x / (2 * float64(k)))
		sum += term
		if term < sum*1e-17 {
			break
		}
	}
	return sum
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
