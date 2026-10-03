// Package stableaudio runs Stable Audio 3 text-to-audio on the ingot
// runtime from the ONNX graphs Stability AI publishes
// (stabilityai/stable-audio-3-optimized): the T5Gemma text encoder, the
// diffusion transformer with its conditioner baked in, and the SAME
// autoencoder's decoder. The host side here is the tokenizer, the
// rectified-flow "pingpong" sampler and its logSNR schedule, as in the
// reference optimized/tflite pipeline.
//
// The weights are under the Stability AI Community License and T5Gemma under
// the Gemma Terms of Use; neither ships with ingot.
package stableaudio

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/tensor"
	"github.com/giraffesyo/ingot/tokenizer"
)

const (
	// SampleRate of the generated audio.
	SampleRate = 44100
	// SamplesPerLatent is the decoder's upsampling: one latent frame is
	// 4096 samples (about 93 ms).
	SamplesPerLatent = 4096

	latentDim  = 256
	condTokens = 256
	condDim    = 768
)

// Variants are the diffusion transformers in the optimized release that
// pair with the SAME-S decoder.
var Variants = []string{"sm-sfx", "sm-music"}

// Model is a loaded Stable Audio 3 pipeline. It is safe for one Generate at
// a time.
type Model struct {
	Variant string
	tok     *tokenizer.SentencePiece
	t5      *graph.Session
	dit     *graph.Session
	dec     *graph.Session
}

// Options for one generation.
type Options struct {
	// Seconds of audio. The model is conditioned on the next whole second
	// and the result cut to this length.
	Seconds float64
	// Steps of the pingpong sampler; 0 is the post-trained models' 8.
	Steps int
	// Seed of the starting noise and each step's fresh noise.
	Seed uint64
}

// Snapshot finds the stabilityai/stable-audio-3-optimized snapshot in the
// Hugging Face cache.
func Snapshot() (string, error) {
	hub := ""
	if h := os.Getenv("HF_HOME"); h != "" {
		hub = filepath.Join(h, "hub")
	} else {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		hub = filepath.Join(home, ".cache", "huggingface", "hub")
	}
	snaps, _ := filepath.Glob(filepath.Join(hub, "models--stabilityai--stable-audio-3-optimized", "snapshots", "*"))
	for i := len(snaps) - 1; i >= 0; i-- {
		if _, err := os.Stat(filepath.Join(snaps[i], "onnx", "t5gemma", "encoder.onnx")); err == nil {
			return snaps[i], nil
		}
	}
	return "", fmt.Errorf("stableaudio: no stabilityai/stable-audio-3-optimized snapshot under %s (hf download stabilityai/stable-audio-3-optimized onnx/sa3-sm-sfx/dit.onnx onnx/same-s/dec_bf16_limiter.onnx onnx/t5gemma/encoder.onnx tokenizer.model)", hub)
}

// Load compiles the three graphs of a variant from a snapshot directory of
// stabilityai/stable-audio-3-optimized.
func Load(dir, variant string) (*Model, error) {
	ok := false
	for _, v := range Variants {
		ok = ok || v == variant
	}
	if !ok {
		return nil, fmt.Errorf("stableaudio: unknown variant %q (have %v)", variant, Variants)
	}
	m := &Model{Variant: variant}
	var err error
	if m.tok, err = tokenizer.LoadSentencePiece(filepath.Join(dir, "tokenizer.model")); err != nil {
		return nil, err
	}
	for _, g := range []struct {
		path string
		into **graph.Session
	}{
		{filepath.Join("onnx", "t5gemma", "encoder.onnx"), &m.t5},
		{filepath.Join("onnx", "sa3-"+variant, "dit.onnx"), &m.dit},
		// The limiter build ends in a peak limiter and rounds to 16-bit
		// without dither, so the same latent always decodes the same.
		{filepath.Join("onnx", "same-s", "dec_bf16_limiter.onnx"), &m.dec},
	} {
		mod, err := onnx.DecodeFile(filepath.Join(dir, g.path))
		if err != nil {
			return nil, fmt.Errorf("stableaudio: %s: %w", g.path, err)
		}
		gr, err := graph.FromONNX(mod)
		if err != nil {
			return nil, fmt.Errorf("stableaudio: %s: %w", g.path, err)
		}
		if *g.into, err = graph.Compile(gr); err != nil {
			return nil, fmt.Errorf("stableaudio: %s: %w", g.path, err)
		}
	}
	return m, nil
}

// Tokens are the prompt's piece ids, truncated to the encoder's 256.
func (m *Model) Tokens(prompt string) []int {
	ids := m.tok.Encode(prompt)
	if len(ids) > condTokens {
		ids = ids[:condTokens]
	}
	return ids
}

// Latents is the latent length for a duration: whole frames, and an even
// count, which the SAME-S decoder needs.
func Latents(seconds float64) int {
	n := max(1, int(math.Ceil(seconds*SampleRate/SamplesPerLatent)))
	return n + n%2
}

// Schedule is the pingpong sampler's sigmas: steps+1 points from 1 down to
// 0, an even grid warped through a logSNR shift.
func Schedule(steps int) []float32 {
	const anchor, end = -6.2, 2.0
	s := make([]float32, steps+1)
	for i := range s {
		t := 1 - float64(i)/float64(steps)
		switch {
		case t <= 0:
			s[i] = 0
		case t >= 1:
			s[i] = 1
		default:
			s[i] = float32(1 / (1 + math.Exp(end-t*(end-anchor))))
		}
	}
	s[0] = 1
	return s
}

// Generate makes stereo audio for a prompt: left and right channels in
// [-1, 1] at SampleRate, trimmed to o.Seconds.
func (m *Model) Generate(prompt string, o Options) (left, right []float32, err error) {
	if o.Seconds <= 0 {
		return nil, nil, fmt.Errorf("stableaudio: seconds must be positive")
	}
	// The models were trained with the duration rounded up to whole seconds
	// (seconds_total = ceil(samples / rate)), and a fractional condition is
	// far outside what its Fourier features saw: it gives noise. So the
	// condition and the latent length are for whole seconds, and the audio
	// is cut to o.Seconds.
	whole := math.Ceil(o.Seconds)
	return m.generate(prompt, o, Latents(whole), float32(whole))
}

// generate samples L latent frames with the duration condition cond.
func (m *Model) generate(prompt string, o Options, L int, cond float32) (left, right []float32, err error) {
	steps := o.Steps
	if steps <= 0 {
		steps = 8
	}
	// Text conditioning.
	ids := make([]int64, condTokens)
	mask := make([]int64, condTokens)
	fmask := make([]float32, condTokens)
	for i, id := range m.Tokens(prompt) {
		ids[i], mask[i], fmask[i] = int64(id), 1, 1
	}
	res, err := m.t5.Run(map[string]*tensor.Tensor{
		"input_ids":      tensor.FromI64(ids, 1, condTokens),
		"attention_mask": tensor.FromI64(mask, 1, condTokens),
	})
	if err != nil {
		return nil, nil, fmt.Errorf("stableaudio: text encoder: %w", err)
	}
	hidden := append([]float32(nil), res["hidden_states"].F32()...)
	m.t5.Release(res)
	if len(hidden) != condTokens*condDim {
		return nil, nil, fmt.Errorf("stableaudio: text encoder returned %d values", len(hidden))
	}

	// Sampling: rectified flow, re-noising the denoised estimate each step.
	n := latentDim * L
	rng := rand.New(rand.NewPCG(o.Seed, 0x5a3))
	noise := func() []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	x := noise()
	sig := Schedule(steps)
	feeds := map[string]*tensor.Tensor{
		"t5_hidden":      tensor.FromF32(hidden, 1, condTokens, condDim),
		"t5_mask":        tensor.FromF32(fmask, 1, condTokens),
		"seconds_total":  tensor.FromF32([]float32{cond}, 1),
		"local_add_cond": tensor.FromF32(make([]float32, (latentDim+1)*L), 1, latentDim+1, L),
	}
	for i := 0; i < steps; i++ {
		tc, tn := sig[i], sig[i+1]
		feeds["x"] = tensor.FromF32(x, 1, latentDim, L)
		feeds["t"] = tensor.FromF32([]float32{tc}, 1)
		res, err := m.dit.Run(feeds)
		if err != nil {
			return nil, nil, fmt.Errorf("stableaudio: step %d: %w", i, err)
		}
		v := res["velocity"].F32()
		if len(v) != n {
			return nil, nil, fmt.Errorf("stableaudio: step %d returned %d values, want %d", i, len(v), n)
		}
		if i < steps-1 && tn > 0 {
			fresh := noise()
			for j := range x {
				x[j] = (1-tn)*(x[j]-tc*v[j]) + tn*fresh[j]
			}
		} else {
			for j := range x {
				x[j] -= tc * v[j]
			}
		}
		m.dit.Release(res)
	}
	for _, v := range x {
		if v != v {
			return nil, nil, fmt.Errorf("stableaudio: the sampler produced NaN")
		}
	}

	res, err = m.dec.Run(map[string]*tensor.Tensor{"latent": tensor.FromF32(x, 1, latentDim, L)})
	if err != nil {
		return nil, nil, fmt.Errorf("stableaudio: decoder: %w", err)
	}
	defer m.dec.Release(res)
	pcm := res["pcm"]
	if s := pcm.Shape(); len(s) != 3 || s[2] != 2 || pcm.DType() != tensor.I32 {
		return nil, nil, fmt.Errorf("stableaudio: decoder returned %s %v", pcm.DType(), pcm.Shape())
	}
	p := pcm.I32()
	frames := len(p) / 2
	if o.Seconds > 0 {
		frames = min(frames, int(math.Round(o.Seconds*SampleRate)))
	}
	left, right = make([]float32, frames), make([]float32, frames)
	for i := 0; i < frames; i++ {
		left[i] = float32(p[2*i]) / 32768
		right[i] = float32(p[2*i+1]) / 32768
	}
	return left, right, nil
}
