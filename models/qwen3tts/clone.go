package qwen3tts

import (
	"encoding/json"
	"fmt"
	"io"
	"math"

	"github.com/giraffesyo/ingot/audio"
	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/tensor"
)

// VoicePrompt is a cloned voice (Base checkpoints): the reference speaker's
// x-vector and, for in-context cloning, the reference clip's codes and
// transcript. Build once per voice with NewVoicePrompt and reuse it for
// every line, which keeps the voice consistent.
type VoicePrompt struct {
	SpeakerEmbed []float32 // [talker hidden]
	RefCodes     [][]int64 // [F][16]; nil: x-vector only
	RefText      string
}

// Cloner holds the speaker encoder and the codec encoder of a Base
// checkpoint.
type Cloner struct {
	m       *Model
	speaker *graph.Session
	mel     *graph.Session
	enc     *codecEncoder
}

// Speaker encoder front end (Qwen3TTSForConditionalGeneration.
// extract_speaker_embedding): 24 kHz, 1024-point Hann STFT, hop 256, 128
// Slaney mel bands over 0–12 kHz, log-compressed.
const (
	melFFT  = 1024
	melHop  = 256
	melBins = 128
	melRate = 24000
	melFMax = 12000
)

// NewCloner builds the speaker encoder (mel front end + ECAPA-TDNN) and the
// codec encoder of a Base checkpoint.
func (m *Model) NewCloner() (cl *Cloner, err error) {
	if m.Cfg.TTSModelType != "base" {
		return nil, fmt.Errorf("qwen3tts: cloning needs a Base checkpoint, not %s", m.Cfg.TTSModelType)
	}
	defer catch(&err)
	cl = &Cloner{m: m}
	gm, err := buildMel()
	if err != nil {
		return nil, err
	}
	if cl.mel, err = graph.Compile(gm); err != nil {
		return nil, err
	}
	gs, err := buildSpeakerEncoder(m.Cfg.SpeakerEncoder, weights{set: m.set, prefix: "speaker_encoder."})
	if err != nil {
		return nil, err
	}
	if cl.speaker, err = graph.Compile(gs); err != nil {
		return nil, err
	}
	if cl.enc, err = newCodecEncoder(m); err != nil {
		return nil, err
	}
	return cl, nil
}

// NewVoicePrompt clones the voice in wav (mono, [-1, 1], at sampleRate;
// other rates than 24 kHz are resampled first, as qwen-tts does with
// librosa — see audio.Resample for how closely). With refText it is
// in-context cloning (the reference codes condition the talker alongside
// its transcript, the reference's default); empty refText is
// x-vector-only cloning.
func (cl *Cloner) NewVoicePrompt(wav []float32, sampleRate int, refText string) (*VoicePrompt, error) {
	if sampleRate != melRate {
		if sampleRate <= 0 {
			return nil, fmt.Errorf("qwen3tts: reference sample rate %d", sampleRate)
		}
		wav = audio.Resample(wav, sampleRate, melRate)
	}
	emb, err := cl.SpeakerEmbedding(wav)
	if err != nil {
		return nil, err
	}
	vp := &VoicePrompt{SpeakerEmbed: emb, RefText: refText}
	if refText != "" {
		if vp.RefCodes, err = cl.enc.encode(wav); err != nil {
			return nil, err
		}
	}
	return vp, nil
}

// SpeakerEmbedding is the reference speaker's x-vector [enc_dim].
func (cl *Cloner) SpeakerEmbedding(wav []float32) ([]float32, error) {
	mel, err := cl.Mel(wav)
	if err != nil {
		return nil, err
	}
	out, err := cl.speaker.Run(map[string]*tensor.Tensor{"mel": mel})
	if err != nil {
		return nil, err
	}
	return append([]float32(nil), out["embedding"].F32()...), nil
}

// Mel is the speaker encoder's log-mel spectrogram [1, 128, frames].
func (cl *Cloner) Mel(wav []float32) (*tensor.Tensor, error) {
	if len(wav) < melFFT {
		return nil, fmt.Errorf("qwen3tts: reference audio too short (%d samples)", len(wav))
	}
	out, err := cl.mel.Run(map[string]*tensor.Tensor{"wav": tensor.FromF32(append([]float32(nil), wav...), 1, 1, len(wav))})
	if err != nil {
		return nil, err
	}
	return out["mel"], nil
}

// buildMel is mel_spectrogram (center=False after a reflect pad of
// (n_fft-hop)/2 each side): the STFT as a strided 1-D conv over windowed
// cos/sin kernels, magnitude sqrt(re²+im²+1e-9), the Slaney filterbank,
// log(max(·, 1e-5)). Input "wav" [1, 1, L]; output "mel" [1, 128, frames].
func buildMel() (*graph.Graph, error) {
	b := graph.NewBuilder("qwen3tts_mel")
	x := b.Input("wav", tensor.F32, 1, 1, -1)
	pad := int64((melFFT - melHop) / 2)
	x = b.Op("Pad", graph.Attr("mode", "reflect"), x, b.Ints(0, 0, pad, 0, 0, pad))
	bins := melFFT/2 + 1
	k := tensor.New(tensor.F32, 2*bins, 1, melFFT)
	for f := range bins {
		for n := range melFFT {
			w := 0.5 - 0.5*math.Cos(2*math.Pi*float64(n)/melFFT) // periodic Hann
			a := 2 * math.Pi * float64(f*n%melFFT) / melFFT
			k.F32()[f*melFFT+n] = float32(w * math.Cos(a))
			k.F32()[(bins+f)*melFFT+n] = float32(-w * math.Sin(a))
		}
	}
	spec := b.Op("Conv", graph.Attr("strides", []int{melHop}), x, b.Const("dft", k)) // [1, 2·bins, F]
	re, im := b.Slice(spec, 1, 0, int64(bins)), b.Slice(spec, 1, int64(bins), int64(2*bins))
	mag := b.Op("Sqrt", nil, b.Add(b.Add(b.Mul(re, re), b.Mul(im, im)), b.Scalar(1e-9)))
	mel := b.Op("MatMul", nil, b.Const("mel_basis", slaneyMel(melRate, melFFT, melBins, 0, melFMax)), mag)
	b.Output("mel", b.Op("Log", nil, b.Op("Max", nil, mel, b.Scalar(1e-5))))
	return b.Build()
}

// slaneyMel is librosa.filters.mel(htk=False, norm="slaney") [bands, bins],
// computed in float64 and stored f32 as librosa does.
func slaneyMel(sr, nFFT, bands int, fmin, fmax float64) *tensor.Tensor {
	const fsp = 200.0 / 3
	minLogHz, minLogMel, logStep := 1000.0, 1000.0/fsp, math.Log(6.4)/27
	hzToMel := func(f float64) float64 {
		if f >= minLogHz {
			return minLogMel + math.Log(f/minLogHz)/logStep
		}
		return f / fsp
	}
	melToHz := func(m float64) float64 {
		if m >= minLogMel {
			return minLogHz * math.Exp(logStep*(m-minLogMel))
		}
		return fsp * m
	}
	bins := nFFT/2 + 1
	fft := make([]float64, bins)
	for i := range fft {
		fft[i] = float64(i) * (float64(sr) / 2) / float64(bins-1)
	}
	lo, hi := hzToMel(fmin), hzToMel(fmax)
	mf := make([]float64, bands+2)
	for i := range mf {
		mf[i] = melToHz(lo + (hi-lo)*float64(i)/float64(bands+1))
	}
	out := tensor.New(tensor.F32, bands, bins)
	for i := range bands {
		enorm := 2 / (mf[i+2] - mf[i])
		for j, f := range fft {
			lower := (f - mf[i]) / (mf[i+1] - mf[i])
			upper := (mf[i+2] - f) / (mf[i+2] - mf[i+1])
			out.F32()[i*bins+j] = float32(math.Max(0, math.Min(lower, upper)) * enorm)
		}
	}
	return out
}

// SpeakerEncoderConfig is config.json's speaker_encoder_config, with the
// reference's class defaults for the keys checkpoints omit.
type SpeakerEncoderConfig struct {
	MelDim            int   `json:"mel_dim"`
	EncDim            int   `json:"enc_dim"`
	Channels          []int `json:"enc_channels"`
	KernelSizes       []int `json:"enc_kernel_sizes"`
	Dilations         []int `json:"enc_dilations"`
	AttentionChannels int   `json:"enc_attention_channels"`
	Res2NetScale      int   `json:"enc_res2net_scale"`
	SEChannels        int   `json:"enc_se_channels"`
	SampleRate        int   `json:"sample_rate"`
}

func (c *SpeakerEncoderConfig) defaults() {
	if c.MelDim == 0 {
		c.MelDim = 128
	}
	if c.EncDim == 0 {
		c.EncDim = 1024
	}
	if c.Channels == nil {
		c.Channels = []int{512, 512, 512, 512, 1536}
	}
	if c.KernelSizes == nil {
		c.KernelSizes = []int{5, 3, 3, 3, 1}
	}
	if c.Dilations == nil {
		c.Dilations = []int{1, 2, 3, 4, 1}
	}
	if c.AttentionChannels == 0 {
		c.AttentionChannels = 128
	}
	if c.Res2NetScale == 0 {
		c.Res2NetScale = 8
	}
	if c.SEChannels == 0 {
		c.SEChannels = 128
	}
	if c.SampleRate == 0 {
		c.SampleRate = 24000
	}
}

// buildSpeakerEncoder is Qwen3TTSSpeakerEncoder (ECAPA-TDNN) over "mel" [1,
// mel_dim, T] (T dynamic) → "embedding" [enc_dim]. Its convs pad "same"
// with reflection.
func buildSpeakerEncoder(c SpeakerEncoderConfig, w weights) (*graph.Graph, error) {
	b := graph.NewBuilder("qwen3tts_speaker")
	x := b.Input("mel", tensor.F32, 1, c.MelDim, -1)
	conv := func(b *graph.Builder, x *graph.Value, w weights, k, dil int) *graph.Value {
		if total := dil * (k - 1); total > 0 {
			l := int64(total / 2)
			x = b.Op("Pad", graph.Attr("mode", "reflect"), x, b.Ints(0, 0, l, 0, 0, int64(total)-l))
		}
		return b.Op("Conv", graph.Attr("dilations", []int{dil}), x, b.Const("weight", w.f32("weight")), b.Const("bias", w.f32("bias")))
	}
	tdnn := func(b *graph.Builder, x *graph.Value, w weights, k, dil int) *graph.Value {
		return b.Op("Relu", nil, conv(b, x, w.scope("conv"), k, dil))
	}
	n := len(c.Channels)
	x = tdnn(b.Scope("blocks.0"), x, w.scope("blocks.0"), c.KernelSizes[0], c.Dilations[0])
	var outs []*graph.Value
	for i := 1; i < n-1; i++ {
		bi, wi := b.Scope(fmt.Sprintf("blocks.%d", i)), w.scope(fmt.Sprintf("blocks.%d", i))
		res := x
		h := tdnn(bi.Scope("tdnn1"), x, wi.scope("tdnn1"), 1, 1)
		// Res2Net: channel chunks, each (after the first) through its own
		// TDNN on chunk + the previous chunk's output.
		C, S := c.Channels[i], c.Res2NetScale
		part := int64(C / S)
		var parts []*graph.Value
		var prev *graph.Value
		for j := range S {
			p := bi.Slice(h, 1, int64(j)*part, int64(j+1)*part)
			switch j {
			case 0:
			case 1:
				p = tdnn(bi.Scope(fmt.Sprintf("res2net.%d", j-1)), p, wi.scope(fmt.Sprintf("res2net_block.blocks.%d", j-1)), c.KernelSizes[i], c.Dilations[i])
			default:
				p = tdnn(bi.Scope(fmt.Sprintf("res2net.%d", j-1)), bi.Add(p, prev), wi.scope(fmt.Sprintf("res2net_block.blocks.%d", j-1)), c.KernelSizes[i], c.Dilations[i])
			}
			parts = append(parts, p)
			prev = p
		}
		h = tdnn(bi.Scope("tdnn2"), bi.Concat(1, parts...), wi.scope("tdnn2"), 1, 1)
		// Squeeze-excitation over the time mean.
		se := bi.Op("ReduceMean", graph.Attr("keepdims", 1), h, bi.Ints(2))
		se = bi.Op("Relu", nil, conv(bi.Scope("se1"), se, wi.scope("se_block.conv1"), 1, 1))
		se = bi.Op("Sigmoid", nil, conv(bi.Scope("se2"), se, wi.scope("se_block.conv2"), 1, 1))
		x = bi.Add(bi.Mul(h, se), res)
		outs = append(outs, x)
	}
	x = tdnn(b.Scope("mfa"), b.Concat(1, outs...), w.scope("mfa"), c.KernelSizes[n-1], c.Dilations[n-1])

	// Attentive statistics pooling. stats(x, weights over time) = weighted
	// mean and sqrt(max(weighted variance, 1e-12)).
	ba := b.Scope("asp")
	stats := func(x, wt *graph.Value) (*graph.Value, *graph.Value) {
		mean := ba.Op("ReduceSum", graph.Attr("keepdims", 1), ba.Mul(wt, x), ba.Ints(2))
		d := ba.Sub(x, mean)
		v := ba.Op("ReduceSum", graph.Attr("keepdims", 1), ba.Mul(wt, ba.Mul(d, d)), ba.Ints(2))
		return mean, ba.Op("Sqrt", nil, ba.Op("Max", nil, v, ba.Scalar(1e-12)))
	}
	// Uniform weights 1/T: the reference's all-ones mask over its sum.
	T := ba.Op("Cast", graph.Attr("to", int(tensor.F32)), ba.Op("Slice", nil, ba.Op("Shape", nil, x), ba.Ints(2), ba.Ints(3), ba.Ints(0)))
	uni := ba.Op("Reciprocal", nil, T)
	mean, std := stats(x, uni)
	shape := ba.Op("Shape", nil, x)
	att := ba.Concat(1, x, ba.Op("Expand", nil, mean, shape), ba.Op("Expand", nil, std, shape))
	wa := w.scope("asp")
	att = ba.Op("Tanh", nil, tdnn(ba.Scope("tdnn"), att, wa.scope("tdnn"), 1, 1))
	att = conv(ba.Scope("conv"), att, wa.scope("conv"), 1, 1)
	att = ba.Op("Softmax", graph.Attr("axis", 2), att)
	mean, std = stats(x, att)
	pooled := b.Concat(1, mean, std) // [1, 2C, 1]
	emb := conv(b.Scope("fc"), pooled, w.scope("fc"), 1, 1)
	b.Output("embedding", b.Reshape(emb, -1))
	return b.Build()
}

// codecEncoder is the speech tokenizer's encoder (a Mimi model): SEANet
// conv encoder, causal transformer, stride-2 downsample to 12.5 Hz, and
// split residual VQ (1 semantic + 15 acoustic codebooks kept). The conv
// padding depends on the input length, so the graph is built per length.
type codecEncoder struct {
	cfg                CodecEncoderConfig
	w                  weights
	semantic, acoustic [][]float32 // codebooks [size, dim]
	semNorms, acNorms  [][]float32 // |e|² per row
	semProj, acProj    *tensor.Tensor
	keep               int // codebooks kept (encoder_valid_num_quantizers)
}

// CodecEncoderConfig is speech_tokenizer/config.json's encoder_config.
type CodecEncoderConfig struct {
	AudioChannels     int     `json:"audio_channels"`
	CodebookDim       int     `json:"codebook_dim"`
	CodebookSize      int     `json:"codebook_size"`
	Compress          int     `json:"compress"`
	DilationGrowth    int     `json:"dilation_growth_rate"`
	HeadDim           int     `json:"head_dim"`
	HiddenAct         string  `json:"hidden_act"`
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	KernelSize        int     `json:"kernel_size"`
	LastKernelSize    int     `json:"last_kernel_size"`
	NormEps           float32 `json:"norm_eps"`
	NumHeads          int     `json:"num_attention_heads"`
	NumKVHeads        int     `json:"num_key_value_heads"`
	NumFilters        int     `json:"num_filters"`
	NumLayers         int     `json:"num_hidden_layers"`
	NumQuantizers     int     `json:"num_quantizers"`
	NumResidualLayers int     `json:"num_residual_layers"`
	NumSemantic       int     `json:"num_semantic_quantizers"`
	PadMode           string  `json:"pad_mode"`
	ResidualKernel    int     `json:"residual_kernel_size"`
	RopeTheta         float64 `json:"rope_theta"`
	UpsamplingRatios  []int   `json:"upsampling_ratios"`
	UseCausalConv     bool    `json:"use_causal_conv"`
	UseConvShortcut   bool    `json:"use_conv_shortcut"`
	VQHidden          int     `json:"vector_quantization_hidden_dimension"`
	FrameRate         float64 `json:"_frame_rate"`
}

func newCodecEncoder(m *Model) (*codecEncoder, error) {
	c := m.Codec.Encoder
	if !c.UseCausalConv || c.PadMode != "constant" || c.UseConvShortcut || c.HiddenAct != "gelu" || c.NumHeads != c.NumKVHeads || c.AudioChannels != 1 {
		return nil, fmt.Errorf("qwen3tts: codec encoder variant not implemented: %+v", c)
	}
	e := &codecEncoder{cfg: c, w: weights{set: m.codecSet, prefix: "encoder."}, keep: m.Codec.EncoderValidQuantizers}
	load := func(group string, n int) (cbs, norms [][]float32) {
		for i := range n {
			wi := e.w.scope(fmt.Sprintf("quantizer.%s.layers.%d.codebook", group, i))
			sum, usage := wi.f32("embed_sum"), wi.f32("cluster_usage")
			D := sum.Shape()[1]
			cb, nr := make([]float32, sum.Numel()), make([]float32, sum.Shape()[0])
			for r := range sum.Shape()[0] {
				u := max(usage.F32()[r], 1e-5)
				var s float32
				for j := range D {
					v := sum.F32()[r*D+j] / u
					cb[r*D+j] = v
					s += v * v
				}
				nr[r] = s
			}
			cbs, norms = append(cbs, cb), append(norms, nr)
		}
		return cbs, norms
	}
	e.semantic, e.semNorms = load("semantic_residual_vector_quantizer", c.NumSemantic)
	e.acoustic, e.acNorms = load("acoustic_residual_vector_quantizer", e.keep-c.NumSemantic)
	e.semProj = e.w.f32("quantizer.semantic_residual_vector_quantizer.input_proj.weight")
	e.acProj = e.w.f32("quantizer.acoustic_residual_vector_quantizer.input_proj.weight")
	return e, nil
}

// encode returns ceil(len/1920) frames of 16 codes for 24 kHz audio.
func (e *codecEncoder) encode(wav []float32) (codes [][]int64, err error) {
	g, err := e.build(len(wav))
	if err != nil {
		return nil, err
	}
	sess, err := graph.Compile(g)
	if err != nil {
		return nil, err
	}
	T := len(wav)
	frames := (T + 1919) / 1920
	cos, sin := newRope(e.cfg.RopeTheta, e.cfg.HeadDim, e.seqLen(T)).at(0, e.seqLen(T))
	out, err := sess.Run(map[string]*tensor.Tensor{
		"wav": tensor.FromF32(append([]float32(nil), wav...), 1, 1, T), "cos": cos, "sin": sin})
	if err != nil {
		return nil, err
	}
	sem, ac := out["semantic"], out["acoustic"] // [1, dim, F]
	F := sem.Shape()[2]
	if frames > F {
		return nil, fmt.Errorf("qwen3tts: encoder produced %d frames, want %d", F, frames)
	}
	all := make([][]int64, F)
	for f := range all {
		all[f] = make([]int64, 0, e.keep)
	}
	rvq(sem.F32(), F, e.semantic, e.semNorms, all)
	rvq(ac.F32(), F, e.acoustic, e.acNorms, all)
	return all[:frames], nil
}

// rvq quantises x [dim, F] residually against each codebook in turn,
// appending each frame's nearest index (Euclidean; first minimum wins).
func rvq(x []float32, F int, books, norms [][]float32, out [][]int64) {
	if len(books) == 0 {
		return
	}
	dim := len(x) / F
	res := make([]float32, F*dim) // [F, dim]
	for d := range dim {
		for f := range F {
			res[f*dim+d] = x[d*F+f]
		}
	}
	size := len(norms[0])
	dots := make([]float32, F*size)
	for q, cb := range books {
		// dots = res · cbᵀ; dist² = |r|² - 2 r·e + |e|² (|r|² is constant per frame).
		gemm.SgemmT(false, true, F, size, dim, 1, res, dim, cb, dim, 0, dots, size)
		for f := range F {
			best, bi := float32(math.Inf(1)), 0
			for j := range size {
				if d := norms[q][j] - 2*dots[f*size+j]; d < best {
					best, bi = d, j
				}
			}
			out[f] = append(out[f], int64(bi))
			row := cb[bi*dim : (bi+1)*dim]
			for d := range dim {
				res[f*dim+d] -= row[d]
			}
		}
	}
}

// seqLen is the transformer's sequence length (25 Hz) for T samples: the
// SEANet output length after every causal conv's extra padding.
func (e *codecEncoder) seqLen(T int) int {
	n := e.convOut(T, e.cfg.KernelSize, 1, 1)
	for _, r := range reversedInts(e.cfg.UpsamplingRatios) {
		n = e.convOut(n, 2*r, r, 1)
	}
	return e.convOut(n, e.cfg.LastKernelSize, 1, 1)
}

// convPads are MimiConv1d's causal paddings for input length n.
func convPads(n, k, stride, dil int) (left, right int) {
	keff := (k-1)*dil + 1
	total := keff - stride
	frames := int(math.Ceil(float64(n-keff+total)/float64(stride)+1)) - 1
	ideal := frames*stride + keff - total
	return total, ideal - n
}

func (e *codecEncoder) convOut(n, k, stride, dil int) int {
	l, r := convPads(n, k, stride, dil)
	return (n+l+r-((k-1)*dil+1))/stride + 1
}

func reversedInts(xs []int) []int {
	out := make([]int, len(xs))
	for i, v := range xs {
		out[len(xs)-1-i] = v
	}
	return out
}

// build is MimiModel._encode_frame up to the RVQ input projections, for T
// samples: inputs "wav" [1, 1, T], "cos"/"sin" for the transformer
// positions; outputs "semantic"/"acoustic" [1, vq_dim, frames].
func (e *codecEncoder) build(T int) (g *graph.Graph, err error) {
	defer catch(&err)
	c := e.cfg
	b := graph.NewBuilder("qwen3tts_codec_encoder")
	x := b.Input("wav", tensor.F32, 1, 1, T)
	cosV := b.Input("cos", tensor.F32, -1, c.HeadDim/2)
	sinV := b.Input("sin", tensor.F32, -1, c.HeadDim/2)
	n := T
	conv := func(b *graph.Builder, x *graph.Value, w weights, k, stride, dil int, mode string, bias bool) *graph.Value {
		l, r := convPads(n, k, stride, dil)
		if l > 0 || r > 0 {
			x = b.Op("Pad", graph.Attr("mode", mode), x, b.Ints(0, 0, int64(l), 0, 0, int64(r)))
		}
		n = (n+l+r-((k-1)*dil+1))/stride + 1
		in := []*graph.Value{x, b.Const("weight", w.f32("conv.weight"))}
		if bias {
			in = append(in, b.Const("bias", w.f32("conv.bias")))
		}
		return b.Op("Conv", graph.Attr("strides", []int{stride}, "dilations", []int{dil}), in...)
	}
	elu := func(b *graph.Builder, x *graph.Value) *graph.Value { return b.Op("Elu", nil, x) }

	we, be := e.w.scope("encoder.layers"), b.Scope("encoder")
	li := 0
	x = conv(be.Scope("0"), x, we.scope("0"), c.KernelSize, 1, 1, "constant", true)
	li++
	scale := 1
	for _, r := range reversedInts(c.UpsamplingRatios) {
		for j := range c.NumResidualLayers {
			bl, wl := be.Scope(fmt.Sprint(li)), we.scope(fmt.Sprint(li))
			dil := int(math.Pow(float64(c.DilationGrowth), float64(j)))
			h := conv(bl.Scope("1"), elu(bl, x), wl.scope("block.1"), c.ResidualKernel, 1, dil, "constant", true)
			h = conv(bl.Scope("3"), elu(bl, h), wl.scope("block.3"), 1, 1, 1, "constant", true)
			x = bl.Add(x, h)
			li++
		}
		x = elu(be, x)
		li++ // the ELU module
		x = conv(be.Scope(fmt.Sprint(li)), x, we.scope(fmt.Sprint(li)), 2*r, r, 1, "constant", true)
		li++
		scale *= 2
	}
	x = elu(be, x)
	li++
	x = conv(be.Scope(fmt.Sprint(li)), x, we.scope(fmt.Sprint(li)), c.LastKernelSize, 1, 1, "constant", true) // [1, hidden, n]

	// Causal transformer over [n, hidden].
	S := n
	mask := tensor.New(tensor.F32, S, S)
	for q := range S {
		for k := q + 1; k < S; k++ {
			mask.F32()[q*S+k] = float32(math.Inf(-1))
		}
	}
	maskV := b.Const("causal", mask)
	H, dh, D := c.NumHeads, c.HeadDim, c.HiddenSize
	wt, bt := e.w.scope("encoder_transformer"), b.Scope("transformer")
	h := bt.Reshape(bt.Transpose(x, 0, 2, 1), -1, int64(D))
	attnScale := float32(1 / math.Sqrt(float64(dh)))
	for i := range c.NumLayers {
		bi, wi := bt.Scope(fmt.Sprintf("layers.%d", i)), wt.scope(fmt.Sprintf("layers.%d", i))
		a := bi.LayerNorm(h, D, wi.f32("input_layernorm.weight"), wi.f32("input_layernorm.bias"), c.NormEps)
		q := bi.Linear(a, wi.f32("self_attn.q_proj.weight"), nil)
		k := bi.Linear(a, wi.f32("self_attn.k_proj.weight"), nil)
		v := bi.Linear(a, wi.f32("self_attn.v_proj.weight"), nil)
		q = bi.Op("ingot.RoPE", graph.Attr("layout", 1), bi.Reshape(q, -1, int64(H), int64(dh)), cosV, sinV)
		k = bi.Op("ingot.RoPE", graph.Attr("layout", 1), bi.Reshape(k, -1, int64(H), int64(dh)), cosV, sinV)
		o := bi.Op("ingot.SDPA", graph.Attr("scale", attnScale, "a_layout", 1, "b_layout", 1, "v_layout", 1, "stride_out", 1),
			bi.Reshape(q, 1, -1, int64(H), int64(dh)), bi.Reshape(k, 1, -1, int64(H), int64(dh)),
			bi.Reshape(v, 1, -1, int64(H), int64(dh)), maskV)
		o = bi.Linear(bi.Reshape(o, -1, int64(H*dh)), wi.f32("self_attn.o_proj.weight"), nil)
		h = bi.Add(h, bi.Mul(o, bi.Const("attn_scale", wi.f32("self_attn_layer_scale.scale"))))
		a = bi.LayerNorm(h, D, wi.f32("post_attention_layernorm.weight"), wi.f32("post_attention_layernorm.bias"), c.NormEps)
		m := bi.Linear(bi.Op("Gelu", nil, bi.Linear(a, wi.f32("mlp.fc1.weight"), nil)), wi.f32("mlp.fc2.weight"), nil)
		h = bi.Add(h, bi.Mul(m, bi.Const("mlp_scale", wi.f32("mlp_layer_scale.scale"))))
	}
	x = b.Transpose(b.Reshape(h, 1, -1, int64(D)), 0, 2, 1)
	// Downsample to the frame rate: causal, replicate-padded, no bias.
	x = conv(b.Scope("downsample"), x, e.w.scope("downsample"), 4, 2, 1, "edge", false)
	b.Output("semantic", b.Op("Conv", nil, x, b.Const("sem_proj", e.semProj)))
	b.Output("acoustic", b.Op("Conv", nil, x, b.Const("ac_proj", e.acProj)))
	return b.Build()
}

// voiceFile is a saved VoicePrompt (SaveVoice / LoadVoice).
type voiceFile struct {
	Format       string    `json:"format"`
	SpeakerEmbed []float32 `json:"speaker_embed"`
	RefCodes     [][]int64 `json:"ref_codes,omitempty"`
	RefText      string    `json:"ref_text,omitempty"`
}

const voiceFormat = "qwen3tts-voice-v1"

// SaveVoice writes vp as JSON, so a cloned voice is built once and reused
// for every later line.
func SaveVoice(w io.Writer, vp *VoicePrompt) error {
	enc := json.NewEncoder(w)
	return enc.Encode(voiceFile{voiceFormat, vp.SpeakerEmbed, vp.RefCodes, vp.RefText})
}

// LoadVoice reads a SaveVoice file.
func LoadVoice(r io.Reader) (*VoicePrompt, error) {
	var f voiceFile
	if err := json.NewDecoder(r).Decode(&f); err != nil {
		return nil, fmt.Errorf("qwen3tts: voice file: %w", err)
	}
	if f.Format != voiceFormat {
		return nil, fmt.Errorf("qwen3tts: voice file format %q, want %q", f.Format, voiceFormat)
	}
	return &VoicePrompt{SpeakerEmbed: f.SpeakerEmbed, RefCodes: f.RefCodes, RefText: f.RefText}, nil
}
