package qwen3tts

import (
	"fmt"
	"strings"

	"github.com/giraffesyo/ingot/generate"
	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/vek"
	"github.com/giraffesyo/ingot/tensor"
)

// Options controls one synthesis. DefaultOptions reads the checkpoint's
// generation_config.json.
type Options struct {
	// Exactly one way of choosing the voice, by checkpoint type:
	// Speaker, a preset (CustomVoice); Instruct, a description of the voice
	// (VoiceDesign; on 1.7B CustomVoice a style instruction alongside the
	// speaker); Clone, a reference prompt from NewVoicePrompt (Base).
	Speaker  string
	Instruct string
	Clone    *VoicePrompt
	Language string // "auto" or a codec_language_id key
	// Streaming feeds the text to the talker one token per generated frame
	// (generate's non_streaming_mode=false) instead of all in the prompt.
	// The reference defaults it on for cloning, off otherwise.
	Streaming bool
	// MaxFrames is generate()'s max_new_tokens: at most MaxFrames-1 frames
	// (the last sampled code 0 never gets its residual codebooks).
	MaxFrames int
	// Talker samples codebook 0 (repetition penalty, the 1024 control
	// tokens suppressed except EOS, EOS masked for the first two frames);
	// Sub samples codebooks 1..15.
	Talker, Sub generate.Sampler
}

// DefaultOptions are the checkpoint's sampling defaults (generation_config.json).
func (m *Model) DefaultOptions() Options {
	var gc struct {
		DoSample    *bool    `json:"do_sample"`
		RepPen      *float32 `json:"repetition_penalty"`
		Temperature *float32 `json:"temperature"`
		TopP        *float32 `json:"top_p"`
		TopK        *int     `json:"top_k"`
		SubSample   *bool    `json:"subtalker_dosample"`
		SubTemp     *float32 `json:"subtalker_temperature"`
		SubTopP     *float32 `json:"subtalker_top_p"`
		SubTopK     *int     `json:"subtalker_top_k"`
		MaxNew      *int     `json:"max_new_tokens"`
	}
	_ = readJSON(m.Dir, "generation_config.json", &gc)
	b := func(p *bool, d bool) bool {
		if p != nil {
			return *p
		}
		return d
	}
	f := func(p *float32, d float32) float32 {
		if p != nil {
			return *p
		}
		return d
	}
	i := func(p *int, d int) int {
		if p != nil {
			return *p
		}
		return d
	}
	// qwen-tts's hard defaults back any key the file lacks.
	return Options{
		Language:  "auto",
		Streaming: m.Cfg.TTSModelType == "base",
		MaxFrames: i(gc.MaxNew, 2048),
		Talker: generate.Sampler{DoSample: b(gc.DoSample, true), Temperature: f(gc.Temperature, 0.9),
			TopK: i(gc.TopK, 50), TopP: f(gc.TopP, 1), RepetitionPenalty: f(gc.RepPen, 1.05)},
		Sub: generate.Sampler{DoSample: b(gc.SubSample, true), Temperature: f(gc.SubTemp, 0.9),
			TopK: i(gc.SubTopK, 50), TopP: f(gc.SubTopP, 1)},
	}
}

// Synth holds the compiled stacks and per-sequence decode state. Not safe
// for concurrent use; one Synth per concurrent utterance.
type Synth struct {
	m        *Model
	talker   lmRunner
	cp       lmRunner
	textProj *graph.Session
	codec    *Codec
	maxT     int
	device   string // "cpu" or "gpu"
}

// Quant selects the weight precision of the decode loop.
type Quant string

const (
	// Full keeps the checkpoint's bf16 weights (the reference-parity mode).
	Full Quant = ""
	// Int8 runs the per-frame talker and code-predictor matmuls on int8
	// weights (groups of 64, Q8 activations): about twice the decode speed;
	// outputs move by the int8 rounding, so sampled audio differs from the
	// bf16 reference (the prompt prefill stays bf16).
	Int8 Quant = "int8"
)

// NewSynth compiles the talker, code predictor, text projection and codec
// for prompts plus generated frames up to maxT positions. device is "cpu",
// "gpu" (Apple Metal: the talker and code-predictor loop on the GPU) or
// "auto" (the GPU when there is one).
func (m *Model) NewSynth(maxT int, q Quant, device string) (*Synth, error) {
	return m.newSynth(maxT, q, device, -1, -1)
}

func (m *Model) newSynth(maxT int, q Quant, device string, talkerLayers, codecLayers int) (s *Synth, err error) {
	if q != Full && q != Int8 {
		return nil, fmt.Errorf("qwen3tts: unknown quantisation %q", q)
	}
	switch device {
	case "auto", "":
		device = "cpu"
		if metalAvailable() {
			device = "gpu"
		}
	case "cpu", "gpu":
	default:
		return nil, fmt.Errorf("qwen3tts: unknown device %q (cpu, gpu, auto)", device)
	}
	defer catch(&err)
	t := m.Cfg.Talker
	p := t.CodePredictor
	if talkerLayers < 0 {
		talkerLayers = t.NumHiddenLayers
	}
	w := weights{set: m.set}
	syn := &Synth{m: m, maxT: maxT, device: device}
	s = syn
	defer func() {
		if err != nil {
			syn.Close()
		}
	}()
	const cpProj = "talker.code_predictor.small_to_mtp_projection"
	hasProj := w.has(cpProj + ".weight")
	if device == "gpu" {
		if talkerLayers != t.NumHiddenLayers {
			return nil, fmt.Errorf("qwen3tts: truncated talkers run on the CPU only")
		}
		tl, err := newMetalLM(m.set, t.LMConfig, "talker.model.", t.HiddenSize, []string{"talker.codec_head.weight"}, "", maxT, max(maxT, 4), q)
		if err != nil {
			return nil, err
		}
		s.talker = tl
		var heads []string
		for g := range t.NumCodeGroups - 1 {
			heads = append(heads, fmt.Sprintf("talker.code_predictor.lm_head.%d.weight", g))
		}
		proj := ""
		if hasProj {
			proj = cpProj
		}
		cl, err := newMetalLM(m.set, p, "talker.code_predictor.model.", t.HiddenSize, heads, proj, t.NumCodeGroups+1, 4, q)
		if err != nil {
			return nil, err
		}
		s.cp = cl
		var tables []string
		for g := range t.NumCodeGroups - 2 {
			tables = append(tables, fmt.Sprintf("talker.code_predictor.model.codec_embedding.%d.weight", g))
		}
		if err := cl.setTables(m.set, tables); err != nil {
			return nil, err
		}
	} else {
		gT, err := buildLM("talker", t.LMConfig, w.scope("talker.model"), t.HiddenSize, nil, nil, w.raw("talker.codec_head.weight"), talkerLayers, q)
		if err != nil {
			return nil, err
		}
		var proj, projB *tensor.Tensor
		if hasProj {
			proj, projB = w.raw(cpProj+".weight"), w.f32(cpProj+".bias")
		}
		gP, err := buildLM("code_predictor", p, w.scope("talker.code_predictor.model"), t.HiddenSize, proj, projB, nil, p.NumHiddenLayers, q)
		if err != nil {
			return nil, err
		}
		tl, err := newCPULM(gT, newRope(t.RopeTheta, t.HeadDim, maxT), t.HiddenSize, maxT, t.HiddenSize, true, nil, q)
		if err != nil {
			return nil, err
		}
		cl, err := newCPULM(gP, newRope(p.RopeTheta, p.HeadDim, t.NumCodeGroups+1), t.HiddenSize, t.NumCodeGroups+1, p.HiddenSize, false, m.cpHead, q)
		if err != nil {
			return nil, err
		}
		s.talker, s.cp = tl, cl
	}
	gX, err := buildTextProjection(w.scope("talker.text_projection"))
	if err != nil {
		return nil, err
	}
	if s.textProj, err = graph.Compile(gX); err != nil {
		return nil, err
	}
	if s.codec, err = m.newCodec(codecLayers, device); err != nil {
		return nil, err
	}
	if device == "gpu" {
		// Warm-up: the GPU codec builds its constant tables (transposed
		// conv weights) on first use; pay that here, not on the first line.
		if _, err := s.codec.Decode([][]int64{make([]int64, t.NumCodeGroups)}); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// Device reports where the decode loop runs ("cpu" or "gpu").
func (s *Synth) Device() string { return s.device }

// Close releases GPU resources (a no-op on the CPU).
func (s *Synth) Close() {
	for _, r := range []lmRunner{s.talker, s.cp} {
		if r != nil {
			r.close()
		}
	}
}

// AssistantText wraps text in the chat template the talker was trained on.
func AssistantText(text string) string {
	return "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n"
}

// project runs text_projection over text-embedding rows of ids.
func (s *Synth) project(ids []int64) (*tensor.Tensor, error) {
	x := tensor.New(tensor.F32, len(ids), s.m.Cfg.Talker.TextHiddenSize)
	if err := embedRows(x.F32(), s.m.textEmb, ids, false); err != nil {
		return nil, err
	}
	out, err := s.textProj.Run(map[string]*tensor.Tensor{"x": x})
	if err != nil {
		return nil, err
	}
	return out["y"], nil
}

// InstructText wraps a voice/style instruction in its chat template.
func InstructText(instruct string) string {
	return "<|im_start|>user\n" + instruct + "<|im_end|>\n"
}

// RefText wraps a cloning reference's transcript in its chat template.
func RefText(text string) string {
	return "<|im_start|>assistant\n" + text + "<|im_end|>\n"
}

// rows accumulates [n, D] embedding rows.
type rows struct {
	D    int
	data []float32
}

// add appends the elementwise sum of src (zeros when empty) and returns the
// stored row, valid for writes until the next add.
func (r *rows) add(src ...[]float32) []float32 {
	r.data = append(r.data, make([]float32, r.D)...)
	row := r.data[len(r.data)-r.D:]
	for _, s := range src {
		for i, v := range s {
			row[i] += v
		}
	}
	return row
}

func (r *rows) n() int { return len(r.data) / r.D }

// Prompt builds the talker's prefill embeddings for ids (the tokenised
// AssistantText) under o's voice — Qwen3TTSForConditionalGeneration.generate
// for every checkpoint type — and the trailing text rows added to the
// generated frames in order (the last repeating: tts_pad once the text is
// consumed).
func (s *Synth) Prompt(ids []int64, o Options) (prompt, trailing []float32, err error) {
	c, t := s.m.Cfg, s.m.Cfg.Talker
	D := t.HiddenSize
	if len(ids) < 9 {
		return nil, nil, fmt.Errorf("qwen3tts: %d prompt tokens: not an assistant-templated text", len(ids))
	}
	lang := strings.ToLower(o.Language)
	if lang == "" {
		lang = "auto"
	}
	var langID int64 = -1
	if lang != "auto" {
		var ok bool
		if langID, ok = t.CodecLanguageID[lang]; !ok || strings.Contains(lang, "dialect") {
			return nil, nil, fmt.Errorf("qwen3tts: unknown language %q (have %v)", o.Language, s.m.Languages())
		}
	}
	// The speaker row: a preset's codec embedding, a clone's x-vector, or
	// none (voice design).
	var spkRow []float32
	switch c.TTSModelType {
	case "custom_voice":
		spk := strings.ToLower(o.Speaker)
		id, ok := t.SpeakerID[spk]
		if !ok {
			return nil, nil, fmt.Errorf("qwen3tts: unknown speaker %q (have %v)", o.Speaker, s.m.Speakers())
		}
		spkRow = make([]float32, D)
		if err := embedRows(spkRow, s.m.codecEmb, []int64{id}, false); err != nil {
			return nil, nil, err
		}
		if d := t.SpeakerDialect[spk]; d != "" && (lang == "chinese" || lang == "auto") {
			langID = t.CodecLanguageID[string(d)]
		}
	case "base":
		if o.Clone == nil {
			return nil, nil, fmt.Errorf("qwen3tts: a Base checkpoint clones a voice: set Options.Clone (NewVoicePrompt)")
		}
		if len(o.Clone.SpeakerEmbed) != D {
			return nil, nil, fmt.Errorf("qwen3tts: speaker embedding has %d dims, talker %d", len(o.Clone.SpeakerEmbed), D)
		}
		spkRow = o.Clone.SpeakerEmbed
	case "voice_design":
		if o.Instruct == "" {
			return nil, nil, fmt.Errorf("qwen3tts: a VoiceDesign checkpoint needs Options.Instruct (a description of the voice)")
		}
	}
	codec := []int64{t.CodecNoThink, t.CodecThinkBOS, t.CodecThinkEOS}
	if langID >= 0 {
		codec = []int64{t.CodecThink, t.CodecThinkBOS, langID, t.CodecThinkEOS}
	}
	codecRows := &rows{D: D}
	for _, id := range codec {
		r := codecRows.add()
		if err := embedRows(r, s.m.codecEmb, []int64{id}, false); err != nil {
			return nil, nil, err
		}
	}
	if spkRow != nil {
		codecRows.add(spkRow)
	}
	cPad, cBOS := make([]float32, D), make([]float32, D)
	if err := embedRows(cPad, s.m.codecEmb, []int64{t.CodecPad}, false); err != nil {
		return nil, nil, err
	}
	if err := embedRows(cBOS, s.m.codecEmb, []int64{t.CodecBOS}, false); err != nil {
		return nil, nil, err
	}
	codecRows.add(cPad)
	codecRows.add(cBOS)

	special, err := s.project([]int64{c.TTSBOS, c.TTSEOS, c.TTSPad})
	if err != nil {
		return nil, nil, err
	}
	bosE, eosE, padE := special.F32()[:D], special.F32()[D:2*D], special.F32()[2*D:3*D]

	out := &rows{D: D}
	// Instruction (voice design / style), projected text rows first.
	if o.Instruct != "" && !(c.TTSModelType == "custom_voice" && c.TTSModelSize == "0b6") {
		iids, err := s.m.Tok.Encode(InstructText(o.Instruct))
		if err != nil {
			return nil, nil, err
		}
		ins, err := s.project(iids)
		if err != nil {
			return nil, nil, err
		}
		out.data = append(out.data, ins.F32()...)
	}
	role, err := s.project(ids[:3]) // <|im_start|>assistant\n
	if err != nil {
		return nil, nil, err
	}
	out.data = append(out.data, role.F32()...)
	// tts_pad × (N-2), tts_bos, each plus codec row i < N-1.
	N := codecRows.n()
	for i := range N - 1 {
		txt := padE
		if i == N-2 {
			txt = bosE
		}
		out.add(txt, codecRows.data[i*D:(i+1)*D])
	}
	text := ids[3 : len(ids)-5]
	tail := &rows{D: D}

	if o.Clone != nil && o.Clone.RefCodes != nil {
		// In-context cloning: the reference transcript + target text against
		// the reference's codec frames (generate_icl_prompt).
		rids, err := s.m.Tok.Encode(RefText(o.Clone.RefText))
		if err != nil {
			return nil, nil, err
		}
		if len(rids) < 6 {
			return nil, nil, fmt.Errorf("qwen3tts: empty reference text")
		}
		tp, err := s.project(append(append([]int64{}, rids[3:len(rids)-2]...), text...))
		if err != nil {
			return nil, nil, err
		}
		txt := &rows{D: D, data: append(append([]float32{}, tp.F32()...), eosE...)}
		cod := &rows{D: D}
		cod.add(cBOS)
		for _, f := range o.Clone.RefCodes {
			r := cod.add()
			if err := s.frameEmbed(r, f); err != nil {
				return nil, nil, err
			}
		}
		T1, T2 := txt.n(), cod.n()
		switch {
		case !o.Streaming:
			for i := range T1 {
				out.add(txt.data[i*D:(i+1)*D], cPad)
			}
			for i := range T2 {
				out.add(cod.data[i*D:(i+1)*D], padE)
			}
		case T1 > T2:
			for i := range T2 {
				out.add(txt.data[i*D:(i+1)*D], cod.data[i*D:(i+1)*D])
			}
			tail.data = append(tail.data, txt.data[T2*D:]...)
		default:
			for i := range T2 {
				tr := padE
				if i < T1 {
					tr = txt.data[i*D : (i+1)*D]
				}
				out.add(tr, cod.data[i*D:(i+1)*D])
			}
		}
	} else {
		tp, err := s.project(text)
		if err != nil {
			return nil, nil, err
		}
		if o.Streaming {
			// First text token with the codec BOS; the rest trail.
			out.add(tp.F32()[:D], cBOS)
			tail.data = append(tail.data, tp.F32()[D:]...)
			tail.add(eosE)
		} else {
			for j := range len(text) {
				out.add(tp.F32()[j*D:(j+1)*D], cPad)
			}
			out.add(eosE, cPad)
			out.add(padE, cBOS)
		}
	}
	tail.add(padE)
	return out.data, tail.data, nil
}

// frameEmbed writes one frame's talker input embedding: codebook 0 through
// the talker's codec embedding plus codebooks 1.. through the code
// predictor's, summed.
func (s *Synth) frameEmbed(dst []float32, frame []int64) error {
	if len(frame) != len(s.m.cpEmb)+1 {
		return fmt.Errorf("qwen3tts: frame has %d codes, want %d", len(frame), len(s.m.cpEmb)+1)
	}
	if err := embedRows(dst, s.m.codecEmb, frame[:1], false); err != nil {
		return err
	}
	for g := 1; g < len(frame); g++ {
		if err := embedRows(dst, s.m.cpEmb[g-1], frame[g:g+1], true); err != nil {
			return err
		}
	}
	return nil
}

// Result is one synthesis.
type Result struct {
	Frames     [][]int64 // [F][16] codes
	Wav        []float32 // 24 kHz mono in [-1, 1]
	SampleRate int
	// PrefillLogits is the talker's first logits row (before processing);
	// kept for parity tests.
	PrefillLogits []float32
	// FirstFrameSubLogits are the code predictor's raw logits for frame 0,
	// codebooks 1..15.
	FirstFrameSubLogits [][]float32
}

// Synthesize speaks text in o's voice (a preset speaker, a designed voice
// or a cloned one, by checkpoint type).
func (s *Synth) Synthesize(text string, o Options) (*Result, error) {
	ids, err := s.m.Tok.Encode(AssistantText(text))
	if err != nil {
		return nil, err
	}
	frames, res, err := s.Codes(ids, o)
	if err != nil {
		return nil, err
	}
	res.SampleRate = s.m.Codec.SampleRate
	if o.Clone == nil || o.Clone.RefCodes == nil {
		res.Wav, err = s.codec.Decode(frames)
		return res, err
	}
	// In-context cloning decodes the reference codes ahead of the new ones
	// (the codec's left context) and cuts the reference's share of the
	// waveform off again.
	all := append(append([][]int64{}, o.Clone.RefCodes...), frames...)
	wav, err := s.codec.Decode(all)
	if err != nil {
		return nil, err
	}
	cut := int(float64(len(o.Clone.RefCodes)) / float64(max(len(all), 1)) * float64(len(wav)))
	res.Wav = wav[cut:]
	return res, nil
}

// Codes runs the talker + code predictor loop for prompt ids and returns
// the frames (stopping at EOS or o.MaxFrames).
func (s *Synth) Codes(ids []int64, o Options) ([][]int64, *Result, error) {
	t := s.m.Cfg.Talker
	D, G := t.HiddenSize, t.NumCodeGroups
	prompt, trailing, err := s.Prompt(ids, o)
	if err != nil {
		return nil, nil, err
	}
	nTrail := len(trailing) / D
	L := len(prompt) / D
	if L+o.MaxFrames > s.maxT {
		return nil, nil, fmt.Errorf("qwen3tts: prompt %d + %d frames exceeds maxT %d", L, o.MaxFrames, s.maxT)
	}
	ts := o.Talker
	ts.EOS, ts.MinNew = int(t.CodecEOS), 2
	for i := t.VocabSize - 1024; i < t.VocabSize; i++ {
		if int64(i) != t.CodecEOS {
			ts.Suppress = append(ts.Suppress, i)
		}
	}
	sub := o.Sub
	res := &Result{}

	s.talker.reset()
	hidden := make([]float32, D)
	logits := make([]float32, t.VocabSize)
	step := func(x []float32, n int) error {
		h, l, err := s.talker.run(x, n, 0)
		if err != nil {
			return err
		}
		copy(hidden, h)
		copy(logits, l)
		return nil
	}
	if err := step(prompt, L); err != nil {
		return nil, nil, err
	}
	res.PrefillLogits = append([]float32(nil), logits...)

	var history []int64
	var frames [][]int64
	next := make([]float32, D)
	in := make([]float32, 2*D)
	for len(history) < o.MaxFrames {
		c0 := int64(ts.Next(logits, history))
		history = append(history, c0)
		if c0 == t.CodecEOS || len(history) == o.MaxFrames {
			break
		}
		// Residual codebooks for this frame: the code predictor over
		// [talker hidden, embed(c0)], then one position per codebook.
		frame := make([]int64, G)
		frame[0] = c0
		s.cp.reset()
		copy(in, hidden)
		if err := embedRows(in[D:2*D], s.m.codecEmb, frame[:1], false); err != nil {
			return nil, nil, err
		}
		if fr, ok := s.cp.(*metalLM); ok && gpuSampleable(sub) {
			// The whole frame in one GPU command buffer: sampling there too,
			// on uniforms drawn here from the sampler's stream.
			us := make([]float32, G-1)
			if sub.DoSample {
				for i := range us {
					us[i] = float32(sub.Uniform())
				}
			}
			codes, lg, err := fr.runFrame(in, frameSampler{greedy: !sub.DoSample, topK: sub.TopK, temperature: sub.Temperature}, us, len(frames) == 0)
			if err != nil {
				return nil, nil, err
			}
			if len(frames) == 0 {
				res.FirstFrameSubLogits = lg
			}
			copy(frame[1:], codes)
		} else {
			n := 2
			var subHistory []int64
			for g := 1; g < G; g++ {
				_, subLogits, err := s.cp.run(in, n, g-1)
				if err != nil {
					return nil, nil, err
				}
				if len(frames) == 0 {
					res.FirstFrameSubLogits = append(res.FirstFrameSubLogits, append([]float32(nil), subLogits...))
				}
				code := int64(sub.Next(subLogits, subHistory))
				subHistory = append(subHistory, code)
				frame[g] = code
				n = 1
				if err := embedRows(in[:D], s.m.cpEmb[g-1], []int64{code}, false); err != nil {
					return nil, nil, err
				}
			}
		}
		frames = append(frames, frame)
		// Next talker input: every codebook's embedding summed, plus the
		// next trailing text row (tts_pad once the text is used up).
		if err := s.frameEmbed(next, frame); err != nil {
			return nil, nil, err
		}
		k := min(len(frames)-1, nTrail-1)
		for i, v := range trailing[k*D : (k+1)*D] {
			next[i] += v
		}
		if err := step(next, 1); err != nil {
			return nil, nil, err
		}
	}
	res.Frames = frames
	return frames, res, nil
}

// gpuSampleable reports whether the GPU frame sampler implements s: argmax,
// or temperature + top-k sampling (no top-p, penalties or masks).
func gpuSampleable(s generate.Sampler) bool {
	return (s.TopP <= 0 || s.TopP >= 1) && (s.RepetitionPenalty == 0 || s.RepetitionPenalty == 1) &&
		len(s.Suppress) == 0 && s.MinNew == 0
}

// gemvBF16 is out = W·x for a bf16 [N, K] weight (a per-step lm_head the
// graph cannot select: the code predictor has one per codebook).
func gemvBF16(out []float32, w *tensor.Tensor, x []float32) {
	N, K := w.Shape()[0], w.Shape()[1]
	wb := w.BF16()
	par.For(N, 64, func(r, _ int) {
		out[r] = vek.DotBF16(x, wb[r*K:(r+1)*K])
	})
}
