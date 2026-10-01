// Package qwen3tts runs Qwen3-TTS-12Hz on the ingot runtime — CustomVoice
// (preset speakers), VoiceDesign (a voice from a text description) and Base
// (a voice cloned from a reference clip) — built in Go over the Hugging Face
// safetensors checkpoint:
//
//   - the talker, a Qwen3 decoder that predicts codebook 0 of each 12.5 Hz
//     audio frame from summed text + codec embeddings;
//   - the code predictor, a small Qwen3 decoder run per frame that fills in
//     codebooks 1..15 from the talker's hidden state;
//   - the speech-tokenizer decoder (codec): RVQ lookup, a sliding-window
//     transformer, and a causal conv upsampler to a 24 kHz waveform;
//   - for cloning, the speaker encoder (ECAPA-TDNN over a log-mel
//     spectrogram) and the codec encoder (Mimi) that turns the reference
//     clip into codes.
//
// Reference: qwen-tts 0.1.1 (Qwen3TTSForConditionalGeneration.generate,
// non-streaming text mode, and Qwen3TTSTokenizerV2Decoder).
package qwen3tts

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
	"github.com/giraffesyo/ingot/tokenizer"
)

// LMConfig is one Qwen3 decoder stack's config (talker or code predictor).
type LMConfig struct {
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float64 `json:"rope_theta"`
	VocabSize         int     `json:"vocab_size"`
	HiddenAct         string  `json:"hidden_act"`
	AttentionBias     bool    `json:"attention_bias"`
	NumCodeGroups     int     `json:"num_code_groups"`
}

// TalkerConfig is config.json's talker_config.
type TalkerConfig struct {
	LMConfig
	TextHiddenSize  int                `json:"text_hidden_size"`
	TextVocabSize   int                `json:"text_vocab_size"`
	CodePredictor   LMConfig           `json:"code_predictor_config"`
	CodecBOS        int64              `json:"codec_bos_id"`
	CodecEOS        int64              `json:"codec_eos_token_id"`
	CodecPad        int64              `json:"codec_pad_id"`
	CodecThink      int64              `json:"codec_think_id"`
	CodecNoThink    int64              `json:"codec_nothink_id"`
	CodecThinkBOS   int64              `json:"codec_think_bos_id"`
	CodecThinkEOS   int64              `json:"codec_think_eos_id"`
	CodecLanguageID map[string]int64   `json:"codec_language_id"`
	SpeakerID       map[string]int64   `json:"spk_id"`
	SpeakerDialect  map[string]dialect `json:"spk_is_dialect"`
	RopeScaling     struct {
		MRopeSection []int `json:"mrope_section"`
		Interleaved  bool  `json:"interleaved"`
	} `json:"rope_scaling"`
}

// dialect is spk_is_dialect's value: false, or the dialect's language key.
type dialect string

func (d *dialect) UnmarshalJSON(b []byte) error {
	if string(b) == "false" {
		*d = ""
		return nil
	}
	return json.Unmarshal(b, (*string)(d))
}

// Config is the checkpoint's config.json.
type Config struct {
	TTSModelType string       `json:"tts_model_type"`
	TTSModelSize string       `json:"tts_model_size"`
	Tokenizer    string       `json:"tokenizer_type"`
	TTSBOS       int64        `json:"tts_bos_token_id"`
	TTSEOS       int64        `json:"tts_eos_token_id"`
	TTSPad       int64        `json:"tts_pad_token_id"`
	Talker       TalkerConfig `json:"talker_config"`
	// SpeakerEncoder is set on Base checkpoints.
	SpeakerEncoder SpeakerEncoderConfig `json:"speaker_encoder_config"`
}

// validate refuses configurations this package does not implement.
func (c *Config) validate() error {
	t, p := c.Talker, c.Talker.CodePredictor
	switch {
	case c.TTSModelType != "custom_voice" && c.TTSModelType != "voice_design" && c.TTSModelType != "base":
		return fmt.Errorf("qwen3tts: tts_model_type %q not implemented", c.TTSModelType)
	case c.Tokenizer != "qwen3_tts_tokenizer_12hz":
		return fmt.Errorf("qwen3tts: tokenizer_type %q not implemented", c.Tokenizer)
	}
	for _, l := range []LMConfig{t.LMConfig, p} {
		if l.HiddenAct != "silu" || l.AttentionBias || l.NumKeyValueHeads == 0 || l.NumAttentionHeads%l.NumKeyValueHeads != 0 || l.HeadDim%2 != 0 {
			return fmt.Errorf("qwen3tts: decoder variant not implemented: %+v", l)
		}
	}
	if p.NumCodeGroups != t.NumCodeGroups || t.NumCodeGroups < 2 {
		return fmt.Errorf("qwen3tts: code groups %d/%d", t.NumCodeGroups, p.NumCodeGroups)
	}
	// Text-only prompts put the same position on every M-RoPE axis, which
	// is plain 1-D RoPE whatever the section split.
	return nil
}

// Model is a loaded checkpoint: config, weights (mapped), tokenizer.
type Model struct {
	Cfg   Config
	Codec CodecConfig
	Tok   *tokenizer.Tokenizer
	Dir   string

	set, codecSet *safetensors.Set
	textEmb       *tensor.Tensor   // [text_vocab, text_hidden] stored dtype
	codecEmb      *tensor.Tensor   // talker codec_embedding [vocab, hidden]
	cpEmb         []*tensor.Tensor // code predictor codec_embedding[g], g = 0..groups-2
	cpHead        []*tensor.Tensor // code predictor lm_head[g] [2048, hidden]
}

// Load opens a snapshot directory (config.json, model.safetensors,
// vocab.json/merges.txt, speech_tokenizer/).
func Load(dir string) (*Model, error) {
	m := &Model{Dir: dir}
	if err := readJSON(dir, "config.json", &m.Cfg); err != nil {
		return nil, err
	}
	if err := m.Cfg.validate(); err != nil {
		return nil, err
	}
	m.Cfg.SpeakerEncoder.defaults()
	var err error
	if m.Codec, err = loadCodecConfig(filepath.Join(dir, "speech_tokenizer")); err != nil {
		return nil, err
	}
	add, err := tokenizer.AddedTokensFromConfig(filepath.Join(dir, "tokenizer_config.json"))
	if err != nil {
		return nil, err
	}
	// qwen-tts loads the processor with fix_mistral_regex=True, which swaps
	// the Qwen2 split for the Mistral one (tools/export/qwen3tts_ref.py saves
	// the backend tokenizer it runs).
	if m.Tok, err = tokenizer.LoadBPE(filepath.Join(dir, "vocab.json"), filepath.Join(dir, "merges.txt"), add, tokenizer.PreMistral); err != nil {
		return nil, err
	}
	if m.set, err = safetensors.OpenDir(dir); err != nil {
		return nil, err
	}
	if m.codecSet, err = safetensors.OpenDir(filepath.Join(dir, "speech_tokenizer")); err != nil {
		return nil, err
	}
	// Every weight is read on the first utterance: start the page-in now.
	m.set.WillNeed()
	m.codecSet.WillNeed()
	w := weights{set: m.set}
	defer catch(&err)
	t := m.Cfg.Talker
	m.textEmb = w.raw("talker.model.text_embedding.weight")
	m.codecEmb = w.raw("talker.model.codec_embedding.weight")
	for g := range t.NumCodeGroups - 1 {
		m.cpEmb = append(m.cpEmb, w.raw(fmt.Sprintf("talker.code_predictor.model.codec_embedding.%d.weight", g)))
		m.cpHead = append(m.cpHead, w.raw(fmt.Sprintf("talker.code_predictor.lm_head.%d.weight", g)))
	}
	if s := m.textEmb.Shape(); len(s) != 2 || s[1] != t.TextHiddenSize {
		return nil, fmt.Errorf("qwen3tts: text_embedding shape %v", s)
	}
	if s := m.codecEmb.Shape(); len(s) != 2 || s[1] != t.HiddenSize {
		return nil, fmt.Errorf("qwen3tts: codec_embedding shape %v", s)
	}
	return m, err
}

// Speakers lists the preset speaker names.
func (m *Model) Speakers() []string {
	var out []string
	for s := range m.Cfg.Talker.SpeakerID {
		out = append(out, s)
	}
	return out
}

// Languages lists the language names ("auto" plus codec_language_id's
// non-dialect keys).
func (m *Model) Languages() []string {
	out := []string{"auto"}
	for l := range m.Cfg.Talker.CodecLanguageID {
		if !contains(l, "dialect") {
			out = append(out, l)
		}
	}
	return out
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// embedRows gathers table rows (bf16 or f32) as f32, adding into dst when
// accumulate is set: dst[i] (+)= table[ids[i]].
func embedRows(dst []float32, table *tensor.Tensor, ids []int64, accumulate bool) error {
	D := table.Shape()[1]
	vocab := table.Shape()[0]
	for i, id := range ids {
		if id < 0 || int(id) >= vocab {
			return fmt.Errorf("qwen3tts: id %d outside table of %d rows", id, vocab)
		}
		row := dst[i*D : (i+1)*D]
		switch table.DType() {
		case tensor.BF16:
			src := table.BF16()[int(id)*D : (int(id)+1)*D]
			if accumulate {
				for j, v := range src {
					row[j] += math.Float32frombits(uint32(v) << 16)
				}
			} else {
				for j, v := range src {
					row[j] = math.Float32frombits(uint32(v) << 16)
				}
			}
		case tensor.F32:
			src := table.F32()[int(id)*D : (int(id)+1)*D]
			if accumulate {
				for j, v := range src {
					row[j] += v
				}
			} else {
				copy(row, src)
			}
		default:
			return fmt.Errorf("qwen3tts: embedding dtype %s", table.DType())
		}
	}
	return nil
}

// weights reads tensors from a checkpoint, panicking with a loadError that
// the Build*/Load entry points recover into an ordinary error — model
// builders then read like the reference module tree.
type weights struct {
	set    *safetensors.Set
	prefix string
}

type loadError struct{ err error }

func (w weights) scope(p string) weights { return weights{w.set, w.prefix + p + "."} }

func (w weights) f32(name string) *tensor.Tensor {
	t, err := w.set.F32(w.prefix + name)
	if err != nil {
		panic(loadError{err})
	}
	return t
}

// raw returns prefix+name in its stored dtype, zero-copy: bf16 Linear
// weights stay views of the mapped file and Gemm packs them from bf16.
func (w weights) raw(name string) *tensor.Tensor {
	t, err := w.set.Tensor(w.prefix + name)
	if err != nil {
		panic(loadError{err})
	}
	return t
}

func (w weights) has(name string) bool {
	_, ok := w.set.Info(w.prefix + name)
	return ok
}

// catch converts a loadError panic into *err; other panics propagate.
func catch(err *error) {
	if r := recover(); r != nil {
		le, ok := r.(loadError)
		if !ok {
			panic(r)
		}
		*err = le.err
	}
}

func readJSON(dir, name string, v any) error {
	raw, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return fmt.Errorf("qwen3tts: %w", err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		return fmt.Errorf("qwen3tts: %s/%s: %w", dir, name, err)
	}
	return nil
}
