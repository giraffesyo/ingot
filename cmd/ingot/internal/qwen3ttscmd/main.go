// Package qwen3ttscmd is ingot qwen3tts: text spoken with Qwen3-TTS-12Hz,
// running entirely on the ingot runtime (pure Go, CPU). The checkpoint type
// picks the voice:
//
//	# CustomVoice: a preset speaker
//	ingot qwen3tts --text "Hello from ingot." --speaker ryan --out hello.wav
//
//	# VoiceDesign: a voice from a description
//	ingot qwen3tts --model 1.7B-VoiceDesign --instruct "A calm, warm woman in her thirties" \
//	    --text "The lighthouse keeper counted every wave." --out ref.wav
//
//	# Base: clone a reference clip (with its transcript), save the voice once,
//	# then reuse it for every line so they all sound alike
//	ingot qwen3tts --model 1.7B-Base --ref-audio ref.wav --ref-text "The lighthouse keeper counted every wave." \
//	    --save-voice keeper.voice
//	ingot qwen3tts --model 1.7B-Base --voice keeper.voice --lines lines.tsv --out-dir out/
//
// --model is a snapshot directory or a Qwen3-TTS-12Hz size+type found in the
// HF cache (0.6B-CustomVoice, 1.7B-VoiceDesign, 1.7B-Base, ...). --lines
// reads "name<TAB>text" per line and writes <out-dir>/<name>.wav. Reference
// audio at other rates than 24 kHz is resampled.
//
// Stress: mark words with asterisks — "I *never* said that" — and they are
// emphasised after synthesis (a pitch accent, a level lift and a little
// lengthening on the aligned word; --stress-pitch/-gain/-stretch). English;
// needs the wav2vec2 aligner (tools/export/align_ref.py). ingot prosody
// measures the result.
package qwen3ttscmd

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/giraffesyo/ingot/audio"
	"github.com/giraffesyo/ingot/models/align"
	"github.com/giraffesyo/ingot/models/qwen3tts"
)

// options are the qwen3tts flags.
type options struct {
	text, lines, outDir, out                  string
	speaker, instruct, refAudio, refText      string
	voice, saveVoice, language, model, device string
	alignerModel, alignerVocab                string
	seed                                      uint64
	maxFrames                                 int
	greedy, list, markup, int8w               bool
	stress                                    audio.Emphasis
}

// Command returns the qwen3tts subcommand.
func Command() *cobra.Command {
	o := options{stress: audio.DefaultEmphasis}
	cmd := &cobra.Command{
		Use:   "qwen3tts",
		Short: "Speak text with Qwen3-TTS (preset, designed or cloned voices)",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return run(&o)
		},
	}
	f := cmd.Flags()
	f.StringVar(&o.text, "text", "", "text to speak")
	f.StringVar(&o.lines, "lines", "", `file of "name<TAB>text" lines to speak, one WAV each into --out-dir`)
	f.StringVar(&o.outDir, "out-dir", ".", "output directory for --lines")
	f.StringVar(&o.out, "out", "out.wav", "output WAV (24 kHz mono, 16-bit) for --text")
	f.StringVar(&o.speaker, "speaker", "ryan", "CustomVoice: preset speaker (--list shows them)")
	f.StringVar(&o.instruct, "instruct", "", "VoiceDesign: description of the voice (CustomVoice 1.7B: a style instruction)")
	f.StringVar(&o.refAudio, "ref-audio", "", "Base: reference clip to clone (WAV; resampled to 24 kHz if needed)")
	f.StringVar(&o.refText, "ref-text", "", "Base: transcript of --ref-audio (in-context cloning; empty: x-vector only)")
	f.StringVar(&o.voice, "voice", "", "Base: a voice saved with --save-voice")
	f.StringVar(&o.saveVoice, "save-voice", "", "Base: save the cloned voice to this file")
	f.StringVar(&o.language, "language", "auto", "language: auto or a supported name (--list)")
	f.Uint64Var(&o.seed, "seed", 0, "sampling seed")
	f.BoolVar(&o.greedy, "greedy", false, "argmax decoding instead of sampling (deterministic; the reference parity mode)")
	f.IntVar(&o.maxFrames, "max-frames", 0, "cap on generated frames per line, 12.5 per second (default: the checkpoint's max_new_tokens)")
	f.StringVar(&o.model, "model", "0.6B-CustomVoice", "snapshot directory, or a Qwen3-TTS-12Hz size+type in the HF cache")
	f.BoolVar(&o.list, "list", false, "list speakers and languages, then exit")
	f.BoolVar(&o.markup, "markup", true, "treat *word* in the text as stress: the word is emphasised after synthesis (pitch accent, level, length)")
	f.Float64Var(&o.stress.PitchST, "stress-pitch", o.stress.PitchST, "stress: pitch accent peak in semitones")
	f.Float64Var(&o.stress.GainDB, "stress-gain", o.stress.GainDB, "stress: level lift in dB")
	f.Float64Var(&o.stress.Stretch, "stress-stretch", o.stress.Stretch, "stress: lengthening factor")
	f.StringVar(&o.alignerModel, "aligner", "testdata/align/wav2vec2-base-960h.onnx", "stress: word aligner ONNX (tools/export/align_ref.py)")
	f.StringVar(&o.alignerVocab, "aligner-vocab", "", "stress: aligner vocab.json (default: facebook/wav2vec2-base-960h in the HF cache)")
	f.BoolVar(&o.int8w, "int8", false, "int8 weights for the decode loop (GPU, or arm64 CPU; x86 CPUs keep bf16): faster, audio differs slightly from full precision")
	f.StringVar(&o.device, "device", "auto", "where the decode loop runs: auto (the GPU when available), gpu, cpu")
	return cmd
}

func run(f *options) error {
	dir, err := resolveModel(f.model)
	if err != nil {
		return err
	}
	t0 := time.Now()
	m, err := qwen3tts.Load(dir)
	if err != nil {
		return err
	}
	if f.list {
		sp, lang := m.Speakers(), m.Languages()
		sort.Strings(sp)
		sort.Strings(lang)
		fmt.Printf("model:     %s %s\n", m.Cfg.TTSModelSize, m.Cfg.TTSModelType)
		fmt.Println("speakers: ", strings.Join(sp, ", "))
		fmt.Println("languages:", strings.Join(lang, ", "))
		return nil
	}

	o := m.DefaultOptions()
	o.Speaker, o.Instruct, o.Language = f.speaker, f.instruct, f.language
	if f.maxFrames > 0 {
		o.MaxFrames = f.maxFrames
	}
	if f.greedy {
		o.Talker.DoSample, o.Sub.DoSample = false, false
	}
	if m.Cfg.TTSModelType == "base" {
		if o.Clone, err = cloneVoice(m, f.voice, f.refAudio, f.refText, f.saveVoice); err != nil {
			return err
		}
	}

	type job struct{ name, text, out string }
	var jobs []job
	switch {
	case f.text != "":
		jobs = append(jobs, job{"", f.text, f.out})
	case f.lines != "":
		jobs2, err := readLines(f.lines)
		if err != nil {
			return err
		}
		for _, j := range jobs2 {
			jobs = append(jobs, job{j[0], j[1], filepath.Join(f.outDir, j[0]+".wav")})
		}
		if err := os.MkdirAll(f.outDir, 0o755); err != nil {
			return err
		}
	case f.saveVoice != "":
		return nil // only building the voice
	default:
		return fmt.Errorf("--text or --lines is required")
	}

	// Stress markup: synthesise the clean text, then emphasise the marked
	// words (English; the aligner loads on first use).
	type markedJob struct {
		clean  string
		marked []int
	}
	marks := make([]markedJob, len(jobs))
	for i, j := range jobs {
		marks[i].clean = j.text
		if f.markup {
			marks[i].clean, marks[i].marked = align.ParseStress(j.text)
			jobs[i].text = marks[i].clean
		}
	}
	var aligner *align.Aligner
	loadAligner := func() (*align.Aligner, error) {
		if aligner != nil {
			return aligner, nil
		}
		vocab := f.alignerVocab
		if vocab == "" {
			home, _ := os.UserHomeDir()
			m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--facebook--wav2vec2-base-960h/snapshots/*/vocab.json"))
			if len(m) == 0 {
				return nil, fmt.Errorf("stress markup needs the facebook/wav2vec2-base-960h aligner (HF cache) or --aligner-vocab")
			}
			vocab = m[0]
		}
		a, err := align.Load(f.alignerModel, vocab)
		if err != nil {
			return nil, fmt.Errorf("stress markup: %w", err)
		}
		aligner = a
		return a, nil
	}

	// One compiled Synth sized for the longest line.
	maxIDs := 0
	for _, j := range jobs {
		ids, err := m.Tok.Encode(qwen3tts.AssistantText(j.text))
		if err != nil {
			return err
		}
		maxIDs = max(maxIDs, len(ids))
	}
	extra := 256 // instruction / reference rows
	if o.Clone != nil {
		extra += len(o.Clone.RefCodes) + 256
	}
	q := qwen3tts.Full
	if f.int8w {
		q = qwen3tts.Int8
	}
	s, err := m.NewSynth(maxIDs+extra+o.MaxFrames, q, f.device)
	if err != nil {
		return err
	}
	defer s.Close()
	fmt.Fprintf(os.Stderr, "qwen3tts: %s %s on %s loaded in %.2f s\n", m.Cfg.TTSModelSize, m.Cfg.TTSModelType, s.Device(), time.Since(t0).Seconds())
	for i, j := range jobs {
		o.Talker.Seed, o.Sub.Seed = f.seed, f.seed+1
		o.Talker.RNG, o.Sub.RNG = nil, nil
		t1 := time.Now()
		res, err := s.Synthesize(j.text, o)
		if err != nil {
			return fmt.Errorf("%s: %w", j.out, err)
		}
		if mk := marks[i]; len(mk.marked) > 0 {
			a, err := loadAligner()
			if err != nil {
				return err
			}
			if res.Wav, err = a.StressMarked(res.Wav, res.SampleRate, mk.clean, mk.marked, f.stress); err != nil {
				return fmt.Errorf("%s: %w", j.out, err)
			}
		}
		if err := writeWAV(j.out, res); err != nil {
			return err
		}
		audio := float64(len(res.Wav)) / float64(res.SampleRate)
		gen := time.Since(t1).Seconds()
		fmt.Fprintf(os.Stderr, "qwen3tts: [%d/%d] %s: %.2f s of audio (%d frames) in %.2f s (%.2fx realtime)\n",
			i+1, len(jobs), j.out, audio, len(res.Frames), gen, audio/gen)
	}
	return nil
}

// cloneVoice loads a saved voice or builds one from reference audio.
func cloneVoice(m *qwen3tts.Model, voice, refAudio, refText, save string) (*qwen3tts.VoicePrompt, error) {
	var vp *qwen3tts.VoicePrompt
	switch {
	case voice != "":
		f, err := os.Open(voice)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		if vp, err = qwen3tts.LoadVoice(f); err != nil {
			return nil, err
		}
	case refAudio != "":
		f, err := os.Open(refAudio)
		if err != nil {
			return nil, err
		}
		wav, rate, err := audio.ReadWAV(f)
		f.Close()
		if err != nil {
			return nil, err
		}
		cl, err := m.NewCloner()
		if err != nil {
			return nil, err
		}
		if vp, err = cl.NewVoicePrompt(wav, rate, refText); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("a Base checkpoint clones a voice: pass --ref-audio (and --ref-text) or --voice")
	}
	if save != "" {
		f, err := os.Create(save)
		if err != nil {
			return nil, err
		}
		if err := qwen3tts.SaveVoice(f, vp); err != nil {
			f.Close()
			return nil, err
		}
		if err := f.Close(); err != nil {
			return nil, err
		}
	}
	return vp, nil
}

func readLines(path string) ([][2]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out [][2]string
	sc := bufio.NewScanner(f)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimRight(sc.Text(), "\r")
		if strings.TrimSpace(line) == "" || strings.HasPrefix(line, "#") {
			continue
		}
		name, text, ok := strings.Cut(line, "\t")
		if !ok || name == "" || strings.TrimSpace(text) == "" {
			return nil, fmt.Errorf("%s:%d: want name<TAB>text", path, n)
		}
		out = append(out, [2]string{name, text})
	}
	return out, sc.Err()
}

func writeWAV(path string, res *qwen3tts.Result) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := audio.WriteWAV(f, res.Wav, res.SampleRate); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// resolveModel maps a size+type name to its snapshot in the HF cache; a
// directory is used as is.
func resolveModel(name string) (string, error) {
	if st, err := os.Stat(name); err == nil && st.IsDir() {
		return name, nil
	}
	hub := os.Getenv("HF_HUB_CACHE")
	if hub == "" {
		if home := os.Getenv("HF_HOME"); home != "" {
			hub = filepath.Join(home, "hub")
		} else {
			u, err := os.UserHomeDir()
			if err != nil {
				return "", err
			}
			hub = filepath.Join(u, ".cache", "huggingface", "hub")
		}
	}
	repo := "Qwen3-TTS-12Hz-" + name
	snaps, _ := filepath.Glob(filepath.Join(hub, "models--Qwen--"+repo, "snapshots", "*"))
	if len(snaps) == 0 {
		return "", fmt.Errorf("checkpoint Qwen/%s not found under %s (download it, or pass a snapshot directory)", repo, hub)
	}
	return snaps[len(snaps)-1], nil
}
