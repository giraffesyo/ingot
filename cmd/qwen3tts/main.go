// Command qwen3tts speaks text with Qwen3-TTS-12Hz, running entirely on the
// ingot runtime (pure Go, CPU). The checkpoint type picks the voice:
//
//	# CustomVoice: a preset speaker
//	qwen3tts -text "Hello from ingot." -speaker ryan -out hello.wav
//
//	# VoiceDesign: a voice from a description
//	qwen3tts -model 1.7B-VoiceDesign -instruct "A calm, warm woman in her thirties" \
//	    -text "The lighthouse keeper counted every wave." -out ref.wav
//
//	# Base: clone a reference clip (with its transcript), save the voice once,
//	# then reuse it for every line so they all sound alike
//	qwen3tts -model 1.7B-Base -ref-audio ref.wav -ref-text "The lighthouse keeper counted every wave." \
//	    -save-voice keeper.voice
//	qwen3tts -model 1.7B-Base -voice keeper.voice -lines lines.tsv -out-dir out/
//
// -model is a snapshot directory or a Qwen3-TTS-12Hz size+type found in the
// HF cache (0.6B-CustomVoice, 1.7B-VoiceDesign, 1.7B-Base, ...). -lines
// reads "name<TAB>text" per line and writes <out-dir>/<name>.wav. Reference
// audio must be 24 kHz.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/giraffesyo/ingot/models/qwen3tts"
)

func main() {
	text := flag.String("text", "", "text to speak")
	lines := flag.String("lines", "", `file of "name<TAB>text" lines to speak, one WAV each into -out-dir`)
	outDir := flag.String("out-dir", ".", "output directory for -lines")
	out := flag.String("out", "out.wav", "output WAV (24 kHz mono, 16-bit) for -text")
	speaker := flag.String("speaker", "ryan", "CustomVoice: preset speaker (-list shows them)")
	instruct := flag.String("instruct", "", "VoiceDesign: description of the voice (CustomVoice 1.7B: a style instruction)")
	refAudio := flag.String("ref-audio", "", "Base: reference clip to clone (WAV, 24 kHz)")
	refText := flag.String("ref-text", "", "Base: transcript of -ref-audio (in-context cloning; empty: x-vector only)")
	voice := flag.String("voice", "", "Base: a voice saved with -save-voice")
	saveVoice := flag.String("save-voice", "", "Base: save the cloned voice to this file")
	language := flag.String("language", "auto", "language: auto or a supported name (-list)")
	seed := flag.Uint64("seed", 0, "sampling seed")
	greedy := flag.Bool("greedy", false, "argmax decoding instead of sampling (deterministic; the reference parity mode)")
	maxFrames := flag.Int("max-frames", 0, "cap on generated frames per line, 12.5 per second (default: the checkpoint's max_new_tokens)")
	model := flag.String("model", "0.6B-CustomVoice", "snapshot directory, or a Qwen3-TTS-12Hz size+type in the HF cache")
	list := flag.Bool("list", false, "list speakers and languages, then exit")
	int8w := flag.Bool("int8", false, "int8 weights for the decode loop (GPU, or arm64 CPU; x86 CPUs keep bf16): faster, audio differs slightly from full precision")
	device := flag.String("device", "auto", "where the decode loop runs: auto (the GPU when available), gpu, cpu")
	flag.Parse()

	dir, err := resolveModel(*model)
	if err != nil {
		fail(err)
	}
	t0 := time.Now()
	m, err := qwen3tts.Load(dir)
	if err != nil {
		fail(err)
	}
	if *list {
		sp, lang := m.Speakers(), m.Languages()
		sort.Strings(sp)
		sort.Strings(lang)
		fmt.Printf("model:     %s %s\n", m.Cfg.TTSModelSize, m.Cfg.TTSModelType)
		fmt.Println("speakers: ", strings.Join(sp, ", "))
		fmt.Println("languages:", strings.Join(lang, ", "))
		return
	}

	o := m.DefaultOptions()
	o.Speaker, o.Instruct, o.Language = *speaker, *instruct, *language
	if *maxFrames > 0 {
		o.MaxFrames = *maxFrames
	}
	if *greedy {
		o.Talker.DoSample, o.Sub.DoSample = false, false
	}
	if m.Cfg.TTSModelType == "base" {
		if o.Clone, err = cloneVoice(m, *voice, *refAudio, *refText, *saveVoice); err != nil {
			fail(err)
		}
	}

	type job struct{ name, text, out string }
	var jobs []job
	switch {
	case *text != "":
		jobs = append(jobs, job{"", *text, *out})
	case *lines != "":
		if jobs2, err := readLines(*lines); err != nil {
			fail(err)
		} else {
			for _, j := range jobs2 {
				jobs = append(jobs, job{j[0], j[1], filepath.Join(*outDir, j[0]+".wav")})
			}
		}
		if err := os.MkdirAll(*outDir, 0o755); err != nil {
			fail(err)
		}
	case *saveVoice != "":
		return // only building the voice
	default:
		fmt.Fprintln(os.Stderr, "qwen3tts: -text or -lines is required")
		flag.Usage()
		os.Exit(2)
	}

	// One compiled Synth sized for the longest line.
	maxIDs := 0
	for _, j := range jobs {
		ids, err := m.Tok.Encode(qwen3tts.AssistantText(j.text))
		if err != nil {
			fail(err)
		}
		maxIDs = max(maxIDs, len(ids))
	}
	extra := 256 // instruction / reference rows
	if o.Clone != nil {
		extra += len(o.Clone.RefCodes) + 256
	}
	q := qwen3tts.Full
	if *int8w {
		q = qwen3tts.Int8
	}
	s, err := m.NewSynth(maxIDs+extra+o.MaxFrames, q, *device)
	if err != nil {
		fail(err)
	}
	defer s.Close()
	fmt.Fprintf(os.Stderr, "qwen3tts: %s %s on %s loaded in %.2f s\n", m.Cfg.TTSModelSize, m.Cfg.TTSModelType, s.Device(), time.Since(t0).Seconds())
	for i, j := range jobs {
		o.Talker.Seed, o.Sub.Seed = *seed, *seed+1
		o.Talker.RNG, o.Sub.RNG = nil, nil
		t1 := time.Now()
		res, err := s.Synthesize(j.text, o)
		if err != nil {
			fail(fmt.Errorf("%s: %w", j.out, err))
		}
		if err := writeWAV(j.out, res); err != nil {
			fail(err)
		}
		audio := float64(len(res.Wav)) / float64(res.SampleRate)
		gen := time.Since(t1).Seconds()
		fmt.Fprintf(os.Stderr, "qwen3tts: [%d/%d] %s: %.2f s of audio (%d frames) in %.2f s (%.2fx realtime)\n",
			i+1, len(jobs), j.out, audio, len(res.Frames), gen, audio/gen)
	}
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
		wav, rate, err := qwen3tts.ReadWAV(f)
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
		return nil, fmt.Errorf("a Base checkpoint clones a voice: pass -ref-audio (and -ref-text) or -voice")
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
	if err := qwen3tts.WriteWAV(f, res.Wav, res.SampleRate); err != nil {
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
		return "", fmt.Errorf("Qwen/%s not found under %s (download it, or pass a snapshot directory)", repo, hub)
	}
	return snaps[len(snaps)-1], nil
}

func fail(err error) {
	msg := err.Error()
	if !strings.HasPrefix(msg, "qwen3tts: ") {
		msg = "qwen3tts: " + msg
	}
	fmt.Fprintln(os.Stderr, msg)
	os.Exit(1)
}
