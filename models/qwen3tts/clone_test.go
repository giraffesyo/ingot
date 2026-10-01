package qwen3tts

import (
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/giraffesyo/ingot/generate"
)

// snapshot returns the local HF snapshot of repo (Qwen/<name>) or skips.
func snapshot(t testing.TB, name string) string {
	t.Helper()
	home, _ := os.UserHomeDir()
	snaps, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--Qwen--"+name+"/snapshots/*"))
	if len(snaps) == 0 {
		t.Skipf("%s not in the HF cache", name)
	}
	return snaps[0]
}

var (
	models   sync.Map // name → *Model
	modelsMu sync.Mutex
)

func loadNamed(t testing.TB, name string) *Model {
	t.Helper()
	dir := snapshot(t, name)
	modelsMu.Lock()
	defer modelsMu.Unlock()
	if m, ok := models.Load(name); ok {
		return m.(*Model)
	}
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	models.Store(name, m)
	return m
}

func greedy(o Options) Options {
	o.Talker = generate.Sampler{RepetitionPenalty: 1.05}
	o.Sub = generate.Sampler{}
	return o
}

func checkCodes(t *testing.T, got [][]int64, ref *refCase, name string) {
	t.Helper()
	codes, shape := ref.i64(t, name)
	want := frames(codes, shape)
	if len(got) != len(want) {
		t.Errorf("%d frames, reference %d", len(got), len(want))
	}
	for f := range min(len(got), len(want)) {
		for g := range want[f] {
			if got[f][g] != want[f][g] {
				t.Fatalf("frame %d codebook %d: %d, reference %d\n got %v\nwant %v", f, g, got[f][g], want[f][g], got[f], want[f])
			}
		}
	}
}

// TestDesignParity: 1.7B VoiceDesign, greedy, from a text description of
// the voice — instruction prompt, logits, exact codes, waveform.
func TestDesignParity(t *testing.T) {
	ref := loadRef(t, "design")
	m := loadNamed(t, "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
	s, err := m.NewSynth(512, Full, "cpu")
	if err != nil {
		t.Fatal(err)
	}
	ids, _ := ref.i64(t, "ids")
	o := greedy(m.DefaultOptions())
	o.Instruct, o.Language = ref.Meta["instruct"].(string), ref.Meta["language"].(string)
	o.MaxFrames = int(ref.Meta["frames"].(float64))
	prompt, _, err := s.Prompt(ids, o)
	if err != nil {
		t.Fatal(err)
	}
	check(t, "prompt", prompt, ref.f32(t, "embeds").F32(), 1e-5)
	got, res, err := s.Codes(ids, o)
	if err != nil {
		t.Fatal(err)
	}
	check(t, "prefill logits", res.PrefillLogits, ref.f32(t, "prefill_logits").F32(), 1e-4)
	var flat []float32
	for _, l := range res.FirstFrameSubLogits {
		flat = append(flat, l...)
	}
	check(t, "frame 0 code-predictor logits", flat, ref.f32(t, "cp_logits").F32(), 1e-4)
	checkCodes(t, got, ref, "codes")
	wav, err := s.codec.Decode(got)
	if err != nil {
		t.Fatal(err)
	}
	check(t, "wav", wav, ref.f32(t, "wav").F32(), 1e-3)
}

// TestCloneParity: 1.7B Base, in-context cloning of the design waveform —
// mel, speaker x-vector, reference codes, prompt + trailing text, exact
// greedy codes, waveform with the reference cut off.
func TestCloneParity(t *testing.T) {
	ref := loadRef(t, "clone")
	m := loadNamed(t, "Qwen3-TTS-12Hz-1.7B-Base")
	cl, err := m.NewCloner()
	if err != nil {
		t.Fatal(err)
	}
	refWav := ref.f32(t, "ref_wav").F32()
	mel, err := cl.Mel(refWav)
	if err != nil {
		t.Fatal(err)
	}
	check(t, "mel", mel.F32(), ref.f32(t, "mel").F32(), 1e-4)
	vp, err := cl.NewVoicePrompt(refWav, 24000, ref.Meta["ref_text"].(string))
	if err != nil {
		t.Fatal(err)
	}
	check(t, "x-vector", vp.SpeakerEmbed, ref.f32(t, "xvector").F32(), 1e-4)
	checkCodes(t, vp.RefCodes, ref, "ref_codes")

	s, err := m.NewSynth(512, Full, "cpu")
	if err != nil {
		t.Fatal(err)
	}
	ids, _ := ref.i64(t, "ids")
	o := greedy(m.DefaultOptions())
	o.Clone, o.Language = vp, ref.Meta["language"].(string)
	o.MaxFrames = int(ref.Meta["frames"].(float64))
	prompt, trailing, err := s.Prompt(ids, o)
	if err != nil {
		t.Fatal(err)
	}
	check(t, "prompt", prompt, ref.f32(t, "embeds").F32(), 1e-5)
	check(t, "trailing", trailing, ref.f32(t, "trailing").F32(), 1e-5)
	res, err := s.Synthesize(ref.Meta["text"].(string), o)
	if err != nil {
		t.Fatal(err)
	}
	check(t, "prefill logits", res.PrefillLogits, ref.f32(t, "prefill_logits").F32(), 1e-4)
	checkCodes(t, res.Frames, ref, "codes")
	check(t, "wav", res.Wav, ref.f32(t, "wav").F32(), 1e-3)
}

// TestCloneBranches: the other two prompt layouts of cloning — a target
// text longer than the reference's codes (the surplus streams in as
// trailing rows) and x-vector-only cloning (no reference codes) — against
// the reference prompt, trailing rows and greedy codes.
func TestCloneBranches(t *testing.T) {
	m := loadNamed(t, "Qwen3-TTS-12Hz-1.7B-Base")
	base := loadRef(t, "clone")
	cl, err := m.NewCloner()
	if err != nil {
		t.Fatal(err)
	}
	s, err := m.NewSynth(512, Full, "cpu")
	if err != nil {
		t.Fatal(err)
	}
	refWav := base.f32(t, "ref_wav").F32()
	for _, name := range []string{"clone_long", "clone_xvec"} {
		t.Run(name, func(t *testing.T) {
			ref := loadRef(t, name)
			vp, err := cl.NewVoicePrompt(refWav, 24000, ref.Meta["ref_text"].(string))
			if err != nil {
				t.Fatal(err)
			}
			ids, _ := ref.i64(t, "ids")
			o := greedy(m.DefaultOptions())
			o.Clone, o.Language = vp, ref.Meta["language"].(string)
			o.MaxFrames = int(ref.Meta["frames"].(float64))
			prompt, trailing, err := s.Prompt(ids, o)
			if err != nil {
				t.Fatal(err)
			}
			check(t, "prompt", prompt, ref.f32(t, "embeds").F32(), 1e-5)
			// Ours ends with the tts_pad row the reference adds once its
			// trailing rows run out.
			want := ref.f32(t, "trailing").F32()
			check(t, "trailing", trailing[:len(want)], want, 1e-5)
			got, _, err := s.Codes(ids, o)
			if err != nil {
				t.Fatal(err)
			}
			checkCodes(t, got, ref, "codes")
		})
	}
}

// TestInt8Close: the int8 decode loop stays close to full precision on the
// 1.7B VoiceDesign case: first-frame code-predictor logits within 3% per
// codebook while the codes they condition on agree (measured ~1%: the int8
// weight rounding), and the first code 0 identical (the prompt prefill
// stays bf16). A drift guard, not parity.
func TestInt8Close(t *testing.T) {
	ref := loadRef(t, "design")
	m := loadNamed(t, "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
	ids, _ := ref.i64(t, "ids")
	o := greedy(m.DefaultOptions())
	o.Instruct, o.Language, o.MaxFrames = ref.Meta["instruct"].(string), "english", int(ref.Meta["frames"].(float64))
	s, err := m.NewSynth(512, Int8, "cpu")
	if err != nil {
		t.Fatal(err)
	}
	got, res, err := s.Codes(ids, o)
	if err != nil {
		t.Fatal(err)
	}
	codes, shape := ref.i64(t, "codes")
	want := frames(codes, shape)
	// Codebook g's logits are conditioned on codes 0..g-1: compare them
	// only while those agree.
	cp := ref.f32(t, "cp_logits").F32()
	for g, l := range res.FirstFrameSubLogits {
		if got[0][g] != want[0][g] {
			break
		}
		_, rel := relErr(l, cp[g*len(l):(g+1)*len(l)])
		if rel > 0.03 {
			t.Errorf("int8 codebook %d logits drifted: rel err %.3g", g+1, rel)
		}
	}
	t.Logf("int8: %d frames (full precision %d); frame 0 codes %v vs %v", len(got), len(want), got[0], want[0])
	if got[0][0] != want[0][0] {
		t.Errorf("first code 0 differs: prefill should be full precision")
	}
}

// TestGPUParity: the Metal decode loop (full precision) on the 1.7B design
// and clone references — prefill logits, first-frame code-predictor logits
// and greedy codes. The GPU sums in a different order, so logits are
// compared numerically; codes are expected to match exactly.
func TestGPUParity(t *testing.T) {
	if !metalAvailable() {
		t.Skip("no Metal device")
	}
	t.Run("design", func(t *testing.T) {
		ref := loadRef(t, "design")
		m := loadNamed(t, "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
		s, err := m.NewSynth(512, Full, "gpu")
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		ids, _ := ref.i64(t, "ids")
		o := greedy(m.DefaultOptions())
		o.Instruct, o.Language = ref.Meta["instruct"].(string), ref.Meta["language"].(string)
		o.MaxFrames = int(ref.Meta["frames"].(float64))
		got, res, err := s.Codes(ids, o)
		if err != nil {
			t.Fatal(err)
		}
		check(t, "prefill logits", res.PrefillLogits, ref.f32(t, "prefill_logits").F32(), 1e-4)
		var flat []float32
		for _, l := range res.FirstFrameSubLogits {
			flat = append(flat, l...)
		}
		check(t, "frame 0 code-predictor logits", flat, ref.f32(t, "cp_logits").F32(), 1e-4)
		checkCodes(t, got, ref, "codes")
	})
	t.Run("clone", func(t *testing.T) {
		ref := loadRef(t, "clone")
		m := loadNamed(t, "Qwen3-TTS-12Hz-1.7B-Base")
		cl, err := m.NewCloner()
		if err != nil {
			t.Fatal(err)
		}
		vp, err := cl.NewVoicePrompt(ref.f32(t, "ref_wav").F32(), 24000, ref.Meta["ref_text"].(string))
		if err != nil {
			t.Fatal(err)
		}
		s, err := m.NewSynth(512, Full, "gpu")
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		ids, _ := ref.i64(t, "ids")
		o := greedy(m.DefaultOptions())
		o.Clone, o.Language = vp, ref.Meta["language"].(string)
		o.MaxFrames = int(ref.Meta["frames"].(float64))
		got, res, err := s.Codes(ids, o)
		if err != nil {
			t.Fatal(err)
		}
		check(t, "prefill logits", res.PrefillLogits, ref.f32(t, "prefill_logits").F32(), 1e-4)
		checkCodes(t, got, ref, "codes")
	})
}
