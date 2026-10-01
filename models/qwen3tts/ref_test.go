package qwen3tts

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/giraffesyo/ingot/generate"
	"github.com/giraffesyo/ingot/tensor"
)

// snapshotDir returns the local Qwen3-TTS 0.6B CustomVoice snapshot or skips.
func snapshotDir(t testing.TB) string {
	t.Helper()
	if d := os.Getenv("QWEN3TTS_DIR"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	snaps, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/*"))
	if len(snaps) == 0 {
		t.Skip("Qwen3-TTS-12Hz-0.6B-CustomVoice not in the HF cache (set QWEN3TTS_DIR)")
	}
	return snaps[0]
}

var (
	modelOnce sync.Once
	model     *Model
	modelErr  error
)

func loadModel(t testing.TB) *Model {
	t.Helper()
	dir := snapshotDir(t)
	modelOnce.Do(func() { model, modelErr = Load(dir) })
	if modelErr != nil {
		t.Fatal(modelErr)
	}
	return model
}

const refDir = "../../testdata/qwen3tts"

// refCase is a manifest from tools/export/qwen3tts_ref.py.
type refCase struct {
	Meta    map[string]any `json:"meta"`
	Inputs  []refTensor    `json:"inputs"`
	Outputs []refTensor    `json:"outputs"`
}

type refTensor struct {
	Name  string `json:"name"`
	File  string `json:"file"`
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
}

func loadRef(t testing.TB, name string) *refCase {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(refDir, name+".json"))
	if err != nil {
		t.Skipf("reference %s not generated (tools/export/qwen3tts_ref.py): %v", name, err)
	}
	var c refCase
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	return &c
}

func (c *refCase) find(t testing.TB, name string) (refTensor, []byte) {
	t.Helper()
	for _, r := range append(append([]refTensor{}, c.Inputs...), c.Outputs...) {
		if r.Name == name {
			raw, err := os.ReadFile(filepath.Join(refDir, r.File))
			if err != nil {
				t.Fatal(err)
			}
			return r, raw
		}
	}
	t.Fatalf("reference has no %q", name)
	return refTensor{}, nil
}

func (c *refCase) f32(t testing.TB, name string) *tensor.Tensor {
	t.Helper()
	r, raw := c.find(t, name)
	if r.DType != "float32" {
		t.Fatalf("%s: dtype %s", name, r.DType)
	}
	out := tensor.New(tensor.F32, r.Shape...)
	for i := range out.F32() {
		out.F32()[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}

func (c *refCase) i64(t testing.TB, name string) ([]int64, []int) {
	t.Helper()
	r, raw := c.find(t, name)
	if r.DType != "int64" {
		t.Fatalf("%s: dtype %s", name, r.DType)
	}
	v := make([]int64, len(raw)/8)
	for i := range v {
		v[i] = int64(binary.LittleEndian.Uint64(raw[8*i:]))
	}
	return v, r.Shape
}

// relErr is max|a-b| / max|b|.
func relErr(a, b []float32) (maxAbs, rel float64) {
	var ref float64
	for i := range b {
		maxAbs = math.Max(maxAbs, math.Abs(float64(a[i]-b[i])))
		ref = math.Max(ref, math.Abs(float64(b[i])))
	}
	return maxAbs, maxAbs / math.Max(ref, 1e-30)
}

func check(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d want %d", name, len(got), len(want))
	}
	abs, rel := relErr(got, want)
	t.Logf("%s: max abs err %.3g (rel %.3g)", name, abs, rel)
	if rel > tol {
		t.Errorf("%s: rel err %.3g > %.3g", name, rel, tol)
	}
}

var (
	synthOnce sync.Once
	synth     *Synth
	synthErr  error
)

func loadSynth(t testing.TB) *Synth {
	t.Helper()
	m := loadModel(t)
	synthOnce.Do(func() { synth, synthErr = m.NewSynth(512, Full, "cpu") })
	if synthErr != nil {
		t.Fatal(synthErr)
	}
	return synth
}

func frames(codes []int64, shape []int) [][]int64 {
	out := make([][]int64, shape[0])
	for i := range out {
		out[i] = codes[i*shape[1] : (i+1)*shape[1]]
	}
	return out
}

// TestPromptParity: the talker's prefill embeddings (text projection +
// codec tags + speaker) against the reference.
func TestPromptParity(t *testing.T) {
	ref := loadRef(t, "prompt")
	s := loadSynth(t)
	ids, _ := ref.i64(t, "ids")
	want := ref.f32(t, "embeds")
	o := s.m.DefaultOptions()
	o.Speaker, o.Language = ref.Meta["speaker"].(string), ref.Meta["language"].(string)
	got, _, err := s.Prompt(ids, o)
	if err != nil {
		t.Fatal(err)
	}
	check(t, "prompt", got, want.F32(), 1e-5)
	// The template tokenisation itself.
	enc, err := s.m.Tok.Encode(AssistantText(ref.Meta["text"].(string)))
	if err != nil {
		t.Fatal(err)
	}
	if len(enc) != len(ids) {
		t.Fatalf("tokenised %d ids, reference %d", len(enc), len(ids))
	}
	for i := range ids {
		if enc[i] != ids[i] {
			t.Fatalf("token %d: %d vs reference %d", i, enc[i], ids[i])
		}
	}
}

// TestGreedyParity: greedy talker + code predictor (processors still on)
// reproduce the reference codes exactly; prefill and first-frame logits
// match numerically.
func TestGreedyParity(t *testing.T) {
	ref := loadRef(t, "greedy")
	s := loadSynth(t)
	ids, _ := ref.i64(t, "ids")
	o := s.m.DefaultOptions()
	o.Speaker, o.Language = ref.Meta["speaker"].(string), ref.Meta["language"].(string)
	o.MaxFrames = int(ref.Meta["frames"].(float64))
	o.Talker = generate.Sampler{RepetitionPenalty: 1.05}
	o.Sub = generate.Sampler{}
	got, res, err := s.Codes(ids, o)
	if err != nil {
		t.Fatal(err)
	}
	check(t, "prefill logits", res.PrefillLogits, ref.f32(t, "prefill_logits").F32(), 1e-4)
	cp := ref.f32(t, "cp_logits")
	var flat []float32
	for _, l := range res.FirstFrameSubLogits {
		flat = append(flat, l...)
	}
	check(t, "frame 0 code-predictor logits", flat, cp.F32(), 1e-4)
	codes, shape := ref.i64(t, "codes")
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

// TestCodecParity: the speech-tokenizer decoder on the reference codes —
// RVQ sum, pre-transformer output, waveform.
func TestCodecParity(t *testing.T) {
	ref := loadRef(t, "decode")
	m := loadModel(t)
	cd, err := m.NewCodec("cpu")
	if err != nil {
		t.Fatal(err)
	}
	codes, shape := ref.i64(t, "codes")
	fr := frames(codes, shape)
	out, err := cd.run(fr)
	if err != nil {
		t.Fatal(err)
	}
	check(t, "rvq", out["rvq"].F32(), ref.f32(t, "rvq").F32(), 1e-5)
	check(t, "pre_transformer", out["pre_tf"].F32(), ref.f32(t, "pre_tf").F32(), 1e-4)
	check(t, "wav", out["wav"].F32(), ref.f32(t, "wav").F32(), 1e-3)
	wav, err := cd.Decode(fr)
	if err != nil {
		t.Fatal(err)
	}
	check(t, "wav (Decode)", wav, ref.f32(t, "wav_trimmed").F32(), 1e-3)
}

// TestCodecGPUParity: the codec decoder through the GPU executor against
// the reference waveform.
func TestCodecGPUParity(t *testing.T) {
	if !metalAvailable() {
		t.Skip("no Metal device")
	}
	ref := loadRef(t, "decode")
	m := loadModel(t)
	cd, err := m.NewCodec("gpu")
	if err != nil {
		t.Fatal(err)
	}
	codes, shape := ref.i64(t, "codes")
	wav, err := cd.Decode(frames(codes, shape))
	if err != nil {
		t.Fatal(err)
	}
	check(t, "wav (GPU)", wav, ref.f32(t, "wav_trimmed").F32(), 1e-3)
}
