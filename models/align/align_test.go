package align

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const refDir = "../../testdata/align"

type clipRef struct {
	Meta struct {
		Text  string   `json:"text"`
		Words []string `json:"words"`
		Spans [][2]int `json:"spans"`
	} `json:"meta"`
	Tensors []struct {
		Name, File string
		Shape      []int
	} `json:"tensors"`
}

func loadClip(t *testing.T) (*clipRef, map[string][]byte) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(refDir, "clip.json"))
	if err != nil {
		t.Skip("aligner references not generated (tools/export/align_ref.py)")
	}
	var c clipRef
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	data := map[string][]byte{}
	for _, tn := range c.Tensors {
		b, err := os.ReadFile(filepath.Join(refDir, tn.File))
		if err != nil {
			t.Fatal(err)
		}
		data[tn.Name] = b
	}
	return &c, data
}

func vocabPath(t *testing.T) string {
	home, _ := os.UserHomeDir()
	m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--facebook--wav2vec2-base-960h/snapshots/*/vocab.json"))
	if len(m) == 0 {
		t.Skip("facebook/wav2vec2-base-960h not in the HF cache")
	}
	return m[0]
}

func f32s(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

// TestAlignParity: the ONNX model through ingot reproduces PyTorch's CTC
// log-probabilities, the Viterbi path matches torchaudio's forced_align
// frame for frame, and the word spans match.
func TestAlignParity(t *testing.T) {
	c, data := loadClip(t)
	a, err := Load(filepath.Join(refDir, "wav2vec2-base-960h.onnx"), vocabPath(t))
	if err != nil {
		t.Fatal(err)
	}
	x := f32s(data["x"]) // already normalised: normalising again is a no-op
	lp, err := a.LogProbs(x, SampleRate)
	if err != nil {
		t.Fatal(err)
	}
	want := f32s(data["logp"])
	V := len(lp[0])
	if len(lp)*V != len(want) {
		t.Fatalf("%d frames × %d, reference %d values", len(lp), V, len(want))
	}
	// Relative to the log-prob's size: unlikely characters sit near -20,
	// where 12 f32 transformer layers move the last digits.
	var maxRel, maxLikely float64
	for f := range lp {
		for i := range V {
			ref := float64(want[f*V+i])
			d := math.Abs(float64(lp[f][i]) - ref)
			maxRel = math.Max(maxRel, d/math.Max(1, math.Abs(ref)))
			if ref > -5 {
				maxLikely = math.Max(maxLikely, d)
			}
		}
	}
	t.Logf("log-prob max rel err %.3g; abs err on likely tokens (> -5) %.3g", maxRel, maxLikely)
	if maxRel > 1e-3 {
		t.Errorf("log-probs differ: rel %.3g", maxRel)
	}
	targets, words := a.Tokens(c.Meta.Text)
	path, err := ForcedAlign(lp, targets, a.blank)
	if err != nil {
		t.Fatal(err)
	}
	ref := data["path"]
	for f := range path {
		tok := int(int64(binary.LittleEndian.Uint64(ref[8*f:])))
		got := a.blank
		if path[f] >= 0 {
			got = targets[path[f]]
		}
		if got != tok {
			t.Fatalf("frame %d: token %d, torchaudio %d", f, got, tok)
		}
	}
	spans := wordSpans(path, targets, words, a.sep)
	if len(spans) != len(c.Meta.Spans) {
		t.Fatalf("%d words, reference %d", len(spans), len(c.Meta.Spans))
	}
	for i, s := range spans {
		if s.FrameStart != c.Meta.Spans[i][0] || s.FrameEnd != c.Meta.Spans[i][1] {
			t.Errorf("%s: frames [%d,%d), torchaudio [%d,%d)", s.Text, s.FrameStart, s.FrameEnd, c.Meta.Spans[i][0], c.Meta.Spans[i][1])
		}
	}
}

// TestForcedAlignOracle: Viterbi against brute force on tiny problems —
// every valid CTC labelling enumerated, the best score must match ours.
func TestForcedAlignOracle(t *testing.T) {
	logp := [][]float32{
		{-0.1, -3, -2}, {-2, -0.2, -3}, {-1, -1, -1.5}, {-3, -2, -0.1}, {-0.5, -2, -1},
	}
	for _, targets := range [][]int{{1, 2}, {1, 1}, {2}, {1, 2, 1}} {
		path, err := ForcedAlign(logp, targets, 0)
		if err != nil {
			t.Fatal(err)
		}
		score := func(p []int) float64 {
			var s float64
			for f, ti := range p {
				tok := 0
				if ti >= 0 {
					tok = targets[ti]
				}
				s += float64(logp[f][tok])
			}
			return s
		}
		// Enumerate labellings: each frame blank (-1) or a target index, valid
		// when indices are non-decreasing by at most 1, start at 0, end at L-1,
		// and a repeat of the same label is separated by a blank.
		best := math.Inf(-1)
		var rec func(f, next int, prevTok int, p []int)
		rec = func(f, next, prevTok int, p []int) {
			if f == len(logp) {
				if next == len(targets) {
					best = math.Max(best, score(p))
				}
				return
			}
			rec(f+1, next, -1, append(p, -1)) // blank
			if next > 0 && prevTok == next-1 {
				rec(f+1, next, next-1, append(p, next-1)) // stay on the label
			}
			// Advance to the next label — directly from the previous one only
			// when the two differ (a repeat needs a blank between).
			if next < len(targets) && !(next > 0 && prevTok == next-1 && targets[next] == targets[next-1]) {
				rec(f+1, next+1, next, append(p, next))
			}
		}
		rec(0, 0, -1, nil)
		if math.Abs(score(path)-best) > 1e-9 {
			t.Errorf("targets %v: Viterbi score %g, brute force %g (path %v)", targets, score(path), best, path)
		}
	}
}
