package qwen3tts

import (
	"fmt"
	"os"
	"testing"

	"github.com/giraffesyo/ingot/generate"
	"github.com/giraffesyo/ingot/graph"
)

// BenchmarkCodes/frames=47: the greedy parity utterance's talker + code
// predictor loop (prefill + 47 frames); reported per frame.
func BenchmarkCodes(b *testing.B) {
	ref := loadRef(b, "greedy")
	s := loadSynth(b)
	ids, _ := ref.i64(b, "ids")
	o := s.m.DefaultOptions()
	o.Speaker, o.Language, o.MaxFrames = "ryan", "english", 48
	o.Talker, o.Sub = generate.Sampler{RepetitionPenalty: 1.05}, generate.Sampler{}
	b.Run("frames=47", func(b *testing.B) {
		var n int
		for b.Loop() {
			f, _, err := s.Codes(ids, o)
			if err != nil {
				b.Fatal(err)
			}
			n = len(f)
		}
		b.ReportMetric(float64(b.Elapsed().Milliseconds())/float64(b.N*n), "ms/frame")
	})
	if os.Getenv("QWEN3TTS_PROFILE") != "" {
		// Per-op time for one more run (profiling perturbs the timing above).
		s.talker.(*cpuLM).sess.Profile, s.cp.(*cpuLM).sess.Profile = true, true
		if _, _, err := s.Codes(ids, o); err != nil {
			b.Fatal(err)
		}
		for name, sess := range map[string]interface {
			Stats() []graph.OpStat
		}{"talker": s.talker.(*cpuLM).sess, "code predictor": s.cp.(*cpuLM).sess} {
			fmt.Printf("%s:\n", name)
			for _, st := range sess.Stats() {
				fmt.Printf("  %-22s %4d nodes %8.2f ms\n", st.OpType, st.Count, float64(st.Total.Microseconds())/1000)
			}
		}
		s.talker.(*cpuLM).sess.Profile, s.cp.(*cpuLM).sess.Profile = false, false
	}
}

// BenchmarkCodec/frames=47: the speech-tokenizer decoder on 47 frames
// (3.76 s of audio).
func BenchmarkCodec(b *testing.B) {
	ref := loadRef(b, "decode")
	m := loadModel(b)
	cd, err := m.NewCodec("cpu")
	if err != nil {
		b.Fatal(err)
	}
	codes, shape := ref.i64(b, "codes")
	fr := frames(codes, shape)
	b.Run("frames=47", func(b *testing.B) {
		for b.Loop() {
			if _, err := cd.Decode(fr); err != nil {
				b.Fatal(err)
			}
		}
	})
}
