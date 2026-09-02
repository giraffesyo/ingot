package ocr

import (
	"image"
	"math"
	"os"
	"testing"
)

const (
	parseqModel   = "../../testdata/models/parseq_nar.onnx"
	parseqCharset = "../../testdata/models/parseq_charset.txt"
)

func loadParseq(t *testing.T) *Parseq {
	t.Helper()
	if _, err := os.Stat(parseqModel); err != nil {
		t.Skip("PARSeq model not present (run tools/export/parseq.py)")
	}
	p, err := NewParseq(parseqModel, parseqCharset)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// TestParseqDecode checks greedy decoding against hand-built logits: the
// argmax path, EOS termination (trailing steps ignored), and the confidence
// as the mean softmax max over the emitted steps including EOS.
func TestParseqDecode(t *testing.T) {
	p := &Parseq{charset: []rune("abc")}
	const T, C = 5, 4 // classes: EOS a b c
	logits := make([]float32, T*C)
	set := func(t, c int, v float32) { logits[t*C+c] = v }
	set(0, 2, 4) // b
	set(1, 1, 2) // a
	set(2, 3, 8) // c
	set(3, 0, 6) // EOS
	set(4, 1, 9) // ignored after EOS
	text, conf := p.decode(logits, T, C)
	if text != "bac" {
		t.Fatalf("text %q, want bac", text)
	}
	sm := func(v float32) float64 { return math.Exp(float64(v)) / (3 + math.Exp(float64(v))) }
	want := (sm(4) + sm(2) + sm(8) + sm(6)) / 4
	if math.Abs(conf-want) > 1e-9 {
		t.Fatalf("conf %v, want %v", conf, want)
	}
	// Ties resolve to the lowest class (EOS wins an all-zero row).
	text, _ = p.decode(make([]float32, T*C), T, C)
	if text != "" {
		t.Fatalf("all-zero logits decoded to %q", text)
	}
}

func TestParseqCharset(t *testing.T) {
	p := loadParseq(t)
	if n := len(p.charset); n != 94 {
		t.Fatalf("charset has %d chars, want 94", n)
	}
	if string(p.charset[:10]) != "0123456789" || p.charset[93] != '~' {
		t.Fatalf("charset order unexpected: %q", string(p.charset))
	}
}

// TestParseqBatch: each box recognised alone must decode identically inside a
// batch (no cross-crop leakage; batch invariance of the runtime).
func TestParseqBatch(t *testing.T) {
	pr := loadParseq(t)
	d, err := NewDetector(corpusDir + "/det.onnx")
	if err != nil {
		t.Skip(err)
	}
	f, err := os.Open(corpusDir + "/corpus/img_006.png")
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	boxes, err := d.Detect(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(boxes) < 3 {
		t.Fatalf("only %d boxes", len(boxes))
	}
	ts, cs, err := pr.RecognizeBatch(img, boxes)
	if err != nil {
		t.Fatal(err)
	}
	nonEmpty := 0
	for i, b := range boxes {
		s1, c1, err := pr.Recognize(img, b)
		if err != nil {
			t.Fatal(err)
		}
		if s1 != ts[i] || math.Abs(c1-cs[i]) > 1e-4 {
			t.Errorf("box %d: alone %q %.5f vs batched %q %.5f", i, s1, c1, ts[i], cs[i])
		}
		if s1 != "" {
			nonEmpty++
		}
		t.Logf("box %d: %q %.3f", i, s1, c1)
	}
	if nonEmpty < len(boxes)/2 {
		t.Errorf("only %d/%d boxes decoded to text", nonEmpty, len(boxes))
	}
}

// TestCorpusParseq scores PARSeq as the pipeline recogniser. The corpus is
// line-level with spaces and CJK, which the 94-char word model cannot express,
// so the score that matters is over the subset of lines within its charset;
// the full-corpus number is logged for the record. Gates are on the subset.
func TestCorpusParseq(t *testing.T) {
	corpus := loadCorpus(t)
	pr := loadParseq(t)
	d, err := NewDetector(corpusDir + "/det.onnx")
	if err != nil {
		t.Skip(err)
	}
	p := &Pipeline{Det: d, Rec: pr, RecBatch: DefaultRecBatch, RecPadRatio: DefaultRecPadRatio}
	inCharset := func(s string) bool {
		for _, r := range s {
			if r < '!' || r > '~' {
				return false
			}
		}
		return s != ""
	}
	all := evalCorpus(t, p, corpus, nil)
	all.log(t, "[all lines] ")
	sub := evalCorpus(t, p, corpus, inCharset)
	sub.log(t, "[charset-only lines] ")
	if sub.matched < 20 {
		t.Fatalf("only %d charset-only lines matched", sub.matched)
	}
	if sub.cer() > 0.15 {
		t.Errorf("charset-only CER %.3f > 0.15", sub.cer())
	}
	if sub.exactRate() < 0.6 {
		t.Errorf("charset-only exact-match %.3f < 0.6", sub.exactRate())
	}
}
