package tokenizer

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
)

// spProto hand-encodes a sentencepiece.ModelProto for the tests.
type spProto struct{ b []byte }

func (p *spProto) tag(field, wt int) { p.b = binary.AppendUvarint(p.b, uint64(field<<3|wt)) }
func (p *spProto) varint(field int, v uint64) {
	p.tag(field, 0)
	p.b = binary.AppendUvarint(p.b, v)
}
func (p *spProto) bytes(field int, b []byte) {
	p.tag(field, 2)
	p.b = binary.AppendUvarint(p.b, uint64(len(b)))
	p.b = append(p.b, b...)
}
func (p *spProto) piece(s string, score float32, kind int) {
	var m spProto
	m.bytes(1, []byte(s))
	m.tag(2, 5)
	m.b = binary.LittleEndian.AppendUint32(m.b, math.Float32bits(score))
	m.varint(3, uint64(kind))
	p.bytes(1, m.b)
}

func tinySP(t *testing.T, modelType uint64, dummyPrefix bool, charsmap []byte) []byte {
	t.Helper()
	var p spProto
	p.piece("<pad>", 0, spControl)
	p.piece("<unk>", 0, spUnknown)
	p.piece("<br>", 0, spUserDefined)
	for i := 0; i < 256; i++ {
		p.piece("<0x"+strings.ToUpper(string("0123456789abcdef"[i>>4]))+strings.ToUpper(string("0123456789abcdef"[i&15]))+">", 0, spByte)
	}
	for _, c := range []string{"▁", "a", "b", "c", "d", "r"} {
		p.piece(c, -100, spNormal)
	}
	// merges, best score first
	p.piece("ab", -1, spNormal)
	p.piece("▁ab", -2, spNormal)
	p.piece("ra", -3, spNormal)
	p.piece("▁abra", -4, spNormal)
	p.piece("ca", -5, spNormal)
	p.piece("bra", -6, spUnused) // never produced
	var tr spProto
	tr.varint(3, modelType)
	tr.varint(35, 1) // byte_fallback
	p.bytes(2, tr.b)
	var nm spProto
	if len(charsmap) > 0 {
		nm.bytes(2, charsmap)
	}
	if !dummyPrefix {
		nm.varint(3, 0)
	}
	nm.varint(4, 0) // keep extra whitespace
	p.bytes(3, nm.b)
	return p.b
}

func spPieces(sp *SentencePiece, ids []int) []string {
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = sp.Piece(id)
	}
	return out
}

func TestSentencePieceBPE(t *testing.T) {
	sp, err := ParseSentencePiece(tinySP(t, 2, false, nil))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		in   string
		want []string
	}{
		// "ab" merges first everywhere (best score), then "▁ab", "ra", "▁abra".
		{"abra", []string{"ab", "ra"}},
		{" abra", []string{"▁abra"}},
		{"ab abra cab", []string{"ab", "▁abra", "▁", "c", "ab"}},
		// a user-defined symbol is kept whole and blocks merges across it
		{"a<br>b", []string{"a", "<br>", "b"}},
		// unknown characters fall back to their UTF-8 bytes
		{"é", []string{"<0xC3>", "<0xA9>"}},
		{"", nil},
	} {
		got := spPieces(sp, sp.Encode(c.in))
		if len(got) == 0 && len(c.want) == 0 {
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("Encode(%q) = %q, want %q", c.in, got, c.want)
		}
	}
	// With the dummy prefix, text starts a word.
	sp, err = ParseSentencePiece(tinySP(t, 2, true, nil))
	if err != nil {
		t.Fatal(err)
	}
	if got := spPieces(sp, sp.Encode("abra")); !reflect.DeepEqual(got, []string{"▁abra"}) {
		t.Errorf("dummy prefix: %q", got)
	}
}

func TestSentencePieceRejects(t *testing.T) {
	if _, err := ParseSentencePiece(tinySP(t, 1, false, nil)); err == nil {
		t.Error("a unigram model loaded")
	}
	if _, err := ParseSentencePiece(tinySP(t, 2, false, []byte{1, 2, 3})); err == nil {
		t.Error("a model with a charsmap loaded")
	}
	if _, err := ParseSentencePiece([]byte{0x0a, 0x7f}); err == nil {
		t.Error("a truncated model loaded")
	}
}

// TestSentencePieceMatchesTokenizerJSON checks the score-ordered merges
// against an independent description of the same tokenizer: the Hugging
// Face tokenizer.json's merge list, applied by rank. Set INGOT_SP_MODEL and
// INGOT_SP_JSON to a model's tokenizer.model and tokenizer.json.
func TestSentencePieceMatchesTokenizerJSON(t *testing.T) {
	modelPath, jsonPath := os.Getenv("INGOT_SP_MODEL"), os.Getenv("INGOT_SP_JSON")
	if modelPath == "" || jsonPath == "" {
		t.Skip("INGOT_SP_MODEL / INGOT_SP_JSON not set")
	}
	sp, err := LoadSentencePiece(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatal(err)
	}
	var tj struct {
		Model struct {
			Type   string         `json:"type"`
			Vocab  map[string]int `json:"vocab"`
			Merges []any          `json:"merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &tj); err != nil {
		t.Fatal(err)
	}
	if tj.Model.Type != "BPE" {
		t.Fatalf("tokenizer.json model type %q", tj.Model.Type)
	}
	rank := map[[2]string]int{}
	for i, m := range tj.Model.Merges {
		var a, b string
		switch v := m.(type) {
		case string:
			a, b, _ = strings.Cut(v, " ")
		case []any:
			a, b = v[0].(string), v[1].(string)
		}
		if _, ok := rank[[2]string{a, b}]; !ok {
			rank[[2]string{a, b}] = i
		}
	}
	byRank := func(text string) []int {
		text = strings.ReplaceAll(text, " ", spSpace)
		var syms []string
		for _, r := range text {
			syms = append(syms, string(r))
		}
		for {
			best, bestRank := -1, len(rank)
			for i := 0; i+1 < len(syms); i++ {
				if r, ok := rank[[2]string{syms[i], syms[i+1]}]; ok && r < bestRank {
					best, bestRank = i, r
				}
			}
			if best < 0 {
				break
			}
			syms[best] += syms[best+1]
			syms = append(syms[:best+1], syms[best+2:]...)
		}
		ids := make([]int, len(syms))
		for i, s := range syms {
			id, ok := tj.Model.Vocab[s]
			if !ok {
				t.Fatalf("%q: piece %q not in tokenizer.json vocab", text, s)
			}
			ids[i] = id
		}
		return ids
	}
	for _, text := range []string{
		"A dog barking in the distance",
		"wings flapping as a flock of doves takes off",
		"heavy canvas banner flapping once in the wind, close up",
		"a single cat meow, dry studio recording",
		"water splashing in a stone fountain",
		"Rocks tumbling down a cliff, debris and gravel, 3 seconds",
		"an owl hooting twice at night",
		"ceramic jar tapped with a knuckle; hollow clay sound",
	} {
		got, want := sp.Encode(text), byRank(text)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%q:\n scores %v %q\n merges %v", text, got, spPieces(sp, got), want)
		}
	}
	t.Logf("vocab %d pieces, dummy prefix %v, trim %v, byte fallback %v, %d merges", sp.Size(), sp.dummyPrefix, sp.trimSpaces, sp.byteFallback, len(rank))
}
