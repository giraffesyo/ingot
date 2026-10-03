package tokenizer

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"unicode/utf8"
)

// SentencePiece is a SentencePiece BPE model read from its .model file (a
// sentencepiece.ModelProto), as Gemma and T5Gemma ship: identity
// normalisation, spaces escaped to "▁", user-defined symbols kept whole,
// score-ordered merges and byte fallback for characters outside the
// vocabulary. Unigram models and models with a normalisation charsmap are
// rejected at load time rather than tokenised approximately.
type SentencePiece struct {
	pieces []string
	scores []float32
	kinds  []int32
	ids    map[string]int
	// user holds the user-defined symbols by first byte, longest first.
	user map[byte][]string

	unk          int
	byteFallback bool
	dummyPrefix  bool
	trimSpaces   bool
	escape       bool
}

// SentencePiece piece types (sentencepiece_model.proto).
const (
	spNormal      = 1
	spUnknown     = 2
	spControl     = 3
	spUserDefined = 4
	spUnused      = 5
	spByte        = 6
)

const spSpace = "▁"

// LoadSentencePiece reads a SentencePiece .model file.
func LoadSentencePiece(path string) (*SentencePiece, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	sp, err := ParseSentencePiece(b)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	return sp, nil
}

// ParseSentencePiece decodes a serialized sentencepiece.ModelProto.
func ParseSentencePiece(b []byte) (*SentencePiece, error) {
	sp := &SentencePiece{ids: map[string]int{}, user: map[byte][]string{}, dummyPrefix: true, trimSpaces: true, escape: true}
	modelType := int64(1) // UNIGRAM is the proto default
	var charsmap []byte
	err := spFields(b, func(field, wt int, v uint64, body []byte) error {
		switch field {
		case 1: // pieces
			piece, score, kind := "", float32(0), int32(spNormal)
			if err := spFields(body, func(f, wt int, v uint64, body []byte) error {
				switch f {
				case 1:
					piece = string(body)
				case 2:
					score = math.Float32frombits(uint32(v))
				case 3:
					kind = int32(v)
				}
				return nil
			}); err != nil {
				return err
			}
			if _, dup := sp.ids[piece]; !dup {
				sp.ids[piece] = len(sp.pieces)
			}
			sp.pieces = append(sp.pieces, piece)
			sp.scores = append(sp.scores, score)
			sp.kinds = append(sp.kinds, kind)
		case 2: // trainer_spec
			return spFields(body, func(f, wt int, v uint64, body []byte) error {
				switch f {
				case 3:
					modelType = int64(v)
				case 35:
					sp.byteFallback = v != 0
				}
				return nil
			})
		case 3: // normalizer_spec
			return spFields(body, func(f, wt int, v uint64, body []byte) error {
				switch f {
				case 2:
					charsmap = body
				case 3:
					sp.dummyPrefix = v != 0
				case 4:
					sp.trimSpaces = v != 0
				case 5:
					sp.escape = v != 0
				}
				return nil
			})
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(sp.pieces) == 0 {
		return nil, errors.New("sentencepiece model has no pieces")
	}
	if modelType != 2 {
		return nil, fmt.Errorf("sentencepiece model type %d is not BPE (2)", modelType)
	}
	if len(charsmap) != 0 {
		return nil, errors.New("sentencepiece model has a normalisation charsmap; only identity normalisation is supported")
	}
	sp.unk = -1
	for i, k := range sp.kinds {
		switch k {
		case spUnknown:
			if sp.unk < 0 {
				sp.unk = i
			}
		case spUserDefined:
			p := sp.pieces[i]
			if p != "" {
				sp.user[p[0]] = append(sp.user[p[0]], p)
			}
		}
	}
	if sp.unk < 0 {
		return nil, errors.New("sentencepiece model has no unknown piece")
	}
	for c, list := range sp.user {
		// longest first, so the prefix match below is the longest match
		for i := 1; i < len(list); i++ {
			for j := i; j > 0 && len(list[j]) > len(list[j-1]); j-- {
				list[j], list[j-1] = list[j-1], list[j]
			}
		}
		sp.user[c] = list
	}
	return sp, nil
}

// spFields walks one protobuf message, calling fn per field with its varint
// or fixed value (v) or its length-delimited body.
func spFields(b []byte, fn func(field, wt int, v uint64, body []byte) error) error {
	trunc := errors.New("truncated sentencepiece model")
	for len(b) > 0 {
		tag, n := binary.Uvarint(b)
		if n <= 0 {
			return trunc
		}
		b = b[n:]
		field, wt := int(tag>>3), int(tag&7)
		var v uint64
		var body []byte
		switch wt {
		case 0:
			v, n = binary.Uvarint(b)
			if n <= 0 {
				return trunc
			}
			b = b[n:]
		case 1:
			if len(b) < 8 {
				return trunc
			}
			v, b = binary.LittleEndian.Uint64(b), b[8:]
		case 2:
			l, n := binary.Uvarint(b)
			if n <= 0 || uint64(len(b)-n) < l {
				return trunc
			}
			body, b = b[n:n+int(l)], b[n+int(l):]
		case 5:
			if len(b) < 4 {
				return trunc
			}
			v, b = uint64(binary.LittleEndian.Uint32(b)), b[4:]
		default:
			return fmt.Errorf("sentencepiece model: wire type %d", wt)
		}
		if err := fn(field, wt, v, body); err != nil {
			return err
		}
	}
	return nil
}

// Size is the vocabulary size.
func (sp *SentencePiece) Size() int { return len(sp.pieces) }

// Piece is the text of a piece id.
func (sp *SentencePiece) Piece(id int) string { return sp.pieces[id] }

// ID is a piece's id, or -1.
func (sp *SentencePiece) ID(piece string) int {
	if id, ok := sp.ids[piece]; ok {
		return id
	}
	return -1
}

// normalize applies the model's whitespace rules: optional trimming and
// collapsing of spaces, the dummy prefix, and the "▁" escape.
func (sp *SentencePiece) normalize(text string) string {
	if sp.trimSpaces {
		text = strings.Join(strings.Fields(text), " ")
	}
	if text == "" {
		return ""
	}
	if sp.dummyPrefix {
		text = " " + text
	}
	if sp.escape {
		text = strings.ReplaceAll(text, " ", spSpace)
	}
	return text
}

// Encode tokenises text into piece ids. No BOS or EOS is added.
func (sp *SentencePiece) Encode(text string) []int {
	text = sp.normalize(text)
	// Split into symbols: user-defined symbols whole, otherwise one rune.
	type sym struct {
		s      string
		frozen bool // a user-defined symbol never merges
	}
	var syms []sym
	for i := 0; i < len(text); {
		matched := false
		for _, u := range sp.user[text[i]] {
			if strings.HasPrefix(text[i:], u) {
				syms = append(syms, sym{u, true})
				i += len(u)
				matched = true
				break
			}
		}
		if matched {
			continue
		}
		_, n := utf8.DecodeRuneInString(text[i:])
		syms = append(syms, sym{text[i : i+n], false})
		i += n
	}
	// Merge the best-scoring adjacent pair until none is in the vocabulary;
	// the leftmost wins a tie.
	for {
		best, bestScore := -1, float32(math.Inf(-1))
		for i := 0; i+1 < len(syms); i++ {
			if syms[i].frozen || syms[i+1].frozen {
				continue
			}
			id, ok := sp.ids[syms[i].s+syms[i+1].s]
			if !ok || sp.kinds[id] == spUnused {
				continue
			}
			if sp.scores[id] > bestScore {
				best, bestScore = i, sp.scores[id]
			}
		}
		if best < 0 {
			break
		}
		syms[best].s += syms[best+1].s
		syms = append(syms[:best+1], syms[best+2:]...)
	}
	ids := make([]int, 0, len(syms))
	for _, s := range syms {
		if id, ok := sp.ids[s.s]; ok && sp.kinds[id] != spUnused {
			ids = append(ids, id)
			continue
		}
		if !sp.byteFallback {
			ids = append(ids, sp.unk)
			continue
		}
		for i := 0; i < len(s.s); i++ {
			if id, ok := sp.ids[fmt.Sprintf("<0x%02X>", s.s[i])]; ok {
				ids = append(ids, id)
			} else {
				ids = append(ids, sp.unk)
			}
		}
	}
	return ids
}
