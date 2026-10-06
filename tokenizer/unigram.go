package tokenizer

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"unicode/utf8"
)

// Unigram is a SentencePiece Unigram model from a Hugging Face
// tokenizer.json, as T5 / umT5 ship it: special added tokens split out of
// the raw text, runs of spaces collapsed (the Replace normaliser), the
// Metaspace pre-tokenizer (spaces as "▁", a "▁" prepended, a split before
// every "▁"), then per piece the Viterbi segmentation that maximises the
// summed log-probabilities, unknown characters scored min_score − 10 and
// consecutive unknowns fused into one <unk> — tokenizers' Unigram model.
// Other normalisers and pre-tokenizers are refused at load time.
type Unigram struct {
	ids    map[string]int64
	scores []float64
	unk    int64
	unkLog float64
	maxLen int     // longest piece, bytes
	added  []added // special tokens, longest first
	eos    []int64 // the post-processor's suffix for a single sequence
}

// LoadUnigram reads a Unigram tokenizer.json.
func LoadUnigram(path string) (*Unigram, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	u, err := parseUnigram(raw)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	return u, nil
}

type unigramJSON struct {
	AddedTokens []struct {
		ID         int64  `json:"id"`
		Content    string `json:"content"`
		Special    bool   `json:"special"`
		Normalized bool   `json:"normalized"`
		LStrip     bool   `json:"lstrip"`
		RStrip     bool   `json:"rstrip"`
		SingleWord bool   `json:"single_word"`
	} `json:"added_tokens"`
	Normalizer    json.RawMessage `json:"normalizer"`
	PreTokenizer  json.RawMessage `json:"pre_tokenizer"`
	PostProcessor *struct {
		Type   string `json:"type"`
		Single []struct {
			Sequence *struct {
				ID string `json:"id"`
			} `json:"Sequence"`
			SpecialToken *struct {
				ID string `json:"id"`
			} `json:"SpecialToken"`
		} `json:"single"`
		SpecialTokens map[string]struct {
			IDs []int64 `json:"ids"`
		} `json:"special_tokens"`
	} `json:"post_processor"`
}

func parseUnigram(raw []byte) (*Unigram, error) {
	var j struct {
		unigramJSON
		Model struct {
			Type     string            `json:"type"`
			UnkID    *int64            `json:"unk_id"`
			ByteFall bool              `json:"byte_fallback"`
			Vocab    []json.RawMessage `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &j); err != nil {
		return nil, err
	}
	if j.Model.Type != "Unigram" {
		return nil, fmt.Errorf("model type %q is not Unigram", j.Model.Type)
	}
	if j.Model.ByteFall {
		return nil, fmt.Errorf("unigram byte fallback is not implemented")
	}
	if j.Model.UnkID == nil {
		return nil, fmt.Errorf("unigram model has no unk_id")
	}
	if err := checkNormalizer(j.Normalizer); err != nil {
		return nil, err
	}
	if err := checkMetaspace(j.PreTokenizer); err != nil {
		return nil, err
	}
	u := &Unigram{ids: make(map[string]int64, len(j.Model.Vocab)), scores: make([]float64, len(j.Model.Vocab)), unk: *j.Model.UnkID}
	minScore := math.Inf(1)
	for i, e := range j.Model.Vocab {
		var pair [2]any
		if err := json.Unmarshal(e, &pair); err != nil {
			return nil, fmt.Errorf("vocab entry %d: %w", i, err)
		}
		piece, ok1 := pair[0].(string)
		score, ok2 := pair[1].(float64)
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("vocab entry %d: want [piece, score]", i)
		}
		if _, dup := u.ids[piece]; !dup {
			u.ids[piece] = int64(i)
		}
		u.scores[i] = score
		minScore = min(minScore, score)
		u.maxLen = max(u.maxLen, len(piece))
	}
	if u.unk < 0 || int(u.unk) >= len(u.scores) {
		return nil, fmt.Errorf("unk_id %d outside the vocabulary", u.unk)
	}
	u.unkLog = minScore - 10 // tokenizers' K_UNK_PENALTY
	for _, a := range j.AddedTokens {
		if !a.Special {
			return nil, fmt.Errorf("non-special added token %q is not implemented", a.Content)
		}
		if a.Normalized || a.LStrip || a.RStrip || a.SingleWord {
			return nil, fmt.Errorf("added token %q: normalized/lstrip/rstrip/single_word not implemented", a.Content)
		}
		u.added = append(u.added, added{a.Content, a.ID})
	}
	sort.SliceStable(u.added, func(a, b int) bool { return len(u.added[a].content) > len(u.added[b].content) })
	if pp := j.PostProcessor; pp != nil {
		if pp.Type != "TemplateProcessing" {
			return nil, fmt.Errorf("post-processor %q is not implemented", pp.Type)
		}
		for i, s := range pp.Single {
			switch {
			case s.Sequence != nil:
				if i != 0 {
					return nil, fmt.Errorf("template with a prefix is not implemented")
				}
			case s.SpecialToken != nil:
				st, ok := pp.SpecialTokens[s.SpecialToken.ID]
				if !ok {
					return nil, fmt.Errorf("template names unknown special token %q", s.SpecialToken.ID)
				}
				u.eos = append(u.eos, st.IDs...)
			}
		}
	}
	return u, nil
}

// checkNormalizer accepts no normaliser or T5's Replace(" {2,}", " "),
// alone or as the only member of a Sequence.
func checkNormalizer(raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var n struct {
		Type        string            `json:"type"`
		Normalizers []json.RawMessage `json:"normalizers"`
		Pattern     struct {
			Regex  *string `json:"Regex"`
			String *string `json:"String"`
		} `json:"pattern"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &n); err != nil {
		return err
	}
	switch n.Type {
	case "Sequence":
		for _, m := range n.Normalizers {
			if err := checkNormalizer(m); err != nil {
				return err
			}
		}
		return nil
	case "Replace":
		if n.Pattern.Regex != nil && *n.Pattern.Regex == " {2,}" && n.Content == " " {
			return nil
		}
	}
	return fmt.Errorf("normalizer %s is not implemented", raw)
}

// checkMetaspace accepts Metaspace("▁", prepend_scheme always, split).
func checkMetaspace(raw json.RawMessage) error {
	var p struct {
		Type          string `json:"type"`
		Replacement   string `json:"replacement"`
		PrependScheme string `json:"prepend_scheme"`
		Split         bool   `json:"split"`
	}
	if err := json.Unmarshal(raw, &p); err != nil {
		return err
	}
	if p.Type != "Metaspace" || p.Replacement != spSpace || p.PrependScheme != "always" || !p.Split {
		return fmt.Errorf("pre-tokenizer %s is not implemented", raw)
	}
	return nil
}

// Size is the model vocabulary size (added tokens included where they
// overlap it, as in T5).
func (u *Unigram) Size() int { return len(u.scores) }

// ID is a piece's id, or -1.
func (u *Unigram) ID(piece string) int64 {
	if id, ok := u.ids[piece]; ok {
		return id
	}
	for _, a := range u.added {
		if a.content == piece {
			return a.id
		}
	}
	return -1
}

// Encode tokenises text. With special, the post-processor's template is
// applied (T5: "</s>" appended), as transformers' add_special_tokens=True.
func (u *Unigram) Encode(text string, special bool) []int64 {
	var ids []int64
	seg := 0
	for i := 0; i < len(text); {
		matched := ""
		var id int64
		for _, a := range u.added {
			if strings.HasPrefix(text[i:], a.content) {
				matched, id = a.content, a.id
				break
			}
		}
		if matched == "" {
			i++
			continue
		}
		ids = u.encodeSegment(ids, text[seg:i])
		ids = append(ids, id)
		i += len(matched)
		seg = i
	}
	ids = u.encodeSegment(ids, text[seg:])
	if special {
		ids = append(ids, u.eos...)
	}
	return ids
}

// encodeSegment normalises and pre-tokenises one stretch of text between
// added tokens, then segments each piece.
func (u *Unigram) encodeSegment(ids []int64, s string) []int64 {
	if s == "" {
		return ids
	}
	s = collapseSpaces(s)
	s = strings.ReplaceAll(s, " ", spSpace)
	if !strings.HasPrefix(s, spSpace) {
		s = spSpace + s
	}
	// Split before every "▁" (MergedWithNext).
	for len(s) > 0 {
		j := strings.Index(s[len(spSpace):], spSpace)
		if j < 0 {
			return u.viterbi(ids, s)
		}
		j += len(spSpace)
		ids = u.viterbi(ids, s[:j])
		s = s[j:]
	}
	return ids
}

// collapseSpaces replaces every run of two or more spaces with one.
func collapseSpaces(s string) string {
	if !strings.Contains(s, "  ") {
		return s
	}
	var b strings.Builder
	prev := false
	for _, r := range s {
		if r == ' ' {
			if prev {
				continue
			}
			prev = true
		} else {
			prev = false
		}
		b.WriteRune(r)
	}
	return b.String()
}

// viterbi appends the best segmentation of piece: best[e] is the best path
// score of piece[:e]; candidates are visited shortest first and replace a
// node only when strictly better, as tokenizers' encode_optimized.
func (u *Unigram) viterbi(ids []int64, piece string) []int64 {
	n := len(piece)
	type node struct {
		score float64
		start int
		id    int64
		set   bool
	}
	best := make([]node, n+1)
	best[0].set = true
	for s := 0; s < n; {
		if !best[s].set {
			_, w := utf8.DecodeRuneInString(piece[s:])
			s += w
			continue
		}
		_, w := utf8.DecodeRuneInString(piece[s:])
		single := false
		for e := s + 1; e <= n && e-s <= u.maxLen; e++ {
			if !utf8.RuneStart(byteAt(piece, e)) && e < n {
				continue
			}
			id, ok := u.ids[piece[s:e]]
			if !ok {
				continue
			}
			c := best[s].score + u.scores[id]
			if t := &best[e]; !t.set || c > t.score {
				*t = node{c, s, id, true}
			}
			if e-s == w {
				single = true
			}
		}
		if !single {
			c := best[s].score + u.unkLog
			if t := &best[s+w]; !t.set || c > t.score {
				*t = node{c, s, u.unk, true}
			}
		}
		s += w
	}
	// Backtrack, fusing runs of unknowns into one <unk>.
	var rev []int64
	for e := n; e > 0; {
		nd := best[e]
		if nd.id == u.unk && len(rev) > 0 && rev[len(rev)-1] == u.unk {
			e = nd.start
			continue
		}
		rev = append(rev, nd.id)
		e = nd.start
	}
	for i := len(rev) - 1; i >= 0; i-- {
		ids = append(ids, rev[i])
	}
	return ids
}

func byteAt(s string, i int) byte {
	if i >= len(s) {
		return 0
	}
	return s[i]
}
