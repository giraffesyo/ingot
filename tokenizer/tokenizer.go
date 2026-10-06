// Package tokenizer implements the byte-level BPE subset of Hugging Face
// tokenizer.json files used by GPT-2-family and Qwen models: added (special)
// tokens split out first, NFC normalisation, the Qwen2/GPT-4-style or the
// Mistral-style pre-tokenizer split, GPT-2 byte-to-unicode mapping, and
// merge-rank BPE. Checkpoints without a tokenizer.json (vocab.json +
// merges.txt) load through LoadBPE. SentencePiece BPE .model files load
// through LoadSentencePiece, T5-style Unigram tokenizer.json files through
// LoadUnigram.
//
// Configurations outside that subset are rejected at load time rather than
// tokenised approximately.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// Tokenizer encodes text to token ids.
type Tokenizer struct {
	vocab  map[string]int64
	ranks  map[[2]string]int
	added  []added // longest content first
	byteCh [256]string
	split  func(string) []string

	mu    sync.Mutex
	cache map[string][]int64 // pre-token → ids
}

type added struct {
	content string
	id      int64
}

// qwen2Split and mistralSplit are the pre-tokenizer regexes this package
// implements by hand (Go's regexp has no lookahead); a tokenizer.json with
// any other split is refused.
const (
	qwen2Split   = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	mistralSplit = `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`
)

// SplitKind selects the pre-tokenizer for LoadBPE.
type SplitKind int

const (
	// PreQwen2 is the Qwen2 / GPT-4-style split (Split).
	PreQwen2 SplitKind = iota
	// PreMistral is the Mistral-style split (SplitMistral): case-aware
	// letter runs, no contraction rule. transformers' fix_mistral_regex
	// installs it on any tokenizer it loads, Qwen ones included.
	PreMistral
)

// Load reads a tokenizer.json.
func Load(path string) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}
	var f struct {
		Normalizer *struct {
			Type string `json:"type"`
		} `json:"normalizer"`
		PreTokenizer struct {
			Type          string `json:"type"`
			Pretokenizers []struct {
				Type    string `json:"type"`
				Pattern struct {
					Regex string `json:"Regex"`
				} `json:"pattern"`
				Behavior       string `json:"behavior"`
				Invert         bool   `json:"invert"`
				AddPrefixSpace bool   `json:"add_prefix_space"`
				UseRegex       bool   `json:"use_regex"`
			} `json:"pretokenizers"`
		} `json:"pre_tokenizer"`
		Model struct {
			Type         string           `json:"type"`
			Vocab        map[string]int64 `json:"vocab"`
			Merges       json.RawMessage  `json:"merges"`
			ByteFallback bool             `json:"byte_fallback"`
			Prefix       string           `json:"continuing_subword_prefix"`
			Suffix       string           `json:"end_of_word_suffix"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int64  `json:"id"`
			Content string `json:"content"`
			LStrip  bool   `json:"lstrip"`
			RStrip  bool   `json:"rstrip"`
			Single  bool   `json:"single_word"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	pt := f.PreTokenizer.Pretokenizers
	switch {
	case f.Model.Type != "BPE" || f.Model.ByteFallback || f.Model.Prefix != "" || f.Model.Suffix != "":
		return nil, fmt.Errorf("tokenizer: model %s (byte_fallback=%v) not supported", f.Model.Type, f.Model.ByteFallback)
	case f.Normalizer != nil && f.Normalizer.Type != "NFC":
		return nil, fmt.Errorf("tokenizer: normalizer %q not supported", f.Normalizer.Type)
	case f.PreTokenizer.Type != "Sequence" || len(pt) != 2 || pt[0].Type != "Split" ||
		(pt[0].Pattern.Regex != qwen2Split && pt[0].Pattern.Regex != mistralSplit) ||
		pt[0].Behavior != "Isolated" || pt[0].Invert ||
		pt[1].Type != "ByteLevel" || pt[1].AddPrefixSpace || pt[1].UseRegex:
		return nil, fmt.Errorf("tokenizer: pre-tokenizer not supported (want the Qwen2 or Mistral split + ByteLevel)")
	}
	split := PreQwen2
	if pt[0].Pattern.Regex == mistralSplit {
		split = PreMistral
	}
	// Merges are either "a b" strings (older files) or ["a", "b"] pairs.
	var pairs [][2]string
	var strs []string
	if err := json.Unmarshal(f.Model.Merges, &pairs); err != nil {
		if err := json.Unmarshal(f.Model.Merges, &strs); err != nil {
			return nil, fmt.Errorf("tokenizer: merges: %w", err)
		}
		if pairs, err = parseMerges(strs); err != nil {
			return nil, err
		}
	}
	var add []AddedToken
	for _, a := range f.AddedTokens {
		if a.LStrip || a.RStrip || a.Single {
			return nil, fmt.Errorf("tokenizer: added token %q: lstrip/rstrip/single_word not supported", a.Content)
		}
		add = append(add, AddedToken{a.Content, a.ID})
	}
	return build(f.Model.Vocab, pairs, add, split), nil
}

// AddedToken is a special token matched verbatim before pre-tokenisation.
type AddedToken struct {
	Content string
	ID      int64
}

// LoadBPE builds a tokenizer from a slow-tokenizer checkpoint: vocab.json
// (token → id), merges.txt (one "a b" pair per line, optional "#version"
// header) and the added tokens (from tokenizer_config.json's
// added_tokens_decoder). Normalisation is NFC, as in Qwen2Tokenizer.
func LoadBPE(vocabPath, mergesPath string, add []AddedToken, split SplitKind) (*Tokenizer, error) {
	raw, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}
	var vocab map[string]int64
	if err := json.Unmarshal(raw, &vocab); err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", vocabPath, err)
	}
	raw, err = os.ReadFile(mergesPath)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}
	var lines []string
	for l := range strings.SplitSeq(string(raw), "\n") {
		if l = strings.TrimRight(l, "\r"); l != "" && !strings.HasPrefix(l, "#version") {
			lines = append(lines, l)
		}
	}
	pairs, err := parseMerges(lines)
	if err != nil {
		return nil, err
	}
	return build(vocab, pairs, add, split), nil
}

// AddedTokensFromConfig reads tokenizer_config.json's added_tokens_decoder.
func AddedTokensFromConfig(path string) ([]AddedToken, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}
	var c struct {
		Added map[string]struct {
			Content string `json:"content"`
			LStrip  bool   `json:"lstrip"`
			RStrip  bool   `json:"rstrip"`
			Single  bool   `json:"single_word"`
		} `json:"added_tokens_decoder"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	var out []AddedToken
	for id, a := range c.Added {
		if a.LStrip || a.RStrip || a.Single {
			return nil, fmt.Errorf("tokenizer: added token %q: lstrip/rstrip/single_word not supported", a.Content)
		}
		var n int64
		if _, err := fmt.Sscan(id, &n); err != nil {
			return nil, fmt.Errorf("tokenizer: added token id %q", id)
		}
		out = append(out, AddedToken{a.Content, n})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

func parseMerges(strs []string) ([][2]string, error) {
	pairs := make([][2]string, 0, len(strs))
	for _, s := range strs {
		a, b, ok := strings.Cut(s, " ")
		if !ok {
			return nil, fmt.Errorf("tokenizer: merge %q", s)
		}
		pairs = append(pairs, [2]string{a, b})
	}
	return pairs, nil
}

func build(vocab map[string]int64, pairs [][2]string, add []AddedToken, split SplitKind) *Tokenizer {
	t := &Tokenizer{vocab: vocab, ranks: map[[2]string]int{}, cache: map[string][]int64{}, split: Split}
	if split == PreMistral {
		t.split = SplitMistral
	}
	for i, p := range pairs {
		if _, dup := t.ranks[p]; !dup {
			t.ranks[p] = i
		}
	}
	for _, a := range add {
		t.added = append(t.added, added{a.Content, a.ID})
	}
	sort.SliceStable(t.added, func(i, j int) bool { return len(t.added[i].content) > len(t.added[j].content) })
	t.byteCh = bytesToUnicode()
	return t
}

// Encode tokenises text (no special tokens are added).
func (t *Tokenizer) Encode(text string) ([]int64, error) {
	var ids []int64
	for len(text) > 0 {
		// Next added token (leftmost; longest among those at one position).
		at, which := len(text), -1
		for i, a := range t.added {
			if j := strings.Index(text[:min(len(text), at+len(a.content))], a.content); j >= 0 && j < at {
				at, which = j, i
			}
		}
		seg := text[:at]
		if seg != "" {
			var err error
			if ids, err = t.encodeSegment(ids, seg); err != nil {
				return nil, err
			}
		}
		if which < 0 {
			break
		}
		ids = append(ids, t.added[which].id)
		text = text[at+len(t.added[which].content):]
	}
	return ids, nil
}

func (t *Tokenizer) encodeSegment(ids []int64, s string) ([]int64, error) {
	s = norm.NFC.String(s)
	for _, piece := range t.split(s) {
		t.mu.Lock()
		c, ok := t.cache[piece]
		t.mu.Unlock()
		if !ok {
			var b strings.Builder
			for i := 0; i < len(piece); i++ {
				b.WriteString(t.byteCh[piece[i]])
			}
			var err error
			if c, err = t.bpe(b.String()); err != nil {
				return nil, err
			}
			t.mu.Lock()
			t.cache[piece] = c
			t.mu.Unlock()
		}
		ids = append(ids, c...)
	}
	return ids, nil
}

// bpe merges a byte-mapped pre-token by ascending merge rank.
func (t *Tokenizer) bpe(word string) ([]int64, error) {
	var syms []string
	for _, r := range word {
		syms = append(syms, string(r))
	}
	for len(syms) > 1 {
		best, at := -1, -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.ranks[[2]string{syms[i], syms[i+1]}]; ok && (best < 0 || r < best) {
				best, at = r, i
			}
		}
		if at < 0 {
			break
		}
		// Merge every occurrence of the best pair, left to right.
		a, b := syms[at], syms[at+1]
		out := syms[:0:0]
		for i := 0; i < len(syms); i++ {
			if i+1 < len(syms) && syms[i] == a && syms[i+1] == b {
				out = append(out, a+b)
				i++
				continue
			}
			out = append(out, syms[i])
		}
		syms = out
	}
	ids := make([]int64, len(syms))
	for i, s := range syms {
		id, ok := t.vocab[s]
		if !ok {
			return nil, fmt.Errorf("tokenizer: symbol %q not in vocab", s)
		}
		ids[i] = id
	}
	return ids, nil
}

// bytesToUnicode is GPT-2's reversible byte → printable-rune map.
func bytesToUnicode() [256]string {
	var m [256]string
	n := 0
	for b := range 256 {
		if (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
			m[b] = string(rune(b))
		} else {
			m[b] = string(rune(256 + n))
			n++
		}
	}
	return m
}

// Split is the Qwen2 pre-tokenizer: qwen2Split's alternatives tried in
// order at each position (leftmost-first, as the Rust/Oniguruma engines
// do), with the lookahead handled explicitly.
func Split(s string) []string {
	var out []string
	for i := 0; i < len(s); {
		n := matchAt(s, i)
		out = append(out, s[i:i+n])
		i += n
	}
	return out
}

func isL(r rune) bool     { return unicode.IsLetter(r) }
func isN(r rune) bool     { return unicode.IsNumber(r) }
func isS(r rune) bool     { return unicode.IsSpace(r) }
func isNL(r rune) bool    { return r == '\r' || r == '\n' }
func isPunct(r rune) bool { return !isS(r) && !isL(r) && !isN(r) } // [^\s\p{L}\p{N}]

// matchAt returns the byte length of the first alternative matching at i.
func matchAt(s string, i int) int {
	r0, w0 := utf8.DecodeRuneInString(s[i:])
	// 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
	if r0 == '\'' {
		rest := strings.ToLower(s[i+1 : min(len(s), i+3)])
		for _, c := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
			if strings.HasPrefix(rest, c) {
				return 1 + len(c)
			}
		}
	}
	// 2. [^\r\n\p{L}\p{N}]?\p{L}+
	j := i
	if !isNL(r0) && !isL(r0) && !isN(r0) {
		if r1, _ := utf8.DecodeRuneInString(s[i+w0:]); i+w0 < len(s) && isL(r1) {
			j = i + w0
		}
	}
	if k := run(s, j, isL); k > j {
		return k - i
	}
	// 3. \p{N}
	if isN(r0) {
		return w0
	}
	// 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
	j = i
	if r0 == ' ' {
		j = i + 1
	}
	if k := run(s, j, isPunct); k > j {
		return run(s, k, isNL) - i
	}
	return matchSpace(s, i, w0)
}

// matchSpace is the whitespace alternatives both splits end with:
// \s*[\r\n]+|\s+(?!\S)|\s+.
func matchSpace(s string, i, w0 int) int {
	// 5. \s*[\r\n]+ — the longest whitespace prefix ending in a newline.
	ws := run(s, i, isS)
	for k := ws; k > i; {
		r, w := utf8.DecodeLastRuneInString(s[i:k])
		if isNL(r) {
			return k - i
		}
		k -= w
	}
	// 6. \s+(?!\S): all of the run at end of text, else all but its last rune.
	if ws > i {
		if ws == len(s) {
			return ws - i
		}
		_, w := utf8.DecodeLastRuneInString(s[i:ws])
		if ws-w > i {
			return ws - w - i
		}
		// 7. \s+
		return ws - i
	}
	return w0 // unreachable: every rune is L, N, whitespace or punctuation
}

// run returns the end of the run of runes satisfying f starting at i.
func run(s string, i int, f func(rune) bool) int {
	for i < len(s) {
		r, w := utf8.DecodeRuneInString(s[i:])
		if !f(r) {
			break
		}
		i += w
	}
	return i
}

// SplitMistral is the Mistral-style pre-tokenizer (mistralSplit): like
// Split, alternatives tried in order at each position with regex
// backtracking resolved explicitly.
func SplitMistral(s string) []string {
	var out []string
	for i := 0; i < len(s); {
		n := matchMistralAt(s, i)
		out = append(out, s[i:i+n])
		i += n
	}
	return out
}

// isUp is [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}], isLow [\p{Ll}\p{Lm}\p{Lo}\p{M}].
func isUp(r rune) bool {
	return unicode.In(r, unicode.Lu, unicode.Lt, unicode.Lm, unicode.Lo, unicode.M)
}
func isLow(r rune) bool {
	return unicode.In(r, unicode.Ll, unicode.Lm, unicode.Lo, unicode.M)
}

// upperLower matches [Up]*[Low]+ at j (greedy Up*, backtracking into it
// until a Low follows) and returns the end, or -1.
func upperLower(s string, j int) int {
	ends := []int{j} // ends[u] = offset after u Up runes
	for k := j; k < len(s); {
		r, w := utf8.DecodeRuneInString(s[k:])
		if !isUp(r) {
			break
		}
		k += w
		ends = append(ends, k)
	}
	for u := len(ends) - 1; u >= 0; u-- {
		if e := run(s, ends[u], isLow); e > ends[u] {
			return e
		}
	}
	return -1
}

// upperThenLower matches [Up]+[Low]* at j and returns the end, or -1.
func upperThenLower(s string, j int) int {
	if k := run(s, j, isUp); k > j {
		return run(s, k, isLow)
	}
	return -1
}

// matchMistralAt returns the byte length of the first alternative of
// mistralSplit matching at i.
func matchMistralAt(s string, i int) int {
	r0, w0 := utf8.DecodeRuneInString(s[i:])
	lead := !isNL(r0) && !isL(r0) && !isN(r0) // [^\r\n\p{L}\p{N}]
	// 1-2. prefix? letters, prefix tried first (greedy ?).
	for _, f := range []func(string, int) int{upperLower, upperThenLower} {
		if lead && i+w0 < len(s) {
			if e := f(s, i+w0); e >= 0 {
				return e - i
			}
		}
		if e := f(s, i); e >= 0 {
			return e - i
		}
	}
	// 3. \p{N}
	if isN(r0) {
		return w0
	}
	// 4.  ?[^\s\p{L}\p{N}]+[\r\n/]*
	j := i
	if r0 == ' ' {
		j = i + 1
	}
	if k := run(s, j, isPunct); k > j {
		return run(s, k, func(r rune) bool { return isNL(r) || r == '/' }) - i
	}
	return matchSpace(s, i, w0)
}
