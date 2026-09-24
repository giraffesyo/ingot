// Package tokenizer implements the byte-level BPE subset of Hugging Face
// tokenizer.json files used by GPT-2-family and Qwen models: added (special)
// tokens split out first, NFC normalisation, the Qwen2/GPT-4-style
// pre-tokenizer split, GPT-2 byte-to-unicode mapping, and merge-rank BPE.
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

	mu    sync.Mutex
	cache map[string][]int64 // pre-token → ids
}

type added struct {
	content string
	id      int64
}

// qwen2Split is the pre-tokenizer regex this package implements by hand (Go's
// regexp has no lookahead); a tokenizer.json with any other split is refused.
const qwen2Split = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

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
		pt[0].Pattern.Regex != qwen2Split || pt[0].Behavior != "Isolated" || pt[0].Invert ||
		pt[1].Type != "ByteLevel" || pt[1].AddPrefixSpace || pt[1].UseRegex:
		return nil, fmt.Errorf("tokenizer: pre-tokenizer not supported (want the Qwen2 split + ByteLevel)")
	}
	t := &Tokenizer{vocab: f.Model.Vocab, ranks: map[[2]string]int{}, cache: map[string][]int64{}}
	// Merges are either "a b" strings (older files) or ["a", "b"] pairs.
	var pairs [][2]string
	var strs []string
	if err := json.Unmarshal(f.Model.Merges, &pairs); err != nil {
		if err := json.Unmarshal(f.Model.Merges, &strs); err != nil {
			return nil, fmt.Errorf("tokenizer: merges: %w", err)
		}
		for _, s := range strs {
			a, b, ok := strings.Cut(s, " ")
			if !ok {
				return nil, fmt.Errorf("tokenizer: merge %q", s)
			}
			pairs = append(pairs, [2]string{a, b})
		}
	}
	for i, p := range pairs {
		if _, dup := t.ranks[p]; !dup {
			t.ranks[p] = i
		}
	}
	for _, a := range f.AddedTokens {
		if a.LStrip || a.RStrip || a.Single {
			return nil, fmt.Errorf("tokenizer: added token %q: lstrip/rstrip/single_word not supported", a.Content)
		}
		t.added = append(t.added, added{a.Content, a.ID})
	}
	sort.SliceStable(t.added, func(i, j int) bool { return len(t.added[i].content) > len(t.added[j].content) })
	t.byteCh = bytesToUnicode()
	return t, nil
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
	for _, piece := range Split(s) {
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
