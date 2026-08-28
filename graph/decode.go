package graph

import (
	"github.com/giraffesyo/ingot/ops"
	"github.com/giraffesyo/ingot/tensor"
)

// Decode is per-sequence autoregressive state: the KV caches of every
// cached-attention node plus the shared position counter. One Decode per
// concurrent sequence; the Session itself stays immutable. See
// docs/DESIGN-kvcache.md.
type Decode struct {
	state ops.DecodeState
}

// Pos returns the number of cached positions.
func (d *Decode) Pos() int { return d.state.Pos }

// NewDecode creates decode state with capacity for maxT positions. Slots
// allocate lazily on first use (shapes are only known at Run time).
func (s *Session) NewDecode(maxT int) *Decode {
	return &Decode{state: ops.DecodeState{MaxT: maxT, Slots: map[string]*ops.DecodeSlot{}}}
}

// RunDecode is Run with decode state attached: cached ingot.SDPA nodes
// append this run's tokens (the feed's T positions) to d and attend
// causally over the cached range. On success the position advances by
// tokens. Not safe for concurrent use of the same Decode.
func (s *Session) RunDecode(d *Decode, feeds map[string]*tensor.Tensor, tokens int) (map[string]*tensor.Tensor, error) {
	res, err := s.run(feeds, &d.state)
	if err == nil {
		d.state.Pos += tokens
	}
	return res, err
}

// CompileDecode compiles g for autoregressive decode: after the standard
// optimizer, the exporter attention cores with runtime-built causal masks
// are rewritten to the cached ingot.SDPA form (see docs/DESIGN-kvcache.md)
// and the orphaned mask chains removed. Run the result via RunDecode.
func CompileDecode(g *Graph) (*Session, error) {
	Optimize(g)
	st := map[string]int{}
	if fuseSDPADecode(g, st) {
		dce(g, st)
	}
	renumber(g)
	return CompileRaw(g)
}
