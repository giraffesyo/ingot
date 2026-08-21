// Package ops implements ONNX operators over tensors. Ops are thin: they
// validate, compute shapes, allocate outputs from the context pool, and
// dispatch to kernels. Nothing here is OCR-specific.
//
// Registry: ops register a Builder per (domain, op_type, since_version).
// Lookup picks the newest registered version <= the model's opset version, as
// ONNX requires.
package ops

import (
	"fmt"
	"sort"
	"sync"

	"github.com/giraffesyo/ocr/tensor"
)

// Ctx is per-session execution state passed to every op.
type Ctx struct {
	Pool *tensor.Pool
}

// New allocates an output tensor from the pool.
func (c *Ctx) New(dt tensor.DType, shape ...int) *tensor.Tensor {
	if c.Pool == nil {
		return tensor.New(dt, shape...)
	}
	return c.Pool.Get(dt, shape...)
}

// NodeInfo describes a node to a Builder.
type NodeInfo struct {
	Name    string
	OpType  string
	Domain  string
	Version int // opset version in effect for Domain
	Attrs   Attrs
	NumIn   int
	NumOut  int
}

// Op is a prepared operator instance. Run computes outputs for the given
// inputs; nil entries in `in` are omitted optional inputs. Outputs must be
// freshly allocated from ctx (or be inputs passed through — see Aliasing).
type Op interface {
	Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error)
}

// Builder constructs an Op for a node, validating attributes up front.
type Builder func(n NodeInfo) (Op, error)

type key struct {
	domain, op string
}

type entry struct {
	since int
	build Builder
}

var (
	regMu    sync.RWMutex
	registry = map[key][]entry{} // sorted by since asc
)

// Register adds a builder for op in domain, valid from opset `since`.
// Domain "" is the default ai.onnx domain.
func Register(domain, op string, since int, b Builder) {
	regMu.Lock()
	defer regMu.Unlock()
	k := key{domain, op}
	es := append(registry[k], entry{since, b})
	sort.Slice(es, func(i, j int) bool { return es[i].since < es[j].since })
	registry[k] = es
}

// Lookup returns the builder for op at the given opset version, or an error
// naming the op so unsupported ops fail loudly at load time.
func Lookup(domain, op string, version int) (Builder, error) {
	regMu.RLock()
	defer regMu.RUnlock()
	es := registry[key{domain, op}]
	var best *entry
	for i := range es {
		if es[i].since <= version {
			best = &es[i]
		}
	}
	if best == nil {
		if len(es) == 0 {
			return nil, fmt.Errorf("ops: unsupported op %q (domain %q)", op, domain)
		}
		return nil, fmt.Errorf("ops: op %q (domain %q) not supported at opset %d (have since %d)", op, domain, version, es[0].since)
	}
	return best.build, nil
}

// Supported lists registered (domain, op) pairs — for diagnostics.
func Supported() []string {
	regMu.RLock()
	defer regMu.RUnlock()
	out := make([]string, 0, len(registry))
	for k := range registry {
		if k.domain == "" {
			out = append(out, k.op)
		} else {
			out = append(out, k.domain+"."+k.op)
		}
	}
	sort.Strings(out)
	return out
}

// Errorf formats an error prefixed with the node name.
func (n NodeInfo) Errorf(format string, a ...any) error {
	return fmt.Errorf("%s(%s): %s", n.OpType, n.Name, fmt.Sprintf(format, a...))
}
