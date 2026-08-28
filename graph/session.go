package graph

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/giraffesyo/ingot/ops"
	"github.com/giraffesyo/ingot/tensor"
)

// Session is a compiled, runnable graph.
//
// Session.Run is safe for concurrent use by multiple goroutines: per-run state
// (buffers, refcounts) is allocated per call, the tensor pool is mutually
// exclusive, ops never mutate their inputs, and the only cross-run mutable state
// (a per-op weight-pack cache) is guarded by sync.Once. Constants are shared
// read-only. The exception is the Profile flag: enabling it makes Run write to
// shared timing counters unsynchronised, so profile a Session from one goroutine
// at a time. See TestSessionConcurrentRun.
type Session struct {
	g     *Graph
	steps []step
	pool  *tensor.Pool
	// refcount of each value id over the whole graph (consumer count, +1 if output)
	uses []int
	nval int
	// compile-time run templates: constant values by id, output flags by id
	constVals []*tensor.Tensor
	isOutput  []bool
	scratch   sync.Pool // *runScratch

	// Profile enables per-node timing (see Stats). Not safe to enable while the
	// Session is run concurrently — the timing counters are unsynchronised.
	Profile bool
	stats   []time.Duration // per step, cumulative
	runs    int
}

// OpStat is aggregated timing for one op type.
type OpStat struct {
	OpType string
	Count  int
	Total  time.Duration
}

// Stats returns per-op-type timing aggregated over all profiled runs, sorted
// by total time descending.
func (s *Session) Stats() []OpStat {
	m := map[string]*OpStat{}
	for i, st := range s.steps {
		o := m[st.node.OpType]
		if o == nil {
			o = &OpStat{OpType: st.node.OpType}
			m[st.node.OpType] = o
		}
		o.Count++
		if i < len(s.stats) {
			o.Total += s.stats[i]
		}
	}
	out := make([]OpStat, 0, len(m))
	for _, o := range m {
		out = append(out, *o)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Total > out[j].Total })
	return out
}

// Runs returns the number of profiled runs (divide Stats totals by this).
func (s *Session) Runs() int { return s.runs }

// Release hands a Run's outputs back to the session's buffer pool. Optional:
// callers that skip it simply leave the buffers to the GC. Call it only once
// per Run result, only after the caller is done reading the tensors, and never
// for tensors it still holds references to. Constant outputs (detached clones)
// are ignored by the pool.
func (s *Session) Release(res map[string]*tensor.Tensor) {
	for _, t := range res {
		if t != nil {
			s.pool.Put(t)
		}
	}
}

// NodeStats returns per-node timing (cumulative over profiled runs) in
// execution order.
func (s *Session) NodeStats() []struct {
	Node  *Node
	Total time.Duration
} {
	out := make([]struct {
		Node  *Node
		Total time.Duration
	}, len(s.steps))
	for i, st := range s.steps {
		out[i].Node = st.node
		if i < len(s.stats) {
			out[i].Total = s.stats[i]
		}
	}
	return out
}

type step struct {
	node *Node
	op   ops.Op
	in   []int // value ids (-1 for nil)
	out  []int // value ids (-1 for nil)
}

// Compile optimises g in place (see Optimize) and resolves every node against
// the ops registry. It returns an error listing all unsupported ops rather than
// stopping at the first.
func Compile(g *Graph) (*Session, error) {
	Optimize(g)
	return CompileRaw(g)
}

// CompileRaw is Compile without the optimizer: the graph runs node-for-node as
// loaded. Useful for A/B-testing rewrites and for debugging.
func CompileRaw(g *Graph) (*Session, error) {
	s := &Session{g: g, pool: tensor.NewPool(), nval: len(g.Values)}
	s.uses = make([]int, s.nval)
	var missing []string
	seen := map[string]bool{}
	for _, n := range g.Nodes {
		ver := g.OpsetVersion(n.Domain)
		b, err := ops.Lookup(n.Domain, n.OpType, ver)
		if err != nil {
			key := n.Domain + "/" + n.OpType
			if !seen[key] {
				seen[key] = true
				missing = append(missing, fmt.Sprintf("%s (opset %d, e.g. node %q)", n.OpType, ver, n.Name))
			}
			continue
		}
		if len(missing) > 0 {
			continue
		}
		op, err := b(ops.NodeInfo{
			Name: n.Name, OpType: n.OpType, Domain: n.Domain, Version: ver,
			Attrs: n.Attrs, NumIn: len(n.Inputs), NumOut: len(n.Outputs),
		})
		if err != nil {
			return nil, fmt.Errorf("graph: compile: %w", err)
		}
		st := step{node: n, op: op}
		for _, v := range n.Inputs {
			if v == nil {
				st.in = append(st.in, -1)
			} else {
				st.in = append(st.in, v.id)
				s.uses[v.id]++
			}
		}
		for _, v := range n.Outputs {
			if v == nil {
				st.out = append(st.out, -1)
			} else {
				st.out = append(st.out, v.id)
			}
		}
		s.steps = append(s.steps, st)
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return nil, fmt.Errorf("graph: unsupported ops:\n  %s\nsupported: %s",
			strings.Join(missing, "\n  "), strings.Join(ops.Supported(), " "))
	}
	for _, v := range g.Outputs {
		s.uses[v.id]++
	}
	s.constVals = make([]*tensor.Tensor, s.nval)
	for _, v := range g.Values {
		if v.Const != nil {
			s.constVals[v.id] = v.Const
		}
	}
	s.isOutput = make([]bool, s.nval)
	for _, v := range g.Outputs {
		s.isOutput[v.id] = true
	}
	return s, nil
}

// runScratch is the pooled per-Run working state.
type runScratch struct {
	vals   []*tensor.Tensor
	live   []int
	pooled []bool
	alias  []int // view id → id of the value owning the shared buffer (-1: none)
	in     []*tensor.Tensor
	ctx    ops.Ctx
}

// Graph returns the underlying graph.
func (s *Session) Graph() *Graph { return s.g }

// Run executes the graph. feeds maps input names to tensors; every graph
// input must be present. Returns graph outputs by name. Output tensors are
// owned by the caller (not pooled). Safe for concurrent use by multiple
// goroutines (see the Session doc; the sole exception is Profile).
func (s *Session) Run(feeds map[string]*tensor.Tensor) (map[string]*tensor.Tensor, error) {
	return s.run(feeds, nil)
}

func (s *Session) run(feeds map[string]*tensor.Tensor, dec *ops.DecodeState) (map[string]*tensor.Tensor, error) {
	sc, _ := s.scratch.Get().(*runScratch)
	if sc == nil {
		sc = &runScratch{
			vals:   make([]*tensor.Tensor, s.nval),
			live:   make([]int, s.nval),
			pooled: make([]bool, s.nval),
			alias:  make([]int, s.nval),
			in:     make([]*tensor.Tensor, 0, 8),
		}
	}
	defer func() {
		clear(sc.vals)
		s.scratch.Put(sc)
	}()
	vals, live, pooled, alias := sc.vals, sc.live, sc.pooled, sc.alias
	copy(vals, s.constVals)
	copy(live, s.uses) // remaining uses
	clear(pooled)
	for i := range alias {
		alias[i] = -1
	}
	for _, v := range s.g.Inputs {
		t, ok := feeds[v.Name]
		if !ok {
			return nil, fmt.Errorf("graph: missing input %q", v.Name)
		}
		if v.DType != tensor.Invalid && t.DType() != v.DType {
			return nil, fmt.Errorf("graph: input %q: dtype %s, model expects %s", v.Name, t.DType(), v.DType)
		}
		vals[v.id] = t
	}
	ctx := &sc.ctx
	ctx.Pool = s.pool
	ctx.Decode = dec
	defer func() { ctx.Decode = nil }()
	isOutput := s.isOutput
	in := sc.in
	defer func() { sc.in = in[:0] }()
	if s.Profile && len(s.stats) != len(s.steps) {
		s.stats = make([]time.Duration, len(s.steps))
	}
	if s.Profile {
		s.runs++
	}
	for si := range s.steps {
		st := &s.steps[si]
		var t0 time.Time
		if s.Profile {
			t0 = time.Now()
		}
		in = in[:0]
		for _, id := range st.in {
			if id < 0 {
				in = append(in, nil)
				continue
			}
			t := vals[id]
			if t == nil {
				return nil, fmt.Errorf("graph: %s: input %q has no value", st.node, s.g.Values[idName(s.g, id)].Name)
			}
			in = append(in, t)
		}
		outs, err := st.op.Run(ctx, in)
		if err != nil {
			return nil, fmt.Errorf("graph: %w", err)
		}
		if s.Profile {
			s.stats[si] += time.Since(t0)
		}
		if len(outs) < len(st.out) {
			// Ops may return fewer outputs if trailing ones are optional.
			for k := len(outs); k < len(st.out); k++ {
				if st.out[k] >= 0 && live[st.out[k]] > 0 {
					return nil, fmt.Errorf("graph: %s produced %d outputs, needs %d", st.node, len(outs), len(st.out))
				}
			}
		}
		for k, id := range st.out {
			if id < 0 || k >= len(outs) {
				continue
			}
			t := outs[k]
			if t == nil {
				if live[id] > 0 {
					return nil, fmt.Errorf("graph: %s: output %d is nil but is used", st.node, k)
				}
				continue
			}
			// Graph outputs stay pooled but are excluded from internal
			// release — the caller owns them until it hands them back via
			// Release (or lets the GC take them).
			vals[id] = t
			pooled[id] = !isOutput[id]
			// A view output (Reshape & friends, or Identity's pass-through)
			// shares its input's buffer. The buffer stays in the custody of
			// its owning value (the root of a view chain): the view's own
			// uses pin the root, and the root returns to the pool only when
			// direct and view uses all drain (see the release loop).
			for _, inID := range st.in {
				if inID >= 0 && vals[inID] != nil && t.SharesBuffer(vals[inID]) {
					root := inID
					if r := alias[inID]; r >= 0 {
						root = r
					}
					alias[id] = root
					pooled[id] = false
					live[root] += live[id]
					break
				}
			}
		}
		// Release inputs whose last use was this step. A view use also
		// releases one pin on the buffer's owning value.
		for _, id := range st.in {
			if id < 0 {
				continue
			}
			live[id]--
			if live[id] == 0 && pooled[id] {
				s.pool.Put(vals[id])
				vals[id] = nil
			}
			if r := alias[id]; r >= 0 {
				live[r]--
				if live[r] == 0 && pooled[r] {
					s.pool.Put(vals[r])
					vals[r] = nil
				}
			}
		}
	}
	res := make(map[string]*tensor.Tensor, len(s.g.Outputs))
	for _, v := range s.g.Outputs {
		t := vals[v.id]
		if t == nil {
			return nil, fmt.Errorf("graph: output %q was not produced", v.Name)
		}
		if v.Const != nil {
			t = t.Clone()
		}
		res[v.Name] = t
	}
	return res, nil
}

func idName(g *Graph, id int) string {
	for name, v := range g.Values {
		if v.id == id {
			return name
		}
	}
	return ""
}
