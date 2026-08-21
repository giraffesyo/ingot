package graph

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/giraffesyo/ocr/ops"
	"github.com/giraffesyo/ocr/tensor"
)

// Session is a compiled, runnable graph.
type Session struct {
	g     *Graph
	steps []step
	pool  *tensor.Pool
	// refcount of each value id over the whole graph (consumer count, +1 if output)
	uses []int
	nval int

	// Profile enables per-node timing (see Stats).
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

// Compile resolves every node against the ops registry. It returns an error
// listing all unsupported ops rather than stopping at the first.
func Compile(g *Graph) (*Session, error) {
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
	return s, nil
}

// Graph returns the underlying graph.
func (s *Session) Graph() *Graph { return s.g }

// Run executes the graph. feeds maps input names to tensors; every graph
// input must be present. Returns graph outputs by name. Output tensors are
// owned by the caller (not pooled).
func (s *Session) Run(feeds map[string]*tensor.Tensor) (map[string]*tensor.Tensor, error) {
	vals := make([]*tensor.Tensor, s.nval)
	live := make([]int, s.nval) // remaining uses
	copy(live, s.uses)
	pooled := make([]bool, s.nval)
	for _, v := range s.g.Values {
		if v.Const != nil {
			vals[v.id] = v.Const
		}
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
	ctx := &ops.Ctx{Pool: s.pool}
	isOutput := map[int]bool{}
	for _, v := range s.g.Outputs {
		isOutput[v.id] = true
	}
	in := make([]*tensor.Tensor, 0, 8)
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
			if isOutput[id] {
				// Detach from pool so the caller owns it.
				t = t.Clone()
			}
			vals[id] = t
			pooled[id] = !isOutput[id]
		}
		// Release inputs whose last use was this step.
		for _, id := range st.in {
			if id < 0 {
				continue
			}
			live[id]--
			if live[id] == 0 && pooled[id] {
				s.pool.Put(vals[id])
				vals[id] = nil
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
