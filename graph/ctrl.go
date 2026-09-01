package graph

import (
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/ops"
	"github.com/giraffesyo/ingot/tensor"
)

// Control flow (If/Loop) executes compiled sub-Sessions from inside the
// executor: the ops implement ops.Op like any kernel, so CompileRaw wires
// them into ordinary steps. Captured outer values arrive as the node's
// trailing inputs (Node.Caps, established at build) and are fed to the
// sub-Session by name. Sub-session outputs are caller-owned tensors; the
// outer executor's alias detection handles pass-through outputs that share
// a fed buffer.

// capRef maps one subgraph capture to the node-input index that carries it.
type capRef struct {
	name string
	idx  int
}

func capRefs(n *Node, sub *Graph) ([]capRef, error) {
	base := len(n.Inputs) - len(n.Caps)
	refs := make([]capRef, 0, len(sub.Captures))
	for _, name := range sub.Captures {
		found := -1
		for i, c := range n.Caps {
			if c == name {
				found = base + i
				break
			}
		}
		if found < 0 {
			return nil, fmt.Errorf("graph: %s: capture %q not wired", n, name)
		}
		refs = append(refs, capRef{name, found})
	}
	return refs, nil
}

// ifOp: ONNX If. Inputs: cond ++ captures. Runs then_branch or else_branch.
type ifOp struct {
	node      *Node
	then, els *Session
	thenCaps  []capRef
	elsCaps   []capRef
	thenOuts  []string
	elsOuts   []string
}

func (o *ifOp) Run(ctx *ops.Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 1 || in[0] == nil || in[0].DType() != tensor.Bool || in[0].Numel() != 1 {
		return nil, fmt.Errorf("graph: %s: If condition must be a bool scalar", o.node)
	}
	br, caps, names := o.els, o.elsCaps, o.elsOuts
	if in[0].Bool()[0] {
		br, caps, names = o.then, o.thenCaps, o.thenOuts
	}
	feeds := make(map[string]*tensor.Tensor, len(caps))
	for _, c := range caps {
		feeds[c.name] = in[c.idx]
	}
	outs, err := br.Run(feeds)
	if err != nil {
		return nil, fmt.Errorf("graph: %s branch: %w", o.node, err)
	}
	res := make([]*tensor.Tensor, len(names))
	for i, nm := range names {
		res[i] = outs[nm]
	}
	return res, nil
}

// loopOp: ONNX Loop. Node inputs: M (opt), cond (opt), K carried initials,
// then captures. Body inputs: iter (i64 scalar), cond_in (bool scalar), K
// carried, then body captures. Body outputs: cond_out, K carried, S scan
// outputs (stacked along a new leading axis).
type loopOp struct {
	node     *Node
	body     *Session
	caps     []capRef
	iterName string
	condName string
	carryIn  []string
	condOut  string
	carryOut []string
	scanOut  []string
}

func (o *loopOp) Run(ctx *ops.Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	K := len(o.carryIn)
	if len(in) != 2+K+len(o.node.Caps) {
		return nil, fmt.Errorf("graph: %s: got %d inputs, want %d", o.node, len(in), 2+K+len(o.node.Caps))
	}
	maxIter := int64(math.MaxInt64)
	if in[0] != nil {
		if in[0].DType() != tensor.I64 || in[0].Numel() != 1 {
			return nil, fmt.Errorf("graph: %s: M must be an int64 scalar", o.node)
		}
		maxIter = in[0].I64()[0]
	}
	cond := true
	if in[1] != nil {
		if in[1].DType() != tensor.Bool || in[1].Numel() != 1 {
			return nil, fmt.Errorf("graph: %s: cond must be a bool scalar", o.node)
		}
		cond = in[1].Bool()[0]
	}
	carried := make([]*tensor.Tensor, K)
	copy(carried, in[2:2+K])
	scans := make([][]*tensor.Tensor, len(o.scanOut))
	feeds := make(map[string]*tensor.Tensor, 2+K+len(o.caps))
	for _, c := range o.caps {
		feeds[c.name] = in[c.idx]
	}
	for it := int64(0); it < maxIter && cond; it++ {
		iterT := tensor.FromI64([]int64{it})
		condT := tensor.New(tensor.Bool)
		condT.Bool()[0] = cond
		feeds[o.iterName] = iterT
		feeds[o.condName] = condT
		for i, nm := range o.carryIn {
			feeds[nm] = carried[i]
		}
		outs, err := o.body.Run(feeds)
		if err != nil {
			return nil, fmt.Errorf("graph: %s body (iteration %d): %w", o.node, it, err)
		}
		co := outs[o.condOut]
		if co == nil || co.DType() != tensor.Bool || co.Numel() != 1 {
			return nil, fmt.Errorf("graph: %s: body cond output must be a bool scalar", o.node)
		}
		cond = co.Bool()[0]
		for i, nm := range o.carryOut {
			carried[i] = outs[nm]
		}
		for j, nm := range o.scanOut {
			scans[j] = append(scans[j], outs[nm])
		}
	}
	res := make([]*tensor.Tensor, 0, K+len(o.scanOut))
	res = append(res, carried...)
	for j := range o.scanOut {
		st, err := stackScan(ctx, scans[j])
		if err != nil {
			return nil, fmt.Errorf("graph: %s scan output %d: %w", o.node, j, err)
		}
		res = append(res, st)
	}
	return res, nil
}

// stackScan concatenates per-iteration tensors along a new leading axis.
func stackScan(ctx *ops.Ctx, ts []*tensor.Tensor) (*tensor.Tensor, error) {
	if len(ts) == 0 {
		return nil, fmt.Errorf("scan output with zero iterations (element shape unknown)")
	}
	es := ts[0].Shape()
	out := ctx.NewUninit(ts[0].DType(), append([]int{len(ts)}, es...)...)
	if ts[0].DType() != tensor.F32 && ts[0].DType() != tensor.I64 {
		return nil, fmt.Errorf("scan output dtype %s unsupported", ts[0].DType())
	}
	n := ts[0].Numel()
	for i, t := range ts {
		if !t.Shape().Equal(tensor.Shape(es)) {
			return nil, fmt.Errorf("scan element shapes differ: %v vs %v", t.Shape(), es)
		}
		switch t.DType() {
		case tensor.F32:
			copy(out.F32()[i*n:], t.F32())
		case tensor.I64:
			copy(out.I64()[i*n:], t.I64())
		}
	}
	return out, nil
}

// compileCtrl builds the executor op for an If or Loop node, compiling its
// subgraphs (each gets the full Optimize pipeline).
func compileCtrl(n *Node) (ops.Op, error) {
	switch n.OpType {
	case "If":
		tg, eg := n.Sub["then_branch"], n.Sub["else_branch"]
		if tg == nil || eg == nil {
			return nil, fmt.Errorf("graph: %s: If needs then_branch and else_branch", n)
		}
		if len(tg.Outputs) != len(eg.Outputs) || len(tg.Outputs) != len(n.Outputs) {
			return nil, fmt.Errorf("graph: %s: branch output counts differ (%d/%d vs node %d)", n, len(tg.Outputs), len(eg.Outputs), len(n.Outputs))
		}
		ts, err := Compile(tg)
		if err != nil {
			return nil, fmt.Errorf("graph: %s then_branch: %w", n, err)
		}
		es, err := Compile(eg)
		if err != nil {
			return nil, fmt.Errorf("graph: %s else_branch: %w", n, err)
		}
		tc, err := capRefs(n, tg)
		if err != nil {
			return nil, err
		}
		ec, err := capRefs(n, eg)
		if err != nil {
			return nil, err
		}
		names := func(g *Graph) []string {
			out := make([]string, len(g.Outputs))
			for i, v := range g.Outputs {
				out[i] = v.Name
			}
			return out
		}
		return &ifOp{node: n, then: ts, els: es, thenCaps: tc, elsCaps: ec,
			thenOuts: names(tg), elsOuts: names(eg)}, nil
	case "Loop":
		bg := n.Sub["body"]
		if bg == nil {
			return nil, fmt.Errorf("graph: %s: Loop needs a body", n)
		}
		nFormal := len(bg.Inputs) - len(bg.Captures)
		if nFormal < 2 {
			return nil, fmt.Errorf("graph: %s: Loop body needs iter and cond inputs", n)
		}
		K := nFormal - 2
		if len(n.Inputs)-len(n.Caps) != 2+K {
			return nil, fmt.Errorf("graph: %s: Loop carries %d values but node has %d inputs", n, K, len(n.Inputs)-len(n.Caps))
		}
		if len(bg.Outputs) < 1+K {
			return nil, fmt.Errorf("graph: %s: Loop body needs cond + %d carried outputs", n, K)
		}
		S := len(bg.Outputs) - 1 - K
		if len(n.Outputs) != K+S {
			return nil, fmt.Errorf("graph: %s: node outputs %d, want %d carried + %d scan", n, len(n.Outputs), K, S)
		}
		bs, err := Compile(bg)
		if err != nil {
			return nil, fmt.Errorf("graph: %s body: %w", n, err)
		}
		bc, err := capRefs(n, bg)
		if err != nil {
			return nil, err
		}
		o := &loopOp{node: n, body: bs, caps: bc,
			iterName: bg.Inputs[0].Name, condName: bg.Inputs[1].Name,
			condOut: bg.Outputs[0].Name}
		for i := 0; i < K; i++ {
			o.carryIn = append(o.carryIn, bg.Inputs[2+i].Name)
			o.carryOut = append(o.carryOut, bg.Outputs[1+i].Name)
		}
		for j := 0; j < S; j++ {
			o.scanOut = append(o.scanOut, bg.Outputs[1+K+j].Name)
		}
		return o, nil
	}
	return nil, fmt.Errorf("graph: %s: unsupported control-flow op", n)
}
