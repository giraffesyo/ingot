//go:build darwin && arm64

package graph

import (
	"fmt"
	"math"
	"os"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/ops"
	"github.com/giraffesyo/ingot/tensor"
)

// GPUSession runs a graph with every node whose Metal implementation
// accepts its inputs on the GPU and the rest on the CPU ops. Memory is
// unified: every tensor the session allocates lives in page-aligned memory
// that is also a Metal buffer, so CPU and GPU ops share tensors without
// copies. Consecutive GPU nodes record into one command buffer; it is
// flushed before any CPU node that touches data (views excepted), which
// also makes buffer recycling safe — a pooled buffer is only handed to a
// CPU writer once every GPU reader has run.
//
// The CPU Session is untouched; GPUSession is opt-in (CompileGPU). Runs
// are serialized.
type GPUSession struct {
	*Session
	dev    *metal.Device
	gops   []gpuOp // per step; nil = CPU only
	mem    *pageMem
	mu     sync.Mutex // guards wraps
	runMu  sync.Mutex
	wraps  map[unsafe.Pointer]*metal.Buffer
	stream *metal.Stream
	// side serves CPU nodes that run while GPU work is pending (see
	// sideRun): session memory, so GPU nodes can read their outputs, but
	// apart from the GPU pool, and recycled only after a flush.
	side *tensor.Pool
	// bf16 selects bf16 matrix products for constant weights (GPUBF16).
	bf16 bool
	// tables caches small constant tables GPU nodes derive from shapes
	// (resize taps), in session memory for the session's lifetime.
	tables map[string]metal.Region

	// GPUTime is the last Run's GPU execution time, summed over its
	// command buffers.
	GPUTime time.Duration
	// GPUSteps and CPUSteps count where the last Run placed its nodes;
	// Flushes counts its GPU round trips (mid-graph flushes before CPU
	// nodes, plus the final one).
	GPUSteps, CPUSteps, Flushes int
	// FlushedBy names the CPU nodes that forced mid-graph flushes.
	FlushedBy []string
	// Profile, when set, runs every GPU node in its own command buffer
	// (repeated, then averaged) and accumulates its GPU execution time per
	// op type (OpTime) and per node (NodeTime) — for finding slow kernels,
	// not for production runs (GPUTime then counts the repeats).
	Profile  bool
	OpTime   map[string]time.Duration
	NodeTime map[*Node]time.Duration
	// Check, when set, flushes after every GPU node, re-runs the node's
	// CPU op on the same inputs and records nodes whose outputs differ
	// (relative max error above 1e-3 of the output scale) in Mismatches —
	// a diagnostic for GPU op bugs, not for production runs.
	Check      bool
	Mismatches []Mismatch
}

// Mismatch is a GPU node whose output disagreed with its CPU op (Check).
type Mismatch struct {
	Node   *Node
	Output int
	MaxAbs float64 // largest |gpu − cpu|
	Scale  float64 // largest |cpu|
	Detail string  // shape / dtype disagreement, if any
}

// CompileGPU optimizes g and compiles it for the GPU (darwin/arm64 with
// Metal); nodes without a GPU implementation run their CPU ops.
func CompileGPU(g *Graph, opts ...GPUOption) (*GPUSession, error) {
	if err := metal.Supported(); err != nil {
		return nil, err
	}
	dev, err := metal.Open()
	if err != nil {
		return nil, err
	}
	for _, prep := range []func() error{dev.Prepare, dev.PrepareConv, dev.PrepareEW, dev.PrepareCNN, dev.PrepareIGEMM} {
		if err := prep(); err != nil {
			return nil, err
		}
	}
	optimize(g, false)
	s, err := CompileRaw(g)
	if err != nil {
		return nil, err
	}
	mem := &pageMem{}
	s.pool = tensor.NewPoolAlloc(mem.alloc)
	gs := &GPUSession{Session: s, dev: dev, mem: mem, wraps: map[unsafe.Pointer]*metal.Buffer{}, stream: dev.NewStream(),
		side: tensor.NewPoolAlloc(mem.alloc), tables: map[string]metal.Region{}}
	// Constants move to aligned memory so GPU nodes can read them in place.
	for id, c := range s.constVals {
		if c != nil && c.Numel() > 0 {
			s.constVals[id] = mem.copyOf(c)
		}
	}
	for _, o := range opts {
		o(gs)
	}
	gs.gops = make([]gpuOp, len(s.steps))
	skip := map[string]bool{} // INGOT_GPU_SKIP=OpType,...: keep those on the CPU (bisecting)
	for _, t := range strings.Split(os.Getenv("INGOT_GPU_SKIP"), ",") {
		if t != "" {
			skip[t] = true
		}
	}
	for i, st := range s.steps {
		if !skip[st.node.OpType] && !skip["*"] {
			gs.gops[i] = gpuOpFor(st.node)
		}
	}
	return gs, nil
}

// DispatchCounts starts counting GPU dispatches per kernel across Runs
// and returns the live map (diagnostics).
func (s *GPUSession) DispatchCounts() map[string]int {
	if s.stream.Counts == nil {
		s.stream.Counts = map[string]int{}
	}
	return s.stream.Counts
}

// Close releases the session's GPU buffers and memory. Tensors it returned
// are invalid afterwards.
func (s *GPUSession) Close() {
	for _, b := range s.wraps {
		b.Release()
	}
	s.wraps = nil
	s.mem.free()
}

// viewOps return views of their input and never allocate: they may run on
// the CPU while GPU work is pending.
// (Identity is not one: its CPU op copies — it runs as a GPU copy, see
// identityGPU, or flushes like any data-reading CPU node.)
var viewOps = map[string]bool{"Reshape": true, "Squeeze": true, "Unsqueeze": true, "Flatten": true}

// metaOps read only their input's shape.
var metaOps = map[string]bool{"Shape": true, "Size": true}

// sideRun reports whether CPU step st may run while GPU work is pending:
// it reads no data a pending GPU node writes (shape math over integer
// tensors, or shape queries), and its outputs come from the side pool
// (never a buffer a pending GPU node still reads).
func sideRun(st *step, in []*tensor.Tensor, pending func(id int) bool) bool {
	if st.node.Domain != "" {
		return false
	}
	if metaOps[st.node.OpType] {
		return true
	}
	for k, t := range in {
		if t == nil {
			continue
		}
		if dt := t.DType(); dt == tensor.F32 || dt == tensor.F16 || dt == tensor.BF16 || pending(st.in[k]) {
			return false
		}
	}
	return true
}

// region returns t's storage as a Metal buffer region, wrapping (once) the
// page-aligned allocation it lives in. ok is false for memory the session
// did not allocate.
func (s *GPUSession) region(t *tensor.Tensor) (metal.Region, bool) {
	buf, off := t.Storage()
	if len(buf) == 0 || !s.mem.owns(buf) {
		return metal.Region{}, false
	}
	key := unsafe.Pointer(&buf[0])
	s.mu.Lock()
	defer s.mu.Unlock()
	b, ok := s.wraps[key]
	if !ok {
		var err error
		if b, err = s.dev.Wrap(buf); err != nil {
			return metal.Region{}, false
		}
		s.wraps[key] = b
	}
	return b.At(off), true
}

// checkNode compares a flushed GPU node's outputs with its CPU op (Check).
func (s *GPUSession) checkNode(st *step, in, gpu []*tensor.Tensor) {
	cpu, err := st.op.Run(&ops.Ctx{Pool: tensor.NewPool()}, in)
	if err != nil {
		s.Mismatches = append(s.Mismatches, Mismatch{Node: st.node, Detail: "cpu op: " + err.Error()})
		return
	}
	for k := range min(len(cpu), len(gpu)) {
		c, g := cpu[k], gpu[k]
		if c == nil || g == nil {
			continue
		}
		if !c.Shape().Equal(g.Shape()) || c.DType() != g.DType() {
			s.Mismatches = append(s.Mismatches, Mismatch{Node: st.node, Output: k,
				Detail: fmt.Sprintf("gpu %s%v, cpu %s%v", g.DType(), g.Shape(), c.DType(), c.Shape())})
			continue
		}
		if c.DType() != tensor.F32 {
			continue
		}
		var maxAbs, scale float64
		for i, v := range c.F32() {
			w := g.F32()[i]
			cv, gv := float64(v), float64(w)
			if cv == gv || math.IsNaN(cv) && math.IsNaN(gv) { // incl. matching ±Inf
				if !math.IsInf(cv, 0) && !math.IsNaN(cv) {
					scale = math.Max(scale, math.Abs(cv))
				}
				continue
			}
			d := math.Abs(gv - cv)
			if d > maxAbs || math.IsNaN(d) {
				maxAbs = d
			}
			if !math.IsInf(cv, 0) && !math.IsNaN(cv) {
				scale = math.Max(scale, math.Abs(cv))
			}
		}
		if maxAbs > 1e-3*math.Max(scale, 1e-6) || math.IsNaN(maxAbs) {
			s.Mismatches = append(s.Mismatches, Mismatch{Node: st.node, Output: k, MaxAbs: maxAbs, Scale: scale})
		}
	}
}

// profileRepeat is how many times Profile encodes each GPU node.
const profileRepeat = 10

// table returns data copied once into session memory under key.
func (s *GPUSession) table(key string, data func() []byte) (metal.Region, bool) {
	if r, ok := s.tables[key]; ok {
		return r, true
	}
	b := data()
	buf := s.mem.alloc(len(b))
	copy(buf, b)
	r, ok := s.region(tensor.FromBytes(tensor.U8, buf, len(buf)))
	if ok {
		s.tables[key] = r
	}
	return r, ok
}

// Run executes the graph (see GPUSession).
func (s *GPUSession) Run(feeds map[string]*tensor.Tensor) (res map[string]*tensor.Tensor, err error) {
	s.runMu.Lock()
	defer s.runMu.Unlock()
	vals := make([]*tensor.Tensor, s.nval)
	live := make([]int, s.nval)
	pooled := make([]bool, s.nval)
	alias := make([]int, s.nval)
	copy(vals, s.constVals)
	copy(live, s.uses)
	for i := range alias {
		alias[i] = -1
	}
	defer func() {
		if ferr := s.stream.Flush(); err == nil && ferr != nil {
			err = fmt.Errorf("graph: gpu: %w", ferr)
		}
	}()
	for _, v := range s.g.Inputs {
		t, ok := feeds[v.Name]
		if !ok {
			return nil, fmt.Errorf("graph: missing input %q", v.Name)
		}
		if v.DType != tensor.Invalid && t.DType() != v.DType {
			return nil, fmt.Errorf("graph: input %q: dtype %s, model expects %s", v.Name, t.DType(), v.DType)
		}
		// Feeds move into session memory so GPU nodes can read them.
		c := s.pool.GetUninit(t.DType(), t.Shape()...)
		copy(c.Bytes(), t.Bytes())
		vals[v.id] = c
		pooled[v.id] = !s.isOutput[v.id]
	}
	ctx := &ops.Ctx{Pool: s.pool}
	sideCtx := &ops.Ctx{Pool: s.side}
	// Side outputs are released only at a flush: GPU nodes may read them.
	sideOut := make([]bool, s.nval)
	var deferred []*tensor.Tensor
	s.GPUTime = 0
	flush := func() error {
		pending := s.stream.Pending()
		err := s.stream.Flush()
		if pending {
			s.GPUTime += s.stream.GPUTime()
		}
		for _, t := range deferred {
			s.side.Put(t)
		}
		deferred = deferred[:0]
		return err
	}
	put := func(id int) {
		if sideOut[id] {
			deferred = append(deferred, vals[id])
		} else {
			s.pool.Put(vals[id])
		}
		vals[id] = nil
	}
	gctx := &gpuCtx{s: s}
	// gpuGen[id] == gen marks values written by GPU work not yet flushed.
	gpuGen := make([]int, s.nval)
	gen := 1
	pending := func(id int) bool {
		if id < 0 {
			return false
		}
		if r := alias[id]; r >= 0 && gpuGen[r] == gen {
			return true
		}
		return gpuGen[id] == gen
	}
	s.GPUSteps, s.CPUSteps, s.Flushes, s.FlushedBy = 0, 0, 0, s.FlushedBy[:0]
	in := make([]*tensor.Tensor, 0, 8)
	for si := range s.steps {
		st := &s.steps[si]
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
		var outs []*tensor.Tensor
		placed := false
		if g := s.gops[si]; g != nil {
			var enc func(e *metal.Encoder)
			var ok bool
			if outs, enc, ok = g.prepare(gctx, st, in); ok {
				s.stream.Encode(enc)
				placed = true
				s.GPUSteps++
				if s.Check {
					if err := flush(); err != nil {
						return nil, fmt.Errorf("graph: gpu: %w", err)
					}
					gen++
					s.checkNode(st, in, outs)
				}
				if s.Profile {
					// Repeat the node (encodings are idempotent: every node
					// overwrites its outputs) so its time is measured at
					// busy-GPU clocks, not after an idle gap.
					for range profileRepeat - 1 {
						s.stream.Encode(enc)
					}
					if err := flush(); err != nil {
						return nil, fmt.Errorf("graph: gpu: %w", err)
					}
					gen++
					if s.OpTime == nil {
						s.OpTime, s.NodeTime = map[string]time.Duration{}, map[*Node]time.Duration{}
					}
					s.OpTime[st.node.OpType] += s.stream.GPUTime() / profileRepeat
					s.NodeTime[st.node] += s.stream.GPUTime() / profileRepeat
				}
			}
		}
		side := false
		if !placed {
			runCtx := ctx
			if s.stream.Pending() && !(st.node.Domain == "" && viewOps[st.node.OpType]) {
				if side = sideRun(st, in, pending); side {
					runCtx = sideCtx
				} else {
					s.Flushes++
					s.FlushedBy = append(s.FlushedBy, st.node.OpType)
					if err := flush(); err != nil {
						return nil, fmt.Errorf("graph: gpu: %w", err)
					}
					gen++
				}
			}
			if outs, err = st.op.Run(runCtx, in); err != nil {
				return nil, fmt.Errorf("graph: %w", err)
			}
			s.CPUSteps++
		}
		if len(outs) < len(st.out) {
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
			vals[id] = t
			pooled[id] = !s.isOutput[id]
			sideOut[id] = side
			if placed {
				gpuGen[id] = gen
			}
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
		for _, id := range st.in {
			if id < 0 {
				continue
			}
			live[id]--
			if live[id] == 0 && pooled[id] {
				put(id)
			}
			if r := alias[id]; r >= 0 {
				live[r]--
				if live[r] == 0 && pooled[r] {
					put(r)
				}
			}
		}
	}
	if s.stream.Pending() {
		s.Flushes++
	}
	if err := flush(); err != nil {
		return nil, fmt.Errorf("graph: gpu: %w", err)
	}
	res = make(map[string]*tensor.Tensor, len(s.g.Outputs))
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

// pageMem hands out page-aligned anonymous mappings (a Metal buffer can
// wrap them without copying) and frees them all at once.
type pageMem struct {
	mu    sync.Mutex
	slabs [][]byte
	bases map[uintptr]bool
}

var pageSize = syscall.Getpagesize()

func (m *pageMem) alloc(n int) []byte {
	size := (max(n, 1) + pageSize - 1) / pageSize * pageSize
	b, err := syscall.Mmap(-1, 0, size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		panic(fmt.Sprintf("graph: gpu: mmap %d bytes: %v", size, err))
	}
	m.mu.Lock()
	if m.bases == nil {
		m.bases = map[uintptr]bool{}
	}
	m.slabs = append(m.slabs, b)
	m.bases[uintptr(unsafe.Pointer(&b[0]))] = true
	m.mu.Unlock()
	return b[:n:n]
}

// owns reports whether buf starts at one of m's allocations.
func (m *pageMem) owns(buf []byte) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.bases[uintptr(unsafe.Pointer(&buf[0]))]
}

// copyOf returns t copied into page-aligned memory.
func (m *pageMem) copyOf(t *tensor.Tensor) *tensor.Tensor {
	src := t.Bytes()
	b := m.alloc(len(src))
	copy(b, src)
	return tensor.FromBytes(t.DType(), b, t.Shape()...)
}

func (m *pageMem) free() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, b := range m.slabs {
		syscall.Munmap(b[:cap(b)])
	}
	m.slabs, m.bases = nil, nil
}
