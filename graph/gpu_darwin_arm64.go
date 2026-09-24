//go:build darwin && arm64

package graph

import (
	"fmt"
	"sync"
	"syscall"
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
// The CPU Session is untouched; GPUSession is opt-in (CompileGPU). Not safe
// for concurrent Runs.
type GPUSession struct {
	*Session
	dev    *metal.Device
	gops   []gpuOp // per step; nil = CPU only
	mem    *pageMem
	mu     sync.Mutex
	wraps  map[unsafe.Pointer]*metal.Buffer
	stream *metal.Stream

	// GPUSteps and CPUSteps count where the last Run placed its nodes;
	// Flushes counts its GPU round trips (mid-graph flushes before CPU
	// nodes, plus the final one).
	GPUSteps, CPUSteps, Flushes int
	// FlushedBy names the CPU nodes that forced mid-graph flushes.
	FlushedBy []string
}

// CompileGPU optimizes g and compiles it for the GPU (darwin/arm64 with
// Metal); nodes without a GPU implementation run their CPU ops.
func CompileGPU(g *Graph) (*GPUSession, error) {
	dev, err := metal.Open()
	if err != nil {
		return nil, err
	}
	for _, prep := range []func() error{dev.Prepare, dev.PrepareConv, dev.PrepareEW, dev.PrepareCNN} {
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
	gs := &GPUSession{Session: s, dev: dev, mem: mem, wraps: map[unsafe.Pointer]*metal.Buffer{}, stream: dev.NewStream()}
	// Constants move to aligned memory so GPU nodes can read them in place.
	for id, c := range s.constVals {
		if c != nil && c.Numel() > 0 {
			s.constVals[id] = mem.copyOf(c)
		}
	}
	gs.gops = make([]gpuOp, len(s.steps))
	for i, st := range s.steps {
		gs.gops[i] = gpuOpFor(st.node)
	}
	return gs, nil
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
var viewOps = map[string]bool{"Reshape": true, "Squeeze": true, "Unsqueeze": true, "Flatten": true, "Identity": true}

// metaOps read only their input's shape.
var metaOps = map[string]bool{"Shape": true, "Size": true}

// sideRun reports whether CPU step st may run while GPU work is pending:
// it reads no data a pending GPU node writes (shape math over integer
// tensors, or shape queries), and its outputs come from heap memory
// (never a pool buffer a pending GPU node still reads).
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

// Run executes the graph (see GPUSession).
func (s *GPUSession) Run(feeds map[string]*tensor.Tensor) (res map[string]*tensor.Tensor, err error) {
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
	sideCtx := &ops.Ctx{Pool: tensor.NewPool()}
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
					if err := s.stream.Flush(); err != nil {
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
			pooled[id] = !s.isOutput[id] && !side
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
	if s.stream.Pending() {
		s.Flushes++
	}
	if err := s.stream.Flush(); err != nil {
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
