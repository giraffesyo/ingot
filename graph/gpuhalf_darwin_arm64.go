//go:build darwin && arm64

package graph

import (
	"encoding/binary"

	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/tensor"
)

// Half-precision activations under GPUBF16.
//
// The bf16 matrix products and the fused attention read bf16 operands; a
// value consumed only by those is produced in bf16 directly instead of
// being written as f32 and cast by each consumer — one pass over the
// value less, at half the bytes. The rounding is the one the cast would
// have applied, so results do not change. Such a value keeps its f32
// tensor (shape, pool buffer): the bf16 data sits packed at the start of
// that buffer and gpuCtx.half records it by data pointer, so views
// (Reshape) carry it along. A step that cannot read bf16 — a CPU op, a GPU
// op that declined — gets the value widened in place first (Run).
//
// Likewise a matrix product whose only consumer is a unary activation
// applies it in its epilogue, and the activation node passes its input
// through (gpuCtx.acted).

// planHalf fills fuseAct, actStep, halfOut and halfIn from the placed ops.
func (s *GPUSession) planHalf() {
	type use struct{ step, slot int }
	users := make([][]use, s.nval)
	for i, st := range s.steps {
		for k, id := range st.in {
			if id >= 0 {
				users[id] = append(users[id], use{i, k})
			}
		}
	}
	n := len(s.steps)
	s.fuseAct, s.actStep, s.halfOut, s.halfIn = make([]int, n), make([]bool, n), make([]bool, n), make([][]bool, n)
	out0 := func(i int) int {
		if st := &s.steps[i]; len(st.out) > 0 {
			return st.out[0]
		}
		return -1
	}
	constWeight := func(st *step) bool {
		if len(st.in) < 2 || st.in[1] < 0 {
			return false
		}
		w := s.constVals[st.in[1]]
		return w != nil && w.DType() == tensor.F32 && w.Shape().Rank() == 2 && w.IsContiguous()
	}
	for i := range s.steps {
		st := &s.steps[i]
		switch g := s.gops[i].(type) {
		case gemmGPU:
		case matmulGPU:
			if g.gelu {
				continue
			}
		default:
			continue
		}
		id := out0(i)
		if !constWeight(st) || id < 0 || s.isOutput[id] || len(users[id]) != 1 || users[id][0].slot != 0 {
			continue
		}
		u := users[id][0].step
		if a, ok := s.gops[u].(unaryGPU); ok && unaryAct[a.op] != 0 && out0(u) >= 0 {
			s.fuseAct[i], s.actStep[u] = unaryAct[a.op], true
		}
	}
	// want[id]: 0 unknown, 1 every consumer reads bf16, 2 not.
	want := make([]int8, s.nval)
	var wants func(id int) bool
	reads := func(u use) bool {
		st := &s.steps[u.step]
		if st.node.Domain == "" && viewOps[st.node.OpType] {
			return u.slot == 0 && out0(u.step) >= 0 && wants(out0(u.step))
		}
		switch g := s.gops[u.step].(type) {
		case gemmGPU:
			return u.slot == 0 && constWeight(st)
		case matmulGPU:
			return u.slot == 0 && constWeight(st)
		case sdpaGPU:
			return u.slot < 3 && g.flashLayout() && (len(st.in) < 4 || st.in[3] < 0)
		}
		return false
	}
	wants = func(id int) bool {
		if want[id] == 0 {
			want[id] = 1
			if s.isOutput[id] || len(users[id]) == 0 {
				want[id] = 2
			}
			for _, u := range users[id] {
				if !reads(u) {
					want[id] = 2
					break
				}
			}
		}
		return want[id] == 1
	}
	for i := range s.steps {
		st := &s.steps[i]
		s.halfIn[i] = make([]bool, len(st.in))
		for k, id := range st.in {
			s.halfIn[i][k] = id >= 0 && reads(use{i, k})
		}
		if id := out0(i); id >= 0 {
			s.halfOut[i] = wants(id)
			if s.fuseAct[i] != 0 { // the output that matters is the activation's
				s.halfOut[i] = wants(out0(users[id][0].step))
			}
		}
	}
}

// unaryAct maps the unary GPU ops a matrix product's epilogue can apply to
// their activation codes (0: not fusable; metal.UnRelu is 0, so Relu is
// keyed explicitly).
var unaryAct = map[int]int{
	metal.UnRelu: metal.ActRelu, metal.UnSigmoid: metal.ActSigmoid, metal.UnSiLU: metal.ActSiLU,
	metal.UnGeluErf: metal.ActGeluErf, metal.UnGeluTanh: metal.ActGeluTanh,
}

// dataPtr identifies a non-empty tensor's storage (shared by its views).
func dataPtr(t *tensor.Tensor) *byte {
	if t == nil || t.Numel() == 0 {
		return nil
	}
	return &t.Bytes()[0]
}

// isHalf reports whether t currently holds bf16 data (see planHalf).
func (c *gpuCtx) isHalf(t *tensor.Tensor) bool { return len(c.half) > 0 && c.half[dataPtr(t)] }

// wantHalf reports whether the current step should write its first output
// as bf16; the op then calls markHalf on the output it allocated.
func (c *gpuCtx) wantHalf() bool { return c.s.halfOut != nil && c.s.halfOut[c.si] && !c.s.Check }

func (c *gpuCtx) markHalf(out *tensor.Tensor) { c.half[dataPtr(out)] = true }

// epilogue is what the current step's matrix product applies to out
// beyond its bias: the following activation node's function (which then
// passes through) and bf16 rounding.
func (c *gpuCtx) epilogue(out *tensor.Tensor) (act int, bf16 bool) {
	if c.s.fuseAct == nil || c.s.Check {
		return 0, false
	}
	if act = c.s.fuseAct[c.si]; act != 0 {
		c.acted[dataPtr(out)] = true
	}
	if c.s.halfOut[c.si] {
		c.markHalf(out)
		bf16 = true
	}
	return act, bf16
}

// widenBF16 expands the bf16 values packed at the start of b to f32 over
// the whole of b, in place (back to front: element i's f32 bytes never
// cover a bf16 element still to be read).
func widenBF16(b []byte) {
	for i := len(b)/4 - 1; i >= 0; i-- {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(binary.LittleEndian.Uint16(b[2*i:]))<<16)
	}
}
