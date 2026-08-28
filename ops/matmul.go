package ops

import (
	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/vek"
	"github.com/giraffesyo/ingot/tensor"
	"os"
	"sync"
)

// gemmOp: Y = alpha*op(A)·op(B) + beta*C, A [M×K], B [K×N], C broadcastable to [M×N].
type gemmOp struct {
	n              NodeInfo
	alpha, beta    float32
	transA, transB bool
	bCache
}

func (o *gemmOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("need A and B")
	}
	a, b := in[0], in[1]
	if a.DType() != tensor.F32 || b.DType() != tensor.F32 || a.Shape().Rank() != 2 || b.Shape().Rank() != 2 {
		return nil, o.n.Errorf("need 2-D f32 A and B, got %v %v", a, b)
	}
	M, K := a.Dim(0), a.Dim(1)
	if o.transA {
		M, K = K, M
	}
	Kb, N := b.Dim(0), b.Dim(1)
	if o.transB {
		Kb, N = N, Kb
	}
	if K != Kb {
		return nil, o.n.Errorf("inner dim mismatch %d vs %d", K, Kb)
	}
	out := ctx.NewUninit(tensor.F32, M, N)
	of := out.F32()
	beta := float32(0)
	if len(in) > 2 && in[2] != nil && o.beta != 0 {
		c := in[2]
		cs := c.Shape()
		cf := c.F32()
		switch {
		case cs.Numel() == M*N && (cs.Rank() == 2 || cs.Rank() == 1 && M == 1):
			copy(of, cf)
		case cs.Numel() == N: // [N] or [1,N]: broadcast over rows
			for i := 0; i < M; i++ {
				copy(of[i*N:(i+1)*N], cf)
			}
		case cs.Numel() == M && cs.Rank() == 2: // [M,1]
			for i := 0; i < M; i++ {
				row := of[i*N : (i+1)*N]
				for j := range row {
					row[j] = cf[i]
				}
			}
		case cs.Numel() == 1:
			for i := range of {
				of[i] = cf[0]
			}
		default:
			return nil, o.n.Errorf("cannot broadcast C %v to [%d,%d]", cs, M, N)
		}
		beta = o.beta
	}
	if !o.transA {
		if pb := o.bCache.get(o.transB, K, N, b.F32(), b.Dim(1)); pb != nil {
			if bp16 := o.bCache.getBF16(b.F32()); bp16 != nil && o.alpha == 1 {
				// bf16 overwrites (beta=0 semantics): the bias preload above
				// is redone as a post-add of beta*C.
				gemm.BgemmWeights(M, a.F32(), a.Dim(1), bp16, of, N, true)
				if beta != 0 {
					c := in[2]
					cs := c.Shape()
					cf := c.F32()
					switch {
					case cs.Numel() == M*N:
						for i := range of {
							of[i] += beta * cf[i]
						}
					case cs.Numel() == N:
						for i := 0; i < M; i++ {
							row := of[i*N : (i+1)*N]
							for j := range row {
								row[j] += beta * cf[j]
							}
						}
					case cs.Numel() == M:
						for i := 0; i < M; i++ {
							vek.AddScalar(of[i*N:(i+1)*N], of[i*N:(i+1)*N], beta*cf[i])
						}
					default: // scalar
						vek.AddScalar(of, of, beta*cf[0])
					}
				}
				return ctx.Out(out), nil
			}
			gemm.SgemmPackedB(M, o.alpha, a.F32(), a.Dim(1), pb, beta, of, N)
			return ctx.Out(out), nil
		}
	}
	gemm.SgemmT(o.transA, o.transB, M, N, K, o.alpha, a.F32(), a.Dim(1), b.F32(), b.Dim(1), beta, of, N)
	return ctx.Out(out), nil
}

// bf16Weights: opt-in bf16 storage+compute for constant MatMul/Gemm weights
// (INGOT_BF16=1), taken only where the bf16 kernel is a measured speedup
// (gemm.BF16Fast: the amd64 VDPBF16PS kernel — 2.6x f32 at transformer
// shapes; Apple's BFMMLA is quarter-rate and never opted in). Accuracy drops
// to bf16 (~3 decimal digits); the flag is for serving, never for
// conformance runs.
var bf16Weights = os.Getenv("INGOT_BF16") == "1" && gemm.BF16Fast()

// bCache caches a pre-packed B for ops whose second operand is the same
// tensor every run (constant weights). A B that ever changes storage marks
// the op dynamic and the cache stays off — packing per call would be a
// pessimisation, and correctness never depends on the cache.
type bCache struct {
	mu  sync.Mutex
	src *float32
	ln  int
	pb  *gemm.PackedB
	bf  *gemm.BPackedB
	dyn bool
}

// get returns the packed form of b (packing on first use), or nil for
// operands that have proven dynamic.
func (c *bCache) get(transB bool, k, n int, b []float32, ldb int) *gemm.PackedB {
	if len(b) == 0 {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.dyn {
		return nil
	}
	if c.src == &b[0] && c.ln == len(b) {
		return c.pb
	}
	if c.src != nil { // storage changed: not a constant
		c.dyn, c.pb, c.bf, c.src = true, nil, nil, nil
		return nil
	}
	c.pb, c.src, c.ln = gemm.PackB(transB, k, n, b, ldb), &b[0], len(b)
	if bf16Weights {
		c.bf = gemm.BPackB(transB, k, n, b, ldb)
	}
	return c.pb
}

// getBF16 returns the bf16 pack for b when the opt-in path is on and b is the
// cached constant (get must have been called for this b first).
func (c *bCache) getBF16(b []float32) *gemm.BPackedB {
	if !bf16Weights || len(b) == 0 {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.dyn || c.src != &b[0] || c.ln != len(b) {
		return nil
	}
	return c.bf
}

// matmulOp: NumPy-style matmul with batch broadcasting.
type matmulOp struct {
	n NodeInfo
	bCache
}

func (o *matmulOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) != 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("need 2 inputs")
	}
	a, b := in[0], in[1]
	if a.DType() != tensor.F32 || b.DType() != tensor.F32 {
		return nil, o.n.Errorf("only f32")
	}
	as, bs := a.Shape(), b.Shape() // read-only below
	if len(as) == 0 || len(bs) == 0 {
		return nil, o.n.Errorf("scalar operands not allowed")
	}
	// Promote 1-D per NumPy rules.
	squeezeM, squeezeN := false, false
	if len(as) == 1 {
		as = tensor.Shape{1, as[0]}
		squeezeM = true
	}
	if len(bs) == 1 {
		bs = tensor.Shape{bs[0], 1}
		squeezeN = true
	}
	M, K := as[len(as)-2], as[len(as)-1]
	Kb, N := bs[len(bs)-2], bs[len(bs)-1]
	if K != Kb {
		return nil, o.n.Errorf("inner dim mismatch: %v · %v", a.Shape(), b.Shape())
	}
	batchA, batchB := as[:len(as)-2], bs[:len(bs)-2]
	var bbuf [8]int
	batch, err := broadcastShapeIn(tensor.Shape(bbuf[:0]), batchA, batchB)
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	var obuf [10]int
	oshape := append(append(tensor.Shape(obuf[:0]), batch...), M, N)
	out := ctx.NewUninit(tensor.F32, oshape...)
	af, bf, of := a.F32(), b.F32(), out.F32()
	nb := batch.Numel()
	if nb == 1 {
		if pb := o.bCache.get(false, K, N, bf, N); pb != nil {
			if bp16 := o.bCache.getBF16(bf); bp16 != nil {
				gemm.BgemmWeights(M, af, K, bp16, of, N, true)
			} else {
				gemm.SgemmPackedB(M, 1, af, K, pb, 0, of, N)
			}
		} else {
			gemm.Sgemm(M, N, K, 1, af, K, bf, N, 0, of, N)
		}
	} else {
		// Per-batch matrix offsets (batch dims may broadcast).
		sc := mmScratchPool.Get().(*mmScratch)
		var sabuf, sbbuf, idxbuf [8]int
		sa := broadcastStridesIn(sabuf[:0], batchA, batch)
		sb := broadcastStridesIn(sbbuf[:0], batchB, batch)
		if cap(sc.offs) < nb {
			sc.offs = make([][2]int, nb)
		}
		offs := sc.offs[:nb]
		idx := idxbuf[:len(batch)]
		if len(batch) > len(idxbuf) {
			idx = make([]int, len(batch))
		}
		for bi := 0; bi < nb; bi++ {
			offA, offB := 0, 0
			for d := range idx {
				offA += idx[d] * sa[d]
				offB += idx[d] * sb[d]
			}
			offs[bi] = [2]int{offA * M * K, offB * K * N}
			for d := len(idx) - 1; d >= 0; d-- {
				idx[d]++
				if idx[d] < batch[d] {
					break
				}
				idx[d] = 0
			}
		}
		// Small per-matrix GEMMs (attention heads: T×T·T×d) are run serially
		// in parallel over the batch; large ones use the parallel GEMM in turn.
		if M*N*K <= 1<<18 && nb > 1 {
			sc.af, sc.bf, sc.of, sc.M, sc.N, sc.K = af, bf, of, M, N, K
			par.Run(nb, max(1, (1<<17)/max(M*N*K, 1)), sc)
			sc.af, sc.bf, sc.of = nil, nil, nil
		} else {
			for bi := 0; bi < nb; bi++ {
				gemm.Sgemm(M, N, K, 1, af[offs[bi][0]:], K, bf[offs[bi][1]:], N, 0, of[bi*M*N:], N)
			}
		}
		mmScratchPool.Put(sc)
	}
	if squeezeM || squeezeN {
		var fbuf [10]int
		fs := append(tensor.Shape(fbuf[:0]), batch...)
		if !squeezeM {
			fs = append(fs, M)
		}
		if !squeezeN {
			fs = append(fs, N)
		}
		out = out.Reshape(fs...)
	}
	return ctx.Out(out), nil
}

// mmScratch carries batched-matmul offsets and the per-batch GEMM task
// (pointer Task: par.Run allocates nothing).
type mmScratch struct {
	offs       [][2]int
	af, bf, of []float32
	M, N, K    int
}

func (m *mmScratch) Run(bi, _ int) {
	gemm.SgemmSerial(m.M, m.N, m.K, 1, m.af[m.offs[bi][0]:], m.K, m.bf[m.offs[bi][1]:], m.N, 0, m.of[bi*m.M*m.N:], m.N)
}

var mmScratchPool = sync.Pool{New: func() any { return new(mmScratch) }}

func init() {
	Register("", "Gemm", 7, func(n NodeInfo) (Op, error) {
		return &gemmOp{
			n:      n,
			alpha:  n.Attrs.Float("alpha", 1),
			beta:   n.Attrs.Float("beta", 1),
			transA: n.Attrs.Int("transA", 0) != 0,
			transB: n.Attrs.Int("transB", 0) != 0,
		}, nil
	})
	Register("", "MatMul", 1, func(n NodeInfo) (Op, error) { return &matmulOp{n: n}, nil })
}
