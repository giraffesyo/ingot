package ops

import (
	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/tensor"
)

// gemmOp: Y = alpha*op(A)·op(B) + beta*C, A [M×K], B [K×N], C broadcastable to [M×N].
type gemmOp struct {
	n              NodeInfo
	alpha, beta    float32
	transA, transB bool
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
	gemm.SgemmT(o.transA, o.transB, M, N, K, o.alpha, a.F32(), a.Dim(1), b.F32(), b.Dim(1), beta, of, N)
	return []*tensor.Tensor{out}, nil
}

// matmulOp: NumPy-style matmul with batch broadcasting.
type matmulOp struct{ n NodeInfo }

func (o *matmulOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) != 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("need 2 inputs")
	}
	a, b := in[0], in[1]
	if a.DType() != tensor.F32 || b.DType() != tensor.F32 {
		return nil, o.n.Errorf("only f32")
	}
	as, bs := a.Shape().Clone(), b.Shape().Clone()
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
	batch, err := broadcastShape(batchA, batchB)
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	oshape := append(batch.Clone(), M, N)
	out := ctx.NewUninit(tensor.F32, oshape...)
	af, bf, of := a.F32(), b.F32(), out.F32()
	nb := batch.Numel()
	if nb == 1 {
		gemm.Sgemm(M, N, K, 1, af, K, bf, N, 0, of, N)
	} else {
		// Per-batch matrix offsets (batch dims may broadcast).
		sa := broadcastStrides(batchA, batch)
		sb := broadcastStrides(batchB, batch)
		offs := make([][2]int, nb)
		idx := make([]int, len(batch))
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
			par.For(nb, max(1, (1<<17)/max(M*N*K, 1)), func(bi, _ int) {
				gemm.SgemmSerial(M, N, K, 1, af[offs[bi][0]:], K, bf[offs[bi][1]:], N, 0, of[bi*M*N:], N)
			})
		} else {
			for bi := 0; bi < nb; bi++ {
				gemm.Sgemm(M, N, K, 1, af[offs[bi][0]:], K, bf[offs[bi][1]:], N, 0, of[bi*M*N:], N)
			}
		}
	}
	if squeezeM || squeezeN {
		fs := batch.Clone()
		if !squeezeM {
			fs = append(fs, M)
		}
		if !squeezeN {
			fs = append(fs, N)
		}
		out = out.Reshape(fs...)
	}
	return []*tensor.Tensor{out}, nil
}

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
	Register("", "MatMul", 1, func(n NodeInfo) (Op, error) { return &matmulOp{n}, nil })
}
