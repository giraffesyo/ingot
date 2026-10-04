package ops

import (
	"sync"

	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/tensor"
)

// sparseConvOp (ingot.SparseConv) is a submanifold sparse convolution over
// a voxel set: X [N, Ci] holds one feature row per occupied cell, nbr
// [N, V] (int32) the row of each of a cell's V kernel neighbours or -1
// where that neighbour is empty (sparse.Neighbors), W [Co, V, Ci] — any
// shape of Co·V·Ci elements with Co first — and an optional bias [Co].
//
//	out[i] = bias + Σ_v W[:, v, :] · X[nbr[i, v]]
//
// The output keeps the input's cells: [N, Co]. It runs as gather + GEMM
// over row tiles — the column matrix [rows, V·Ci] of a tile is gathered
// (zeros for empty neighbours) and multiplied against the weights, packed
// once.
type sparseConvOp struct {
	n NodeInfo

	packMu  sync.Mutex
	pb      *gemm.PackedB
	packSrc *float32
	packLen int
}

// sparseColFloats bounds a tile's column matrix (64 MB).
const sparseColFloats = 1 << 24

func (o *sparseConvOp) packedWeights(wf []float32, Co, K int) *gemm.PackedB {
	o.packMu.Lock()
	defer o.packMu.Unlock()
	if o.packSrc != &wf[0] || o.packLen != len(wf) {
		o.pb, o.packSrc, o.packLen = gemm.PackB(true, K, Co, wf, K), &wf[0], len(wf)
	}
	return o.pb
}

func (o *sparseConvOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 3 || in[0] == nil || in[1] == nil || in[2] == nil {
		return nil, o.n.Errorf("need X, neighbours and W")
	}
	x, nbr, w := in[0], in[1], in[2]
	if x.DType() != tensor.F32 || w.DType() != tensor.F32 || nbr.DType() != tensor.I32 {
		return nil, o.n.Errorf("want f32 X and W and int32 neighbours, got %s %s %s", x.DType(), w.DType(), nbr.DType())
	}
	xs, ns, ws := x.Shape(), nbr.Shape(), w.Shape()
	if len(xs) != 2 || len(ns) != 2 || ns[0] != xs[0] || len(ws) < 2 {
		return nil, o.n.Errorf("want X [N, Ci], neighbours [N, V], W [Co, …], got %v %v %v", xs, ns, ws)
	}
	N, Ci, V, Co := xs[0], xs[1], ns[1], ws[0]
	K := V * Ci
	if w.Numel() != Co*K {
		return nil, o.n.Errorf("W %v does not hold Co·V·Ci = %d·%d·%d weights", ws, Co, V, Ci)
	}
	var bias []float32
	if len(in) > 3 && in[3] != nil {
		if bias = in[3].F32(); len(bias) != Co {
			return nil, o.n.Errorf("bias has %d entries for %d output channels", len(bias), Co)
		}
	}
	out := ctx.NewUninit(tensor.F32, N, Co)
	if N == 0 {
		return ctx.Out(out), nil
	}
	xf, nf, of := x.F32(), nbr.I32(), out.F32()
	pb := o.packedWeights(w.F32(), Co, K)
	rows := min(N, max(1, sparseColFloats/K))
	col := ctx.NewUninit(tensor.F32, rows, K)
	cf := col.F32()
	epi := gemm.Epilogue{Bias: bias}
	var bad error
	for n0 := 0; n0 < N; n0 += rows {
		m := min(rows, N-n0)
		var oob sync.Once
		par.For(m, max(1, 16384/K), func(i, _ int) {
			dst := cf[i*K : (i+1)*K]
			for v, j := range nf[(n0+i)*V : (n0+i+1)*V] {
				switch {
				case j < 0:
					clear(dst[v*Ci : (v+1)*Ci])
				case int(j) >= N:
					oob.Do(func() { bad = o.n.Errorf("neighbour row %d out of range [0, %d)", j, N) })
					clear(dst[v*Ci : (v+1)*Ci])
				default:
					copy(dst[v*Ci:(v+1)*Ci], xf[int(j)*Ci:])
				}
			}
		})
		if bad != nil {
			break
		}
		if bias != nil {
			gemm.SgemmPackedBEpi(m, cf, K, pb, of[n0*Co:], Co, &epi)
		} else {
			gemm.SgemmPackedB(m, 1, cf, K, pb, 0, of[n0*Co:], Co)
		}
	}
	if ctx.Pool != nil {
		ctx.Pool.Put(col)
	}
	if bad != nil {
		return nil, bad
	}
	return ctx.Out(out), nil
}

func init() {
	Register("ingot", "SparseConv", 1, func(n NodeInfo) (Op, error) { return &sparseConvOp{n: n}, nil })
}
