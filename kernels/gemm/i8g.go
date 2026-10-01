package gemm

import (
	"math"
	"sync"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/vek"
)

// I8GWeights is an [N×K] weight matrix quantised to int8 for decode: per
// output row, groups of vek.I8Group weights share one f32 scale
// (symmetric, w ≈ s·q, q in [-127, 127]) — half the bytes of bf16, the
// bandwidth that bounds m=1 GEMV.
type I8GWeights struct {
	N, K int
	Q    []int8    // [N×K]
	S    []float32 // [N×K/I8Group]
}

// QuantizeI8GBF16 quantises bf16 weights (bits, row stride ldw) of shape
// [n×k]; k must be a multiple of vek.I8Group.
func QuantizeI8GBF16(w []uint16, ldw, n, k int) *I8GWeights {
	return quantizeI8G(n, k, func(r, c int) float32 { return math.Float32frombits(uint32(w[r*ldw+c]) << 16) })
}

// QuantizeI8GF32 quantises f32 weights [n×k] (row stride ldw).
func QuantizeI8GF32(w []float32, ldw, n, k int) *I8GWeights {
	return quantizeI8G(n, k, func(r, c int) float32 { return w[r*ldw+c] })
}

func quantizeI8G(n, k int, at func(r, c int) float32) *I8GWeights {
	G := vek.I8Group
	if k%G != 0 {
		panic("gemm: I8G quantisation needs K a multiple of the group size")
	}
	q := &I8GWeights{N: n, K: k, Q: make([]int8, n*k), S: make([]float32, n*k/G)}
	par.For(n, 16, func(r, _ int) {
		for g := range k / G {
			var amax float32
			for c := g * G; c < (g+1)*G; c++ {
				amax = max(amax, float32(math.Abs(float64(at(r, c)))))
			}
			s := amax / 127
			q.S[r*(k/G)+g] = s
			if s == 0 {
				continue
			}
			for c := g * G; c < (g+1)*G; c++ {
				v := math.RoundToEven(float64(at(r, c) / s))
				q.Q[r*k+c] = int8(max(-127, min(127, v)))
			}
		}
	})
	return q
}

type q8Buf struct {
	xh, xl []int8
	sx     []float32
}

// q8Scratch holds the per-call activation quantisation buffers.
var q8Scratch = sync.Pool{New: func() any { return new(q8Buf) }}

// GemvI8G computes y[j] = alpha·Σ_k x[k]·W[j,k] + beta·y[j] for quantised
// weights (beta == 0 overwrites y). x is split into ~14-bit fixed point
// per group (vek.QuantizeX16), so every group is two int8 dots and the
// weight rounding is the only approximation that matters.
func GemvI8G(y []float32, w *I8GWeights, x []float32, alpha, beta float32) {
	K, gk := w.K, w.K/vek.I8Group
	sc := q8Scratch.Get().(*q8Buf)
	defer q8Scratch.Put(sc)
	if cap(sc.xh) < K {
		sc.xh, sc.xl, sc.sx = make([]int8, K), make([]int8, K), make([]float32, gk)
	}
	xh, xl, sx := sc.xh[:K], sc.xl[:K], sc.sx[:gk]
	vek.QuantizeX16(xh, xl, sx, x[:K])
	grain := max(1, minTaskMACs/max(K, 1))
	par.For(w.N, grain, func(j, _ int) {
		v := alpha * vek.DotQ8(w.Q[j*K:(j+1)*K], xh, xl, sx, w.S[j*gk:(j+1)*gk])
		if beta != 0 {
			v += beta * y[j]
		}
		y[j] = v
	})
}
