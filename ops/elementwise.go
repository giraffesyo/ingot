package ops

import (
	"math"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/vek"
	"github.com/giraffesyo/ingot/tensor"
)

// ---- binary arithmetic with broadcasting ----

type binaryOp struct {
	n    NodeInfo
	fn   func(x, y float32) float32
	kind byte // '+', '-', '*', '/' for specialised loops, 0 otherwise
}

func (o *binaryOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) != 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("expected 2 inputs")
	}
	a, b := in[0], in[1]
	if a.DType() == tensor.I64 && b.DType() == tensor.I64 {
		return o.runI64(ctx, a, b)
	}
	if a.DType() != tensor.F32 || b.DType() != tensor.F32 {
		return nil, o.n.Errorf("unsupported dtypes %s, %s", a.DType(), b.DType())
	}
	if o.kind != 0 {
		if out := binaryFast(ctx, a, b, o.kind); out != nil {
			return []*tensor.Tensor{out}, nil
		}
	}
	out, err := binaryF32(ctx, a, b, o.fn)
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	return []*tensor.Tensor{out}, nil
}

// binaryFast handles same-shape and scalar-broadcast cases for + - * / with
// vek kernels (chunked in parallel for large tensors); returns nil for shapes
// that need the generic broadcasting path.
func binaryFast(ctx *Ctx, a, b *tensor.Tensor, kind byte) *tensor.Tensor {
	af, bf := a.F32(), b.F32()
	switch {
	case a.Shape().Equal(b.Shape()):
		out := ctx.NewUninit(tensor.F32, a.Shape()...)
		of := out.F32()
		binParallel(len(of), func(lo, hi int) {
			d, x, y := of[lo:hi], af[lo:hi], bf[lo:hi]
			switch kind {
			case '+':
				vek.Add(d, x, y)
			case '-':
				vek.Sub(d, x, y)
			case '*':
				vek.Mul(d, x, y)
			case '/':
				vek.Div(d, x, y)
			}
		})
		return out
	case len(bf) == 1:
		out := ctx.NewUninit(tensor.F32, a.Shape()...)
		of := out.F32()
		y := bf[0]
		binParallel(len(of), func(lo, hi int) {
			d, x := of[lo:hi], af[lo:hi]
			switch kind {
			case '+':
				vek.AddScalar(d, x, y)
			case '-':
				vek.AddScalar(d, x, -y)
			case '*':
				vek.MulScalar(d, x, y)
			case '/':
				vek.MulScalar(d, x, 1/y)
			}
		})
		return out
	case len(af) == 1:
		// scalar OP vector: only + and * are order-independent.
		if kind != '+' && kind != '*' {
			return nil
		}
		out := ctx.NewUninit(tensor.F32, b.Shape()...)
		of := out.F32()
		x := af[0]
		binParallel(len(of), func(lo, hi int) {
			d, y := of[lo:hi], bf[lo:hi]
			if kind == '+' {
				vek.AddScalar(d, y, x)
			} else {
				vek.MulScalar(d, y, x)
			}
		})
		return out
	}
	// Per-block broadcast: one operand is 1 in a trailing suffix of dims and
	// matches the other in the leading prefix (e.g. squeeze-excite scale
	// [N,C,1,1] · activation [N,C,H,W]). Each small-operand element then scales
	// a contiguous block; use the scalar vek kernels per block.
	if out := blockBroadcastFast(ctx, a, b, kind); out != nil {
		return out
	}
	// Row broadcast: the small operand equals a trailing suffix of the big
	// operand's shape (bias add [N,T,C]+[C], LayerNorm scale, ...): each row of
	// the big operand combines with the whole small operand.
	if out := rowBroadcastFast(ctx, a, b, kind); out != nil {
		return out
	}
	return nil
}

// rowBroadcastFast handles a·b where the smaller operand (leading 1s dropped)
// equals a trailing suffix of the larger's shape, using same-shape vek kernels
// per row. Order-dependent ops (- /) are only taken when the big operand is
// on the left.
func rowBroadcastFast(ctx *Ctx, a, b *tensor.Tensor, kind byte) *tensor.Tensor {
	big, small, swapped := a, b, false
	if a.Numel() < b.Numel() {
		big, small, swapped = b, a, true
	}
	if swapped && (kind == '-' || kind == '/') {
		return nil
	}
	bs, ss := big.Shape(), small.Shape()
	for len(ss) > 0 && ss[0] == 1 {
		ss = ss[1:]
	}
	if len(ss) == 0 || len(ss) > len(bs) {
		return nil
	}
	for i := range ss {
		if ss[i] != bs[len(bs)-len(ss)+i] {
			return nil
		}
	}
	row := small.Numel()
	nRows := big.Numel() / row
	if nRows < 2 {
		return nil
	}
	out := ctx.NewUninit(tensor.F32, bs...)
	of, bigf, smf := out.F32(), big.F32(), small.F32()
	grain := max(1, unaryChunk/row)
	if len(of) <= 2*unaryChunk {
		grain = nRows // tiny: stay on the caller
	}
	par.For(nRows, grain, func(r, _ int) {
		d, x := of[r*row:(r+1)*row], bigf[r*row:(r+1)*row]
		switch kind {
		case '+':
			vek.Add(d, x, smf)
		case '*':
			vek.Mul(d, x, smf)
		case '-':
			vek.Sub(d, x, smf)
		case '/':
			vek.Div(d, x, smf)
		}
	})
	return out
}

// blockBroadcastFast handles a·b where the smaller operand broadcasts over a
// contiguous trailing block of the larger (prefix dims equal, suffix dims 1).
func blockBroadcastFast(ctx *Ctx, a, b *tensor.Tensor, kind byte) *tensor.Tensor {
	big, small, swapped := a, b, false
	if a.Numel() < b.Numel() {
		big, small, swapped = b, a, true
	}
	bs, ss := big.Shape(), small.Shape()
	if len(ss) != len(bs) {
		return nil
	}
	// Require small dims to equal big in a prefix and be 1 in the suffix.
	block := 1
	suffix := true
	for i := len(bs) - 1; i >= 0; i-- {
		if suffix && ss[i] == 1 {
			block *= bs[i]
			continue
		}
		suffix = false
		if ss[i] != bs[i] {
			return nil
		}
	}
	if block == 1 {
		return nil // no benefit; same-shape path would have caught equal shapes
	}
	nBlocks := small.Numel()
	out := ctx.NewUninit(tensor.F32, bs...)
	of, bigf, smf := out.F32(), big.F32(), small.F32()
	// scalar OP block: - and / are not order-independent, so only handle them
	// when the big operand is the left (a) operand.
	rev := swapped // small operand is on the left of the original expression
	apply := func(d, x []float32, s float32) {
		switch kind {
		case '+':
			vek.AddScalar(d, x, s)
		case '*':
			vek.MulScalar(d, x, s)
		case '-':
			if rev {
				// s - x
				vek.MulScalar(d, x, -1)
				vek.AddScalar(d, d, s)
			} else {
				vek.AddScalar(d, x, -s)
			}
		case '/':
			if rev {
				return // s / x: no scalar kernel; fall back handled by caller
			}
			vek.MulScalar(d, x, 1/s)
		}
	}
	if kind == '/' && rev {
		return nil
	}
	grain := max(1, unaryChunk/max(block, 1))
	if len(of) <= 2*unaryChunk {
		grain = nBlocks // tiny: stay on the caller
	}
	par.For(nBlocks, grain, func(k, _ int) {
		lo := k * block
		apply(of[lo:lo+block], bigf[lo:lo+block], smf[k])
	})
	return out
}

// binParallel splits [0,n) into unaryChunk-sized pieces and runs fn in parallel
// for large n, inline otherwise.
func binParallel(n int, fn func(lo, hi int)) {
	if n <= 2*unaryChunk {
		fn(0, n)
		return
	}
	chunks := (n + unaryChunk - 1) / unaryChunk
	par.For(chunks, 1, func(c, _ int) {
		lo := c * unaryChunk
		fn(lo, min(lo+unaryChunk, n))
	})
}

// runI64 handles integer shape arithmetic (Add/Sub/Mul/Div on int64), which
// appears in shape-computation subgraphs exported from PyTorch.
func (o *binaryOp) runI64(ctx *Ctx, a, b *tensor.Tensor) ([]*tensor.Tensor, error) {
	os, err := broadcastShape(a.Shape(), b.Shape())
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	out := ctx.New(tensor.I64, os...)
	ast := broadcastStrides(a.Shape(), os)
	bst := broadcastStrides(b.Shape(), os)
	ai, bi, oi := a.I64(), b.I64(), out.I64()
	idx := make([]int, len(os))
	for k := range oi {
		offA, offB := 0, 0
		for d := range idx {
			offA += idx[d] * ast[d]
			offB += idx[d] * bst[d]
		}
		x, y := ai[offA], bi[offB]
		var r int64
		switch o.n.OpType {
		case "Add":
			r = x + y
		case "Sub":
			r = x - y
		case "Mul":
			r = x * y
		case "Div":
			if y == 0 {
				return nil, o.n.Errorf("integer division by zero")
			}
			r = x / y
		default:
			return nil, o.n.Errorf("int64 not supported for %s", o.n.OpType)
		}
		oi[k] = r
		for d := len(idx) - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < os[d] {
				break
			}
			idx[d] = 0
		}
	}
	return []*tensor.Tensor{out}, nil
}

// ---- unary elementwise ----

// unaryOp applies vec(dst, src) over chunks in parallel. vec is a whole-slice
// kernel (no per-element indirect call).
type unaryOp struct {
	n   NodeInfo
	vec func(dst, src []float32)
}

// vecOf lifts a per-element function to a slice kernel (for rare ops).
func vecOf(fn func(float32) float32) func(dst, src []float32) {
	return func(dst, src []float32) {
		dst = dst[:len(src)]
		for i, v := range src {
			dst[i] = fn(v)
		}
	}
}

// unaryChunk is the per-task element count for parallel elementwise ops.
const unaryChunk = 16384

func (o *unaryOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 1 || in[0] == nil {
		return nil, o.n.Errorf("missing input")
	}
	x := in[0]
	if x.DType() != tensor.F32 {
		return nil, o.n.Errorf("unsupported dtype %s", x.DType())
	}
	out := ctx.NewUninit(tensor.F32, x.Shape()...)
	xf, of := x.F32(), out.F32()
	n := len(of)
	if n <= 2*unaryChunk {
		o.vec(of, xf)
		return []*tensor.Tensor{out}, nil
	}
	chunks := (n + unaryChunk - 1) / unaryChunk
	par.For(chunks, 1, func(c, _ int) {
		lo := c * unaryChunk
		hi := min(lo+unaryChunk, n)
		o.vec(of[lo:hi], xf[lo:hi])
	})
	return []*tensor.Tensor{out}, nil
}

func hardSigmoidVec(alpha, beta float32) func(dst, src []float32) {
	return func(dst, src []float32) { vek.HardSigmoid(dst, src, alpha, beta) }
}

func sigmoidVec(dst, src []float32) { vek.Sigmoid(dst, src) }

func gelu(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
}

func geluTanh(x float32) float32 {
	const c = 0.7978845608028654 // sqrt(2/pi)
	return 0.5 * x * (1 + float32(math.Tanh(c*float64(x+0.044715*x*x*x))))
}

func init() {
	bin := func(name string, kind byte, fn func(x, y float32) float32) {
		Register("", name, 7, func(n NodeInfo) (Op, error) { return &binaryOp{n: n, fn: fn, kind: kind}, nil })
	}
	bin("Add", '+', func(x, y float32) float32 { return x + y })
	bin("Sub", '-', func(x, y float32) float32 { return x - y })
	bin("Mul", '*', func(x, y float32) float32 { return x * y })
	bin("Div", '/', func(x, y float32) float32 { return x / y })
	bin("Pow", 0, func(x, y float32) float32 {
		switch y {
		case 2:
			return x * x
		case 0.5:
			return float32(math.Sqrt(float64(x)))
		}
		return float32(math.Pow(float64(x), float64(y)))
	})
	bin("Max", 0, func(x, y float32) float32 { return max(x, y) })
	bin("Min", 0, func(x, y float32) float32 { return min(x, y) })

	un := func(name string, since int, fn func(x float32) float32) {
		v := vecOf(fn)
		Register("", name, since, func(n NodeInfo) (Op, error) { return &unaryOp{n, v}, nil })
	}
	Register("", "Relu", 6, func(n NodeInfo) (Op, error) { return &unaryOp{n, vek.Relu}, nil })
	Register("", "Sigmoid", 6, func(n NodeInfo) (Op, error) { return &unaryOp{n, sigmoidVec}, nil })
	Register("", "HardSwish", 14, func(n NodeInfo) (Op, error) { return &unaryOp{n, vek.HardSwish}, nil })
	// Runtime-internal fused op produced by the graph optimizer (domain "ingot").
	Register("ingot", "HardSwish", 1, func(n NodeInfo) (Op, error) { return &unaryOp{n, vek.HardSwish}, nil })
	Register("ingot", "SiLU", 1, func(n NodeInfo) (Op, error) { return &unaryOp{n, vek.SiLU}, nil })
	un("Tanh", 6, func(x float32) float32 { return float32(math.Tanh(float64(x))) })
	Register("", "Exp", 6, func(n NodeInfo) (Op, error) { return &unaryOp{n, vek.Exp}, nil })
	un("Log", 6, func(x float32) float32 { return float32(math.Log(float64(x))) })
	un("Sqrt", 6, func(x float32) float32 { return float32(math.Sqrt(float64(x))) })
	un("Abs", 6, func(x float32) float32 { return float32(math.Abs(float64(x))) })
	un("Neg", 6, func(x float32) float32 { return -x })
	un("Erf", 9, func(x float32) float32 { return float32(math.Erf(float64(x))) })
	un("Reciprocal", 6, func(x float32) float32 { return 1 / x })
	un("Floor", 6, func(x float32) float32 { return float32(math.Floor(float64(x))) })
	un("Ceil", 6, func(x float32) float32 { return float32(math.Ceil(float64(x))) })
	un("Round", 11, func(x float32) float32 { return float32(math.RoundToEven(float64(x))) })
	un("Softplus", 1, func(x float32) float32 { return float32(math.Log1p(math.Exp(float64(x)))) })
	un("Mish", 18, func(x float32) float32 {
		return x * float32(math.Tanh(math.Log1p(math.Exp(float64(x)))))
	})
	un("Identity", 1, func(x float32) float32 { return x }) // replaced below for non-f32

	Register("", "HardSigmoid", 6, func(n NodeInfo) (Op, error) {
		return &unaryOp{n, hardSigmoidVec(n.Attrs.Float("alpha", 0.2), n.Attrs.Float("beta", 0.5))}, nil
	})
	Register("", "LeakyRelu", 6, func(n NodeInfo) (Op, error) {
		a := n.Attrs.Float("alpha", 0.01)
		return &unaryOp{n, func(dst, src []float32) { vek.LeakyRelu(dst, src, a) }}, nil
	})
	Register("", "Elu", 6, func(n NodeInfo) (Op, error) {
		a := n.Attrs.Float("alpha", 1)
		return &unaryOp{n, vecOf(func(x float32) float32 {
			if x < 0 {
				return a * (float32(math.Exp(float64(x))) - 1)
			}
			return x
		})}, nil
	})
	Register("", "Gelu", 20, func(n NodeInfo) (Op, error) {
		if n.Attrs.String("approximate", "none") == "tanh" {
			return &unaryOp{n, vecOf(geluTanh)}, nil
		}
		return &unaryOp{n, vecOf(gelu)}, nil
	})
	Register("", "Identity", 1, func(n NodeInfo) (Op, error) { return identityOp{n}, nil })
	Register("", "Clip", 11, func(n NodeInfo) (Op, error) { return &clipOp{n: n}, nil })
	Register("", "Clip", 6, func(n NodeInfo) (Op, error) {
		return &clipOp{n: n, lo: n.Attrs.Float("min", float32(math.Inf(-1))), hi: n.Attrs.Float("max", float32(math.Inf(1))), attr: true}, nil
	})
}

type identityOp struct{ n NodeInfo }

func (o identityOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 1 || in[0] == nil {
		return nil, o.n.Errorf("missing input")
	}
	return []*tensor.Tensor{in[0].Clone()}, nil
}

// clipOp: opset>=11 takes min/max as optional inputs; opset 6 as attributes.
type clipOp struct {
	n      NodeInfo
	lo, hi float32
	attr   bool
}

func (o *clipOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 1 || in[0] == nil {
		return nil, o.n.Errorf("missing input")
	}
	lo, hi := o.lo, o.hi
	if !o.attr {
		lo, hi = float32(math.Inf(-1)), float32(math.Inf(1))
		if len(in) > 1 && in[1] != nil {
			lo = in[1].F32()[0]
		}
		if len(in) > 2 && in[2] != nil {
			hi = in[2].F32()[0]
		}
	}
	x := in[0]
	if x.DType() != tensor.F32 {
		return nil, o.n.Errorf("unsupported dtype %s", x.DType())
	}
	out := ctx.NewUninit(tensor.F32, x.Shape()...)
	xf, of := x.F32(), out.F32()
	binParallel(len(of), func(l, h int) { vek.Clip(of[l:h], xf[l:h], lo, hi) })
	return []*tensor.Tensor{out}, nil
}
