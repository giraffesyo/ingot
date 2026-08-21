package ops

import (
	"math"

	"github.com/giraffesyo/ocr/tensor"
)

// ---- binary arithmetic with broadcasting ----

type binaryOp struct {
	n  NodeInfo
	fn func(x, y float32) float32
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
	out, err := binaryF32(ctx, a, b, o.fn)
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	return []*tensor.Tensor{out}, nil
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

type unaryOp struct {
	n  NodeInfo
	fn func(x float32) float32
}

func (o *unaryOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 1 || in[0] == nil {
		return nil, o.n.Errorf("missing input")
	}
	x := in[0]
	if x.DType() != tensor.F32 {
		return nil, o.n.Errorf("unsupported dtype %s", x.DType())
	}
	out := ctx.New(tensor.F32, x.Shape()...)
	xf, of := x.F32(), out.F32()
	fn := o.fn
	for i := range of {
		of[i] = fn(xf[i])
	}
	return []*tensor.Tensor{out}, nil
}

func sigmoid(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }

func hardSigmoid(alpha, beta float32) func(float32) float32 {
	return func(x float32) float32 { return min(1, max(0, alpha*x+beta)) }
}

func gelu(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
}

func geluTanh(x float32) float32 {
	const c = 0.7978845608028654 // sqrt(2/pi)
	return 0.5 * x * (1 + float32(math.Tanh(c*float64(x+0.044715*x*x*x))))
}

func init() {
	bin := func(name string, fn func(x, y float32) float32) {
		Register("", name, 7, func(n NodeInfo) (Op, error) { return &binaryOp{n, fn}, nil })
	}
	bin("Add", func(x, y float32) float32 { return x + y })
	bin("Sub", func(x, y float32) float32 { return x - y })
	bin("Mul", func(x, y float32) float32 { return x * y })
	bin("Div", func(x, y float32) float32 { return x / y })
	bin("Pow", func(x, y float32) float32 {
		switch y {
		case 2:
			return x * x
		case 0.5:
			return float32(math.Sqrt(float64(x)))
		}
		return float32(math.Pow(float64(x), float64(y)))
	})
	bin("Max", func(x, y float32) float32 { return max(x, y) })
	bin("Min", func(x, y float32) float32 { return min(x, y) })

	un := func(name string, since int, fn func(x float32) float32) {
		Register("", name, since, func(n NodeInfo) (Op, error) { return &unaryOp{n, fn}, nil })
	}
	un("Relu", 6, func(x float32) float32 { return max(0, x) })
	un("Sigmoid", 6, sigmoid)
	un("Tanh", 6, func(x float32) float32 { return float32(math.Tanh(float64(x))) })
	un("Exp", 6, func(x float32) float32 { return float32(math.Exp(float64(x))) })
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
	un("HardSwish", 14, func(x float32) float32 { return x * min(1, max(0, x/6+0.5)) })
	un("Mish", 18, func(x float32) float32 {
		return x * float32(math.Tanh(math.Log1p(math.Exp(float64(x)))))
	})
	un("Identity", 1, func(x float32) float32 { return x }) // replaced below for non-f32

	Register("", "HardSigmoid", 6, func(n NodeInfo) (Op, error) {
		return &unaryOp{n, hardSigmoid(n.Attrs.Float("alpha", 0.2), n.Attrs.Float("beta", 0.5))}, nil
	})
	Register("", "LeakyRelu", 6, func(n NodeInfo) (Op, error) {
		a := n.Attrs.Float("alpha", 0.01)
		return &unaryOp{n, func(x float32) float32 {
			if x < 0 {
				return a * x
			}
			return x
		}}, nil
	})
	Register("", "Elu", 6, func(n NodeInfo) (Op, error) {
		a := n.Attrs.Float("alpha", 1)
		return &unaryOp{n, func(x float32) float32 {
			if x < 0 {
				return a * (float32(math.Exp(float64(x))) - 1)
			}
			return x
		}}, nil
	})
	Register("", "Gelu", 20, func(n NodeInfo) (Op, error) {
		if n.Attrs.String("approximate", "none") == "tanh" {
			return &unaryOp{n, geluTanh}, nil
		}
		return &unaryOp{n, gelu}, nil
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
	out := ctx.New(tensor.F32, x.Shape()...)
	xf, of := x.F32(), out.F32()
	for i := range of {
		of[i] = min(hi, max(lo, xf[i]))
	}
	return []*tensor.Tensor{out}, nil
}
