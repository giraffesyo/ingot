package graph

import (
	"github.com/giraffesyo/ingot/tensor"
)

// Generic layer helpers over Builder, named after their PyTorch modules so
// a model file reads like the reference implementation. They emit standard
// ONNX nodes (or fused ingot ones where the runtime has them); nothing here
// is model-specific.

// Linear is torch.nn.Linear over a 2-D x [M, in]: x·wᵀ + bias with w stored
// [out, in] as PyTorch keeps it. It lowers to Gemm(transB=1), which packs
// w once on first use — no transposed copy is ever built.
func (b *Builder) Linear(x *Value, w, bias *tensor.Tensor) *Value {
	in := []*Value{x, b.Const("weight", w)}
	if bias != nil {
		in = append(in, b.Const("bias", bias))
	}
	return b.Op("Gemm", Attr("transB", 1), in...)
}

// LayerNorm normalises over the last axis. A nil scale is all-ones (PyTorch
// elementwise_affine=False).
func (b *Builder) LayerNorm(x *Value, dim int, scale, bias *tensor.Tensor, eps float32) *Value {
	if scale == nil {
		scale = Ones(dim)
	}
	in := []*Value{x, b.Const("scale", scale)}
	if bias != nil {
		in = append(in, b.Const("bias", bias))
	}
	return b.Op("LayerNormalization", Attr("axis", -1, "epsilon", eps), in...)
}

// RMSNorm is x · rsqrt(mean(x², last axis) + eps) · weight, the
// Llama/Qwen form; nil weight skips the scale.
func (b *Builder) RMSNorm(x *Value, weight *tensor.Tensor, eps float32) *Value {
	ms := b.Op("ReduceMean", Attr("keepdims", 1), b.Mul(x, x), b.Ints(-1))
	inv := b.Op("Reciprocal", nil, b.Op("Sqrt", nil, b.Add(ms, b.Scalar(eps))))
	y := b.Mul(x, inv)
	if weight != nil {
		y = b.Mul(y, b.Const("weight", weight))
	}
	return y
}

// Elementwise and shape shorthands.

func (b *Builder) Add(x, y *Value) *Value { return b.Op("Add", nil, x, y) }
func (b *Builder) Sub(x, y *Value) *Value { return b.Op("Sub", nil, x, y) }
func (b *Builder) Mul(x, y *Value) *Value { return b.Op("Mul", nil, x, y) }
func (b *Builder) Div(x, y *Value) *Value { return b.Op("Div", nil, x, y) }

// SiLU is x·sigmoid(x) as one fused op.
func (b *Builder) SiLU(x *Value) *Value { return b.Op("ingot.SiLU", nil, x) }

// GeluTanh is GELU with the tanh approximation (PyTorch approximate="tanh").
func (b *Builder) GeluTanh(x *Value) *Value { return b.Op("Gelu", Attr("approximate", "tanh"), x) }

func (b *Builder) Tanh(x *Value) *Value { return b.Op("Tanh", nil, x) }

// Reshape to shape (int64 semantics: 0 copies the input dim, -1 infers).
func (b *Builder) Reshape(x *Value, shape ...int64) *Value {
	return b.Op("Reshape", nil, x, b.Ints(shape...))
}

func (b *Builder) Transpose(x *Value, perm ...int) *Value {
	return b.Op("Transpose", Attr("perm", perm), x)
}

func (b *Builder) Concat(axis int, xs ...*Value) *Value {
	return b.Op("Concat", Attr("axis", axis), xs...)
}

// Slice takes [start, end) along one axis.
func (b *Builder) Slice(x *Value, axis int, start, end int64) *Value {
	return b.Op("Slice", nil, x, b.Ints(start), b.Ints(end), b.Ints(int64(axis)))
}

// Ones returns an all-ones f32 vector.
func Ones(n int) *tensor.Tensor {
	t := tensor.New(tensor.F32, n)
	for i := range t.F32() {
		t.F32()[i] = 1
	}
	return t
}
