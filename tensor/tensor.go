package tensor

import (
	"fmt"
	"unsafe"
)

// Tensor is a dense n-d array. The zero value is not usable; use New/From*.
//
// Storage is a byte-addressable buffer; typed accessors reinterpret it. A Tensor
// may be a view (non-owning) of another tensor's storage. Strides are in elements.
type Tensor struct {
	dtype   DType
	shape   Shape
	strides []int
	buf     []byte // len == cap == numel*dtype.Size() for owning tensors
	offset  int    // element offset into buf for views
	pool    *Pool  // non-nil if buf came from a pool
}

// New allocates a zeroed tensor of the given dtype and shape.
func New(dt DType, shape ...int) *Tensor {
	s := Shape(shape)
	return &Tensor{
		dtype:   dt,
		shape:   s.Clone(),
		strides: s.Strides(),
		buf:     make([]byte, s.Numel()*dt.Size()),
	}
}

// FromF32 wraps an existing []float32 (no copy) with the given shape.
func FromF32(data []float32, shape ...int) *Tensor {
	s := Shape(shape)
	if len(data) != s.Numel() {
		panic(fmt.Sprintf("tensor: data len %d != numel %d for shape %v", len(data), s.Numel(), s))
	}
	t := &Tensor{dtype: F32, shape: s.Clone(), strides: s.Strides()}
	if len(data) > 0 {
		t.buf = unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
	}
	return t
}

// DType returns the element type.
func (t *Tensor) DType() DType { return t.dtype }

// Shape returns the shape (do not mutate).
func (t *Tensor) Shape() Shape { return t.shape }

// Strides returns element strides (do not mutate).
func (t *Tensor) Strides() []int { return t.strides }

// Numel returns the element count.
func (t *Tensor) Numel() int { return t.shape.Numel() }

// Dim returns the size of dimension i (negative indexes from the end).
func (t *Tensor) Dim(i int) int {
	if i < 0 {
		i += len(t.shape)
	}
	return t.shape[i]
}

// IsContiguous reports whether the tensor is dense row-major.
func (t *Tensor) IsContiguous() bool {
	exp := 1
	for i := len(t.shape) - 1; i >= 0; i-- {
		if t.shape[i] != 1 && t.strides[i] != exp {
			return false
		}
		exp *= t.shape[i]
	}
	return true
}

// F32 returns the underlying float32 storage (contiguous tensors only).
// Panics if dtype != F32.
func (t *Tensor) F32() []float32 {
	t.mustDType(F32)
	t.mustContiguous()
	n := t.Numel()
	if n == 0 {
		return nil
	}
	p := unsafe.Pointer(&t.buf[t.offset*4])
	return unsafe.Slice((*float32)(p), n)
}

// I64 returns the underlying int64 storage (contiguous tensors only).
func (t *Tensor) I64() []int64 {
	t.mustDType(I64)
	t.mustContiguous()
	n := t.Numel()
	if n == 0 {
		return nil
	}
	p := unsafe.Pointer(&t.buf[t.offset*8])
	return unsafe.Slice((*int64)(p), n)
}

// Bytes returns the raw storage of a contiguous tensor.
func (t *Tensor) Bytes() []byte {
	t.mustContiguous()
	sz := t.dtype.Size()
	return t.buf[t.offset*sz : (t.offset+t.Numel())*sz]
}

// Reshape returns a view with a new shape (same numel, contiguous only).
func (t *Tensor) Reshape(shape ...int) *Tensor {
	s := Shape(shape)
	if s.Numel() != t.Numel() {
		panic(fmt.Sprintf("tensor: cannot reshape %v to %v", t.shape, s))
	}
	t.mustContiguous()
	return &Tensor{dtype: t.dtype, shape: s.Clone(), strides: s.Strides(), buf: t.buf, offset: t.offset}
}

// SharesBuffer reports whether t and u are views of the same storage.
func (t *Tensor) SharesBuffer(u *Tensor) bool {
	return u != nil && t != nil && len(t.buf) > 0 && len(u.buf) > 0 && &t.buf[0] == &u.buf[0]
}

// Clone returns a deep copy (always contiguous).
func (t *Tensor) Clone() *Tensor {
	out := New(t.dtype, t.shape...)
	if t.IsContiguous() {
		copy(out.buf, t.Bytes())
		return out
	}
	panic("tensor: Clone of non-contiguous tensor not yet implemented")
}

func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(%s%v)", t.dtype, t.shape)
}

func (t *Tensor) mustDType(d DType) {
	if t.dtype != d {
		panic(fmt.Sprintf("tensor: want %s, have %s", d, t.dtype))
	}
}

func (t *Tensor) mustContiguous() {
	if !t.IsContiguous() {
		panic("tensor: operation requires contiguous tensor")
	}
}

// I32 returns the underlying int32 storage (contiguous tensors only).
func (t *Tensor) I32() []int32 {
	t.mustDType(I32)
	t.mustContiguous()
	n := t.Numel()
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*int32)(unsafe.Pointer(&t.buf[t.offset*4])), n)
}

// I16 returns the tensor data as []int16 (dtype I16).
func (t *Tensor) I16() []int16 {
	t.mustDType(I16)
	t.mustContiguous()
	n := t.Numel()
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*int16)(unsafe.Pointer(&t.buf[t.offset*2])), n)
}

// Bool returns the underlying bool storage (contiguous tensors only).
func (t *Tensor) Bool() []bool {
	t.mustDType(Bool)
	t.mustContiguous()
	n := t.Numel()
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*bool)(unsafe.Pointer(&t.buf[t.offset])), n)
}

// U8 returns the underlying uint8 storage (contiguous tensors only).
func (t *Tensor) U8() []uint8 {
	t.mustDType(U8)
	t.mustContiguous()
	return t.buf[t.offset : t.offset+t.Numel()]
}

// I8 returns the underlying int8 storage (contiguous tensors only).
func (t *Tensor) I8() []int8 {
	t.mustDType(I8)
	t.mustContiguous()
	n := t.Numel()
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*int8)(unsafe.Pointer(&t.buf[t.offset])), n)
}

// FromI64 wraps an existing []int64 (no copy) with the given shape.
func FromI64(data []int64, shape ...int) *Tensor {
	s := Shape(shape)
	if len(data) != s.Numel() {
		panic(fmt.Sprintf("tensor: data len %d != numel %d for shape %v", len(data), s.Numel(), s))
	}
	t := &Tensor{dtype: I64, shape: s.Clone(), strides: s.Strides()}
	if len(data) > 0 {
		t.buf = unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*8)
	}
	return t
}

// Scalar returns a rank-0 f32 tensor.
func Scalar(v float32) *Tensor {
	t := New(F32)
	t.F32()[0] = v
	return t
}
