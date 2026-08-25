// Package tensor provides the core n-dimensional array type used by the runtime.
//
// Tensors are dense, row-major (C order) by default. Data is stored as a typed
// slice; DType records the element type. Views share storage.
package tensor

import "fmt"

// DType is an element type.
type DType uint8

const (
	Invalid DType = iota
	F32
	F16
	BF16
	I8
	U8
	I16
	I32
	I64
	Bool
)

var dtypeNames = [...]string{
	Invalid: "invalid", F32: "f32", F16: "f16", BF16: "bf16",
	I8: "i8", U8: "u8", I16: "i16", I32: "i32", I64: "i64", Bool: "bool",
}

func (d DType) String() string {
	if int(d) < len(dtypeNames) {
		return dtypeNames[d]
	}
	return fmt.Sprintf("dtype(%d)", d)
}

// Size returns the size in bytes of one element.
func (d DType) Size() int {
	switch d {
	case F32, I32:
		return 4
	case F16, BF16, I16:
		return 2
	case I8, U8, Bool:
		return 1
	case I64:
		return 8
	}
	panic("tensor: invalid dtype")
}
