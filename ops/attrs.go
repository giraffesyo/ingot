package ops

import "github.com/giraffesyo/ocr/tensor"

// AttrKind enumerates attribute value kinds (mirrors ONNX AttributeType).
type AttrKind uint8

const (
	KindUndefined AttrKind = iota
	KindFloat
	KindInt
	KindString
	KindTensor
	KindFloats
	KindInts
	KindStrings
)

// Attr is a decoded node attribute.
type Attr struct {
	Kind    AttrKind
	F       float32
	I       int64
	S       string
	T       *tensor.Tensor
	Floats  []float32
	Ints    []int64
	Strings []string
}

// Attrs maps attribute name to value.
type Attrs map[string]Attr

// Int returns the int attribute or def.
func (a Attrs) Int(name string, def int64) int64 {
	if v, ok := a[name]; ok && v.Kind == KindInt {
		return v.I
	}
	return def
}

// Float returns the float attribute or def.
func (a Attrs) Float(name string, def float32) float32 {
	if v, ok := a[name]; ok && v.Kind == KindFloat {
		return v.F
	}
	return def
}

// String returns the string attribute or def.
func (a Attrs) String(name string, def string) string {
	if v, ok := a[name]; ok && v.Kind == KindString {
		return v.S
	}
	return def
}

// Ints returns the ints attribute or def.
func (a Attrs) Ints(name string, def []int64) []int64 {
	if v, ok := a[name]; ok && v.Kind == KindInts {
		return v.Ints
	}
	return def
}

// Floats returns the floats attribute or def.
func (a Attrs) Floats(name string, def []float32) []float32 {
	if v, ok := a[name]; ok && v.Kind == KindFloats {
		return v.Floats
	}
	return def
}

// Tensor returns the tensor attribute or nil.
func (a Attrs) Tensor(name string) *tensor.Tensor {
	if v, ok := a[name]; ok && v.Kind == KindTensor {
		return v.T
	}
	return nil
}

// Has reports whether the attribute is present.
func (a Attrs) Has(name string) bool { _, ok := a[name]; return ok }

func intsToShape(v []int64) tensor.Shape {
	s := make(tensor.Shape, len(v))
	for i, d := range v {
		s[i] = int(d)
	}
	return s
}
