package tensor

import (
	"fmt"
	"strings"
)

// Shape is a list of dimension sizes. Rank 0 is a scalar.
type Shape []int

// Numel returns the number of elements (1 for a scalar, 0 if any dim is 0).
func (s Shape) Numel() int {
	n := 1
	for _, d := range s {
		n *= d
	}
	return n
}

// Rank is the number of dimensions.
func (s Shape) Rank() int { return len(s) }

// Equal reports whether two shapes are identical.
func (s Shape) Equal(o Shape) bool {
	if len(s) != len(o) {
		return false
	}
	for i := range s {
		if s[i] != o[i] {
			return false
		}
	}
	return true
}

// Clone returns a copy.
func (s Shape) Clone() Shape { return append(Shape(nil), s...) }

// Strides returns contiguous row-major strides in elements.
func (s Shape) Strides() []int {
	st := make([]int, len(s))
	acc := 1
	for i := len(s) - 1; i >= 0; i-- {
		st[i] = acc
		acc *= s[i]
	}
	return st
}

func (s Shape) String() string {
	var b strings.Builder
	b.WriteByte('[')
	for i, d := range s {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%d", d)
	}
	b.WriteByte(']')
	return b.String()
}
