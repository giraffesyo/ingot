package onnx

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// wire types
const (
	wtVarint = 0
	wtI64    = 1
	wtLen    = 2
	wtI32    = 5
)

var errTruncated = errors.New("onnx: truncated message")

// reader is a cursor over a protobuf message body.
type reader struct {
	buf []byte
	pos int
}

func (r *reader) eof() bool { return r.pos >= len(r.buf) }

func (r *reader) varint() (uint64, error) {
	var x uint64
	var s uint
	for i := 0; i < 10; i++ {
		if r.pos >= len(r.buf) {
			return 0, errTruncated
		}
		b := r.buf[r.pos]
		r.pos++
		if b < 0x80 {
			if i == 9 && b > 1 {
				return 0, errors.New("onnx: varint overflow")
			}
			return x | uint64(b)<<s, nil
		}
		x |= uint64(b&0x7f) << s
		s += 7
	}
	return 0, errors.New("onnx: varint too long")
}

// tag reads a field tag, returning field number and wire type.
func (r *reader) tag() (int, int, error) {
	v, err := r.varint()
	if err != nil {
		return 0, 0, err
	}
	return int(v >> 3), int(v & 7), nil
}

func (r *reader) bytes() ([]byte, error) {
	n, err := r.varint()
	if err != nil {
		return nil, err
	}
	if n > uint64(len(r.buf)-r.pos) {
		return nil, errTruncated
	}
	b := r.buf[r.pos : r.pos+int(n)]
	r.pos += int(n)
	return b, nil
}

func (r *reader) fixed32() (uint32, error) {
	if r.pos+4 > len(r.buf) {
		return 0, errTruncated
	}
	v := binary.LittleEndian.Uint32(r.buf[r.pos:])
	r.pos += 4
	return v, nil
}

func (r *reader) fixed64() (uint64, error) {
	if r.pos+8 > len(r.buf) {
		return 0, errTruncated
	}
	v := binary.LittleEndian.Uint64(r.buf[r.pos:])
	r.pos += 8
	return v, nil
}

// skip skips a field of the given wire type.
func (r *reader) skip(wt int) error {
	switch wt {
	case wtVarint:
		_, err := r.varint()
		return err
	case wtI64:
		_, err := r.fixed64()
		return err
	case wtLen:
		_, err := r.bytes()
		return err
	case wtI32:
		_, err := r.fixed32()
		return err
	}
	return fmt.Errorf("onnx: unknown wire type %d", wt)
}

// Scalar readers with wire-type checks.

func (r *reader) int64Field(wt int) (int64, error) {
	if wt != wtVarint {
		return 0, fmt.Errorf("onnx: expected varint, got wire type %d", wt)
	}
	v, err := r.varint()
	return int64(v), err
}

func (r *reader) floatField(wt int) (float32, error) {
	if wt != wtI32 {
		return 0, fmt.Errorf("onnx: expected fixed32, got wire type %d", wt)
	}
	v, err := r.fixed32()
	return math.Float32frombits(v), err
}

func (r *reader) doubleField(wt int) (float64, error) {
	if wt != wtI64 {
		return 0, fmt.Errorf("onnx: expected fixed64, got wire type %d", wt)
	}
	v, err := r.fixed64()
	return math.Float64frombits(v), err
}

func (r *reader) bytesField(wt int) ([]byte, error) {
	if wt != wtLen {
		return nil, fmt.Errorf("onnx: expected length-delimited, got wire type %d", wt)
	}
	return r.bytes()
}

func (r *reader) stringField(wt int) (string, error) {
	b, err := r.bytesField(wt)
	return string(b), err
}

// Repeated numeric fields may be packed (length-delimited) or not.

func (r *reader) repeatedInt64(wt int, dst []int64) ([]int64, error) {
	if wt == wtLen {
		b, err := r.bytes()
		if err != nil {
			return dst, err
		}
		sub := reader{buf: b}
		for !sub.eof() {
			v, err := sub.varint()
			if err != nil {
				return dst, err
			}
			dst = append(dst, int64(v))
		}
		return dst, nil
	}
	v, err := r.int64Field(wt)
	return append(dst, v), err
}

func (r *reader) repeatedInt32(wt int, dst []int32) ([]int32, error) {
	if wt == wtLen {
		b, err := r.bytes()
		if err != nil {
			return dst, err
		}
		sub := reader{buf: b}
		for !sub.eof() {
			v, err := sub.varint()
			if err != nil {
				return dst, err
			}
			dst = append(dst, int32(v))
		}
		return dst, nil
	}
	v, err := r.int64Field(wt)
	return append(dst, int32(v)), err
}

func (r *reader) repeatedUint64(wt int, dst []uint64) ([]uint64, error) {
	if wt == wtLen {
		b, err := r.bytes()
		if err != nil {
			return dst, err
		}
		sub := reader{buf: b}
		for !sub.eof() {
			v, err := sub.varint()
			if err != nil {
				return dst, err
			}
			dst = append(dst, v)
		}
		return dst, nil
	}
	v, err := r.varint()
	return append(dst, v), err
}

func (r *reader) repeatedFloat(wt int, dst []float32) ([]float32, error) {
	if wt == wtLen {
		b, err := r.bytes()
		if err != nil {
			return dst, err
		}
		if len(b)%4 != 0 {
			return dst, errors.New("onnx: packed float field length not multiple of 4")
		}
		for i := 0; i < len(b); i += 4 {
			dst = append(dst, math.Float32frombits(binary.LittleEndian.Uint32(b[i:])))
		}
		return dst, nil
	}
	v, err := r.floatField(wt)
	return append(dst, v), err
}

func (r *reader) repeatedDouble(wt int, dst []float64) ([]float64, error) {
	if wt == wtLen {
		b, err := r.bytes()
		if err != nil {
			return dst, err
		}
		if len(b)%8 != 0 {
			return dst, errors.New("onnx: packed double field length not multiple of 8")
		}
		for i := 0; i < len(b); i += 8 {
			dst = append(dst, math.Float64frombits(binary.LittleEndian.Uint64(b[i:])))
		}
		return dst, nil
	}
	v, err := r.doubleField(wt)
	return append(dst, v), err
}
