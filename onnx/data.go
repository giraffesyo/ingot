package onnx

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Numel returns the element count implied by Dims.
func (t *Tensor) Numel() int {
	n := 1
	for _, d := range t.Dims {
		n *= int(d)
	}
	return n
}

// Float32s returns the tensor's data as float32, converting from the stored
// representation (raw_data little-endian, float_data, or int32_data for
// float16/bfloat16 per the ONNX spec). Only Float, Float16, BFloat16 and
// Double are supported.
func (t *Tensor) Float32s() ([]float32, error) {
	n := t.Numel()
	out := make([]float32, n)
	switch t.DataType {
	case Float:
		if t.RawData != nil {
			if len(t.RawData) != n*4 {
				return nil, fmt.Errorf("onnx: tensor %q raw_data len %d != %d", t.Name, len(t.RawData), n*4)
			}
			for i := range out {
				out[i] = math.Float32frombits(binary.LittleEndian.Uint32(t.RawData[i*4:]))
			}
			return out, nil
		}
		if len(t.FloatData) != n {
			return nil, fmt.Errorf("onnx: tensor %q float_data len %d != %d", t.Name, len(t.FloatData), n)
		}
		copy(out, t.FloatData)
		return out, nil
	case Double:
		if t.RawData != nil {
			if len(t.RawData) != n*8 {
				return nil, fmt.Errorf("onnx: tensor %q raw_data len %d != %d", t.Name, len(t.RawData), n*8)
			}
			for i := range out {
				out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(t.RawData[i*8:])))
			}
			return out, nil
		}
		if len(t.DoubleData) != n {
			return nil, fmt.Errorf("onnx: tensor %q double_data len %d != %d", t.Name, len(t.DoubleData), n)
		}
		for i, v := range t.DoubleData {
			out[i] = float32(v)
		}
		return out, nil
	case Float16, BFloat16:
		conv := floatMToFloat32
		if t.DataType == BFloat16 {
			conv = bfloatMToFloat32
		}
		if t.RawData != nil {
			if len(t.RawData) != n*2 {
				return nil, fmt.Errorf("onnx: tensor %q raw_data len %d != %d", t.Name, len(t.RawData), n*2)
			}
			for i := range out {
				out[i] = conv(binary.LittleEndian.Uint16(t.RawData[i*2:]))
			}
			return out, nil
		}
		if len(t.Int32Data) != n {
			return nil, fmt.Errorf("onnx: tensor %q int32_data len %d != %d", t.Name, len(t.Int32Data), n)
		}
		for i, v := range t.Int32Data {
			out[i] = conv(uint16(v))
		}
		return out, nil
	}
	return nil, fmt.Errorf("onnx: tensor %q: cannot convert %s to float32", t.Name, t.DataType)
}

// Int64s returns integer tensor data as int64 (Int64, Int32, Int8, Uint8, Bool, Int16, Uint16, Uint32, Uint64).
func (t *Tensor) Int64s() ([]int64, error) {
	n := t.Numel()
	out := make([]int64, n)
	raw := t.RawData
	switch t.DataType {
	case Int64:
		if raw != nil {
			if len(raw) != n*8 {
				return nil, fmt.Errorf("onnx: tensor %q raw_data len %d != %d", t.Name, len(raw), n*8)
			}
			for i := range out {
				out[i] = int64(binary.LittleEndian.Uint64(raw[i*8:]))
			}
			return out, nil
		}
		if len(t.Int64Data) != n {
			return nil, fmt.Errorf("onnx: tensor %q int64_data len %d != %d", t.Name, len(t.Int64Data), n)
		}
		copy(out, t.Int64Data)
		return out, nil
	case Int32:
		if raw != nil {
			if len(raw) != n*4 {
				return nil, fmt.Errorf("onnx: tensor %q raw_data len %d != %d", t.Name, len(raw), n*4)
			}
			for i := range out {
				out[i] = int64(int32(binary.LittleEndian.Uint32(raw[i*4:])))
			}
			return out, nil
		}
		for i, v := range t.Int32Data {
			out[i] = int64(v)
		}
		return out, nil
	case Uint32:
		if raw != nil {
			for i := range out {
				out[i] = int64(binary.LittleEndian.Uint32(raw[i*4:]))
			}
			return out, nil
		}
		for i, v := range t.Uint64Data {
			out[i] = int64(v)
		}
		return out, nil
	case Uint64:
		if raw != nil {
			for i := range out {
				out[i] = int64(binary.LittleEndian.Uint64(raw[i*8:]))
			}
			return out, nil
		}
		for i, v := range t.Uint64Data {
			out[i] = int64(v)
		}
		return out, nil
	case Int16:
		if raw != nil {
			for i := range out {
				out[i] = int64(int16(binary.LittleEndian.Uint16(raw[i*2:])))
			}
			return out, nil
		}
		for i, v := range t.Int32Data {
			out[i] = int64(int16(v))
		}
		return out, nil
	case Uint16:
		if raw != nil {
			for i := range out {
				out[i] = int64(binary.LittleEndian.Uint16(raw[i*2:]))
			}
			return out, nil
		}
		for i, v := range t.Int32Data {
			out[i] = int64(uint16(v))
		}
		return out, nil
	case Int8:
		if raw != nil {
			for i := range out {
				out[i] = int64(int8(raw[i]))
			}
			return out, nil
		}
		for i, v := range t.Int32Data {
			out[i] = int64(int8(v))
		}
		return out, nil
	case Uint8, Bool:
		if raw != nil {
			for i := range out {
				out[i] = int64(raw[i])
			}
			return out, nil
		}
		for i, v := range t.Int32Data {
			out[i] = int64(uint8(v))
		}
		return out, nil
	}
	return nil, fmt.Errorf("onnx: tensor %q: cannot convert %s to int64", t.Name, t.DataType)
}

func floatMToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// subnormal: normalise
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | e<<23 | mant<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
}

func bfloatMToFloat32(h uint16) float32 {
	return math.Float32frombits(uint32(h) << 16)
}
