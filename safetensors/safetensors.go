// Package safetensors reads Hugging Face .safetensors weight files: an
// 8-byte little-endian header length, a JSON header mapping tensor names to
// dtype, shape and byte range, then the raw little-endian data.
//
// Files are memory-mapped where the platform allows (read-only, shared with
// the page cache), so opening a multi-GB checkpoint is instant and a tensor
// is only paged in when read. Tensor returns zero-copy views in the stored
// dtype; F32 converts (bf16/f16/f64 widen or narrow into a new f32 buffer,
// f32 is a view when aligned).
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/tensor"
)

// Info describes one stored tensor.
type Info struct {
	DType string // safetensors dtype name: F32, BF16, F16, F64, I64, I32, ...
	Shape []int
	Begin int // byte range within the data section
	End   int
}

// File is an open .safetensors file.
type File struct {
	path     string
	raw      []byte // the whole file (mapped or read)
	data     []byte // raw[8+headerLen:]
	unmap    func() error
	tensors  map[string]Info
	Metadata map[string]string
}

// maxHeader bounds the JSON header (the format caps it at 100 MB).
const maxHeader = 100 << 20

// Open maps (or reads) path and parses its header. The File must stay open
// while any tensor returned by Tensor, or by F32 as a view, is in use.
func Open(path string) (*File, error) {
	raw, unmap, err := mapFile(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: %w", err)
	}
	f, err := parse(raw)
	if err != nil {
		if unmap != nil {
			unmap()
		}
		return nil, fmt.Errorf("safetensors: %s: %w", path, err)
	}
	f.path, f.unmap = path, unmap
	return f, nil
}

func parse(raw []byte) (*File, error) {
	if len(raw) < 8 {
		return nil, fmt.Errorf("file too short (%d bytes)", len(raw))
	}
	n := binary.LittleEndian.Uint64(raw)
	if n > maxHeader || n > uint64(len(raw)-8) {
		return nil, fmt.Errorf("header length %d exceeds file size %d", n, len(raw))
	}
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+n], &hdr); err != nil {
		return nil, fmt.Errorf("header: %w", err)
	}
	f := &File{raw: raw, data: raw[8+n:], tensors: make(map[string]Info, len(hdr))}
	for name, msg := range hdr {
		if name == "__metadata__" {
			if err := json.Unmarshal(msg, &f.Metadata); err != nil {
				return nil, fmt.Errorf("__metadata__: %w", err)
			}
			continue
		}
		var e struct {
			DType   string `json:"dtype"`
			Shape   []int  `json:"shape"`
			Offsets [2]int `json:"data_offsets"`
		}
		if err := json.Unmarshal(msg, &e); err != nil {
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}
		size, ok := dtypeSize(e.DType)
		if !ok {
			return nil, fmt.Errorf("tensor %q: unknown dtype %q", name, e.DType)
		}
		numel := 1
		for _, d := range e.Shape {
			if d < 0 {
				return nil, fmt.Errorf("tensor %q: negative dim in %v", name, e.Shape)
			}
			numel *= d
		}
		b, en := e.Offsets[0], e.Offsets[1]
		if b < 0 || en < b || en > len(f.data) {
			return nil, fmt.Errorf("tensor %q: data_offsets [%d,%d] outside data section of %d bytes", name, b, en, len(f.data))
		}
		if en-b != numel*size {
			return nil, fmt.Errorf("tensor %q: %d bytes for %s%v, want %d", name, en-b, e.DType, e.Shape, numel*size)
		}
		f.tensors[name] = Info{DType: e.DType, Shape: e.Shape, Begin: b, End: en}
	}
	return f, nil
}

// Close releases the mapping. Tensors viewing the file are invalid after it.
func (f *File) Close() error {
	if f.unmap == nil {
		return nil
	}
	err := f.unmap()
	f.unmap, f.raw, f.data = nil, nil, nil
	return err
}

// Names lists the stored tensors, sorted.
func (f *File) Names() []string {
	out := make([]string, 0, len(f.tensors))
	for n := range f.tensors {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// Info returns the header entry for name.
func (f *File) Info(name string) (Info, bool) {
	i, ok := f.tensors[name]
	return i, ok
}

// Tensor returns a zero-copy view of name in its stored dtype. Unaligned
// storage (possible when the header length is not a multiple of the element
// size) is copied instead.
func (f *File) Tensor(name string) (*tensor.Tensor, error) {
	info, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: %s: no tensor %q", f.path, name)
	}
	dt, ok := tensorDType(info.DType)
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q: dtype %s has no runtime equivalent", name, info.DType)
	}
	b := f.data[info.Begin:info.End]
	if !tensor.Aligned(dt, b) {
		b = append([]byte(nil), b...)
	}
	return tensor.FromBytes(dt, b, info.Shape...), nil
}

// F32 returns name as float32: a view for aligned F32 storage, otherwise a
// new buffer converted from BF16, F16 or F64.
func (f *File) F32(name string) (*tensor.Tensor, error) {
	info, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: %s: no tensor %q", f.path, name)
	}
	if info.DType == "F32" {
		return f.Tensor(name)
	}
	src := f.data[info.Begin:info.End]
	out := tensor.New(tensor.F32, info.Shape...)
	dst := out.F32()
	switch info.DType {
	case "BF16":
		convert(dst, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(src[2*i:])) << 16)
			}
		})
	case "F16":
		convert(dst, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				dst[i] = HalfToF32(binary.LittleEndian.Uint16(src[2*i:]))
			}
		})
	case "F64":
		convert(dst, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				dst[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(src[8*i:])))
			}
		})
	default:
		return nil, fmt.Errorf("safetensors: tensor %q: cannot convert %s to f32", name, info.DType)
	}
	return out, nil
}

// convChunk is the per-task element count for parallel dtype conversion.
const convChunk = 1 << 18

func convert(dst []float32, fn func(lo, hi int)) {
	n := len(dst)
	par.For((n+convChunk-1)/convChunk, 1, func(c, _ int) {
		fn(c*convChunk, min((c+1)*convChunk, n))
	})
}

// HalfToF32 widens an IEEE 754 binary16 value.
func HalfToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	switch {
	case exp == 0x1f: // inf / nan
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	case exp != 0: // normal
		return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
	case mant == 0: // ±0
		return math.Float32frombits(sign)
	}
	// subnormal: renormalise
	e := uint32(113)
	for mant&0x400 == 0 {
		mant <<= 1
		e--
	}
	return math.Float32frombits(sign | e<<23 | (mant&0x3ff)<<13)
}

func dtypeSize(s string) (int, bool) {
	switch s {
	case "F64", "I64", "U64":
		return 8, true
	case "F32", "I32", "U32":
		return 4, true
	case "F16", "BF16", "I16", "U16":
		return 2, true
	case "I8", "U8", "BOOL", "F8_E4M3", "F8_E5M2":
		return 1, true
	}
	return 0, false
}

func tensorDType(s string) (tensor.DType, bool) {
	switch s {
	case "F32":
		return tensor.F32, true
	case "F16":
		return tensor.F16, true
	case "BF16":
		return tensor.BF16, true
	case "I64":
		return tensor.I64, true
	case "I32":
		return tensor.I32, true
	case "I16":
		return tensor.I16, true
	case "I8":
		return tensor.I8, true
	case "U8":
		return tensor.U8, true
	case "BOOL":
		return tensor.Bool, true
	}
	return tensor.Invalid, false
}

// readFile is the portable fallback for mapFile.
func readFile(path string) ([]byte, func() error, error) {
	b, err := os.ReadFile(path)
	return b, nil, err
}
