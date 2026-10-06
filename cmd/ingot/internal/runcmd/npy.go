package runcmd

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/giraffesyo/ingot/tensor"
)

// NumPy .npy (format 1.0/2.0/3.0), little-endian, C order. Reading maps
// f4 → f32, f8 → f32 (converted), i8 → i64, i4 → i32, u1 → u8, b1 → bool.

var npyDescr = regexp.MustCompile(`'descr':\s*'([^']*)'`)
var npyFortran = regexp.MustCompile(`'fortran_order':\s*(True|False)`)
var npyShape = regexp.MustCompile(`'shape':\s*\(([^)]*)\)`)

func readNPY(path string) (*tensor.Tensor, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 10 || string(raw[:6]) != "\x93NUMPY" {
		return nil, fmt.Errorf("%s: not a .npy file", path)
	}
	var hlen, off int
	switch raw[6] {
	case 1:
		hlen, off = int(binary.LittleEndian.Uint16(raw[8:])), 10
	case 2, 3:
		if len(raw) < 12 {
			return nil, fmt.Errorf("%s: short header", path)
		}
		hlen, off = int(binary.LittleEndian.Uint32(raw[8:])), 12
	default:
		return nil, fmt.Errorf("%s: .npy version %d", path, raw[6])
	}
	if off+hlen > len(raw) {
		return nil, fmt.Errorf("%s: short header", path)
	}
	header, data := string(raw[off:off+hlen]), raw[off+hlen:]
	d, f, sh := npyDescr.FindStringSubmatch(header), npyFortran.FindStringSubmatch(header), npyShape.FindStringSubmatch(header)
	if d == nil || f == nil || sh == nil {
		return nil, fmt.Errorf("%s: unparsed header %q", path, header)
	}
	if f[1] == "True" {
		return nil, fmt.Errorf("%s: Fortran order not supported", path)
	}
	var shape []int
	for _, s := range strings.Split(sh[1], ",") {
		if s = strings.TrimSpace(s); s != "" {
			n, err := strconv.Atoi(s)
			if err != nil {
				return nil, fmt.Errorf("%s: shape %q", path, sh[1])
			}
			shape = append(shape, n)
		}
	}
	n := 1
	for _, v := range shape {
		n *= v
	}
	descr := strings.TrimLeft(d[1], "<|=")
	if strings.HasPrefix(d[1], ">") {
		return nil, fmt.Errorf("%s: big-endian data not supported", path)
	}
	need := func(size int) error {
		if len(data) < n*size {
			return fmt.Errorf("%s: %d bytes of data for %d elements of %s", path, len(data), n, descr)
		}
		return nil
	}
	switch descr {
	case "f4":
		if err := need(4); err != nil {
			return nil, err
		}
		return tensor.FromBytes(tensor.F32, append([]byte(nil), data[:4*n]...), shape...), nil
	case "f8":
		if err := need(8); err != nil {
			return nil, err
		}
		t := tensor.New(tensor.F32, shape...)
		for i := range t.F32() {
			t.F32()[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(data[8*i:])))
		}
		return t, nil
	case "i8":
		if err := need(8); err != nil {
			return nil, err
		}
		return tensor.FromBytes(tensor.I64, append([]byte(nil), data[:8*n]...), shape...), nil
	case "i4":
		if err := need(4); err != nil {
			return nil, err
		}
		return tensor.FromBytes(tensor.I32, append([]byte(nil), data[:4*n]...), shape...), nil
	case "u1":
		if err := need(1); err != nil {
			return nil, err
		}
		return tensor.FromBytes(tensor.U8, append([]byte(nil), data[:n]...), shape...), nil
	case "b1":
		if err := need(1); err != nil {
			return nil, err
		}
		return tensor.FromBytes(tensor.Bool, append([]byte(nil), data[:n]...), shape...), nil
	}
	return nil, fmt.Errorf("%s: dtype %q not supported", path, d[1])
}

func writeNPY(w io.Writer, t *tensor.Tensor) error {
	descr := map[tensor.DType]string{tensor.F32: "<f4", tensor.I64: "<i8", tensor.I32: "<i4", tensor.U8: "|u1", tensor.Bool: "|b1"}[t.DType()]
	if descr == "" {
		return fmt.Errorf("dtype %s not writable as .npy", t.DType())
	}
	dims := make([]string, len(t.Shape()))
	for i, d := range t.Shape() {
		dims[i] = strconv.Itoa(d)
	}
	shape := strings.Join(dims, ", ")
	if len(dims) == 1 {
		shape += ","
	}
	h := fmt.Sprintf("{'descr': '%s', 'fortran_order': False, 'shape': (%s), }", descr, shape)
	pad := 64 - (10+len(h)+1)%64
	if pad == 64 {
		pad = 0
	}
	h += strings.Repeat(" ", pad) + "\n"
	var b bytes.Buffer
	b.WriteString("\x93NUMPY\x01\x00")
	b.Write(binary.LittleEndian.AppendUint16(nil, uint16(len(h))))
	b.WriteString(h)
	b.Write(t.Bytes())
	_, err := w.Write(b.Bytes())
	return err
}
