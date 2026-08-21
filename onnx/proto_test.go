package onnx

import (
	"encoding/binary"
	"math"
	"testing"
)

// Minimal protobuf encoder for building test fixtures.
type enc struct{ b []byte }

func (e *enc) varint(v uint64) {
	for v >= 0x80 {
		e.b = append(e.b, byte(v)|0x80)
		v >>= 7
	}
	e.b = append(e.b, byte(v))
}
func (e *enc) tag(f, wt int)        { e.varint(uint64(f)<<3 | uint64(wt)) }
func (e *enc) int64(f int, v int64) { e.tag(f, wtVarint); e.varint(uint64(v)) }
func (e *enc) str(f int, s string) {
	e.tag(f, wtLen)
	e.varint(uint64(len(s)))
	e.b = append(e.b, s...)
}
func (e *enc) bytes(f int, b []byte) {
	e.tag(f, wtLen)
	e.varint(uint64(len(b)))
	e.b = append(e.b, b...)
}
func (e *enc) msg(f int, sub *enc) { e.bytes(f, sub.b) }
func (e *enc) float(f int, v float32) {
	e.tag(f, wtI32)
	e.b = binary.LittleEndian.AppendUint32(e.b, math.Float32bits(v))
}
func (e *enc) packedInt64(f int, vs []int64) {
	var s enc
	for _, v := range vs {
		s.varint(uint64(v))
	}
	e.bytes(f, s.b)
}
func (e *enc) packedFloat(f int, vs []float32) {
	var s enc
	for _, v := range vs {
		s.b = binary.LittleEndian.AppendUint32(s.b, math.Float32bits(v))
	}
	e.bytes(f, s.b)
}

func buildTestModel() []byte {
	// initializer W: float [2,3] raw_data
	var w enc
	w.packedInt64(1, []int64{2, 3})
	w.int64(2, int64(Float))
	w.str(8, "W")
	var raw []byte
	for _, v := range []float32{1, 2, 3, 4, 5, 6} {
		raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(v))
	}
	w.bytes(9, raw)

	// initializer shape: int64 [2] int64_data (unpacked style mixed in)
	var sh enc
	sh.int64(1, 2)
	sh.int64(2, int64(Int64))
	sh.str(8, "shape")
	sh.int64(7, 1)
	sh.int64(7, 6)

	// node: MatMul(X, W) -> Y with attributes
	var attr enc
	attr.str(1, "alpha")
	attr.float(2, 0.5)
	attr.int64(20, int64(AttrFloat))
	var attr2 enc
	attr2.str(1, "perm")
	attr2.packedInt64(8, []int64{1, 0})
	attr2.int64(20, int64(AttrInts))
	var node enc
	node.str(1, "X")
	node.str(1, "W")
	node.str(2, "Y")
	node.str(3, "mm")
	node.str(4, "MatMul")
	node.msg(5, &attr)
	node.msg(5, &attr2)

	// input X: float [N, 2]
	var dim0 enc
	dim0.str(2, "N")
	var dim1 enc
	dim1.int64(1, 2)
	var shape enc
	shape.msg(1, &dim0)
	shape.msg(1, &dim1)
	var tt enc
	tt.int64(1, int64(Float))
	tt.msg(2, &shape)
	var tp enc
	tp.msg(1, &tt)
	var x enc
	x.str(1, "X")
	x.msg(2, &tp)

	var y enc
	y.str(1, "Y")

	var g enc
	g.msg(1, &node)
	g.str(2, "testgraph")
	g.msg(5, &w)
	g.msg(5, &sh)
	g.msg(11, &x)
	g.msg(12, &y)

	var op enc
	op.str(1, "")
	op.int64(2, 17)

	var m enc
	m.int64(1, 9)
	m.str(2, "test")
	m.msg(7, &g)
	m.msg(8, &op)
	m.int64(99, 42) // unknown field, must be skipped
	return m.b
}

func TestDecode(t *testing.T) {
	m, err := Decode(buildTestModel())
	if err != nil {
		t.Fatal(err)
	}
	if m.IRVersion != 9 || m.ProducerName != "test" {
		t.Fatalf("header: %+v", m)
	}
	if len(m.OpsetImport) != 1 || m.OpsetImport[0].Version != 17 {
		t.Fatalf("opset: %+v", m.OpsetImport)
	}
	g := m.Graph
	if g.Name != "testgraph" || len(g.Nodes) != 1 || len(g.Initializer) != 2 {
		t.Fatalf("graph: %+v", g)
	}
	n := g.Nodes[0]
	if n.OpType != "MatMul" || len(n.Input) != 2 || n.Input[1] != "W" || n.Output[0] != "Y" {
		t.Fatalf("node: %+v", n)
	}
	if a := n.Attr("alpha"); a == nil || a.Type != AttrFloat || a.F != 0.5 {
		t.Fatalf("attr alpha: %+v", a)
	}
	if a := n.Attr("perm"); a == nil || a.Type != AttrInts || len(a.Ints) != 2 || a.Ints[0] != 1 {
		t.Fatalf("attr perm: %+v", a)
	}
	w := g.Initializer[0]
	if w.Name != "W" || w.DataType != Float || len(w.Dims) != 2 || w.Dims[1] != 3 {
		t.Fatalf("W: %+v", w)
	}
	f, err := w.Float32s()
	if err != nil || len(f) != 6 || f[5] != 6 {
		t.Fatalf("W data: %v %v", f, err)
	}
	sh := g.Initializer[1]
	iv, err := sh.Int64s()
	if err != nil || len(iv) != 2 || iv[1] != 6 {
		t.Fatalf("shape data: %v %v", iv, err)
	}
	x := g.Input[0]
	if x.Name != "X" || x.ElemType != Float || !x.HasShape || len(x.Shape) != 2 ||
		x.Shape[0] != -1 || x.ShapeParams[0] != "N" || x.Shape[1] != 2 {
		t.Fatalf("input X: %+v", x)
	}
	if g.Output[0].Name != "Y" || g.Output[0].HasShape {
		t.Fatalf("output Y: %+v", g.Output[0])
	}
}

func TestTruncated(t *testing.T) {
	b := buildTestModel()
	for _, cut := range []int{1, 5, 20, len(b) / 2, len(b) - 1} {
		if _, err := Decode(b[:cut]); err == nil {
			t.Errorf("cut=%d: expected error", cut)
		}
	}
}

func TestFloat16(t *testing.T) {
	cases := map[uint16]float32{
		0x0000: 0, 0x3c00: 1, 0xc000: -2, 0x3555: 0.333251953125, 0x7bff: 65504,
		0x0001: 5.960464477539063e-08, 0x8000: float32(math.Copysign(0, -1)),
	}
	for h, want := range cases {
		if got := floatMToFloat32(h); got != want {
			t.Errorf("f16 %#04x: got %g want %g", h, got, want)
		}
	}
	if got := floatMToFloat32(0x7c00); !math.IsInf(float64(got), 1) {
		t.Errorf("f16 inf: %g", got)
	}
	if got := floatMToFloat32(0x7e00); !math.IsNaN(float64(got)) {
		t.Errorf("f16 nan: %g", got)
	}
	if got := bfloatMToFloat32(0x3f80); got != 1 {
		t.Errorf("bf16 1.0: %g", got)
	}
}
