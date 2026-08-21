package onnx

import "fmt"

// DataType mirrors onnx.TensorProto.DataType.
type DataType int32

const (
	Undefined  DataType = 0
	Float      DataType = 1
	Uint8      DataType = 2
	Int8       DataType = 3
	Uint16     DataType = 4
	Int16      DataType = 5
	Int32      DataType = 6
	Int64      DataType = 7
	String     DataType = 8
	Bool       DataType = 9
	Float16    DataType = 10
	Double     DataType = 11
	Uint32     DataType = 12
	Uint64     DataType = 13
	Complex64  DataType = 14
	Complex128 DataType = 15
	BFloat16   DataType = 16
)

var dataTypeNames = map[DataType]string{
	Undefined: "undefined", Float: "float", Uint8: "uint8", Int8: "int8", Uint16: "uint16",
	Int16: "int16", Int32: "int32", Int64: "int64", String: "string", Bool: "bool",
	Float16: "float16", Double: "double", Uint32: "uint32", Uint64: "uint64",
	Complex64: "complex64", Complex128: "complex128", BFloat16: "bfloat16",
}

func (d DataType) String() string {
	if s, ok := dataTypeNames[d]; ok {
		return s
	}
	return fmt.Sprintf("datatype(%d)", int32(d))
}

// AttrType mirrors onnx.AttributeProto.AttributeType.
type AttrType int32

const (
	AttrUndefined AttrType = 0
	AttrFloat     AttrType = 1
	AttrInt       AttrType = 2
	AttrString    AttrType = 3
	AttrTensor    AttrType = 4
	AttrGraph     AttrType = 5
	AttrFloats    AttrType = 6
	AttrInts      AttrType = 7
	AttrStrings   AttrType = 8
	AttrTensors   AttrType = 9
	AttrGraphs    AttrType = 10
)

// Model is onnx.ModelProto.
type Model struct {
	IRVersion       int64
	ProducerName    string
	ProducerVersion string
	Domain          string
	ModelVersion    int64
	DocString       string
	Graph           *Graph
	OpsetImport     []OpsetID
	Metadata        map[string]string
}

// OpsetID is onnx.OperatorSetIdProto.
type OpsetID struct {
	Domain  string
	Version int64
}

// Graph is onnx.GraphProto.
type Graph struct {
	Name        string
	Nodes       []*Node
	Initializer []*Tensor
	Input       []*ValueInfo
	Output      []*ValueInfo
	ValueInfo   []*ValueInfo
	DocString   string
}

// Node is onnx.NodeProto.
type Node struct {
	Name      string
	OpType    string
	Domain    string
	Input     []string
	Output    []string
	Attribute []*Attribute
	DocString string
}

// Attr returns the named attribute or nil.
func (n *Node) Attr(name string) *Attribute {
	for _, a := range n.Attribute {
		if a.Name == name {
			return a
		}
	}
	return nil
}

// Attribute is onnx.AttributeProto.
type Attribute struct {
	Name    string
	Type    AttrType
	F       float32
	I       int64
	S       []byte
	T       *Tensor
	G       *Graph
	Floats  []float32
	Ints    []int64
	Strings [][]byte
	Tensors []*Tensor
	Graphs  []*Graph
}

// Tensor is onnx.TensorProto. Exactly one of RawData or the typed *Data
// slices is populated (per the ONNX spec); see Tensor.Float32s etc. in data.go.
type Tensor struct {
	Name         string
	Dims         []int64
	DataType     DataType
	RawData      []byte
	FloatData    []float32
	Int32Data    []int32
	Int64Data    []int64
	DoubleData   []float64
	Uint64Data   []uint64
	StringData   [][]byte
	DocString    string
	ExternalData map[string]string
	DataLocation int32 // 0 = default, 1 = external
}

// ValueInfo is onnx.ValueInfoProto (tensor types only).
type ValueInfo struct {
	Name      string
	DocString string
	// ElemType is 0 if the type is not a tensor type / unknown.
	ElemType DataType
	// Shape dims: >=0 fixed; -1 symbolic (see ShapeParams).
	Shape       []int64
	ShapeParams []string // symbolic names, "" for fixed dims
	HasShape    bool
}

// Decode parses a serialized ModelProto.
func Decode(b []byte) (*Model, error) {
	m := &Model{Metadata: map[string]string{}}
	r := reader{buf: b}
	for !r.eof() {
		f, wt, err := r.tag()
		if err != nil {
			return nil, err
		}
		switch f {
		case 1:
			m.IRVersion, err = r.int64Field(wt)
		case 2:
			m.ProducerName, err = r.stringField(wt)
		case 3:
			m.ProducerVersion, err = r.stringField(wt)
		case 4:
			m.Domain, err = r.stringField(wt)
		case 5:
			m.ModelVersion, err = r.int64Field(wt)
		case 6:
			m.DocString, err = r.stringField(wt)
		case 7:
			var gb []byte
			if gb, err = r.bytesField(wt); err == nil {
				m.Graph, err = decodeGraph(gb)
			}
		case 8:
			var ob []byte
			if ob, err = r.bytesField(wt); err == nil {
				var o OpsetID
				o, err = decodeOpsetID(ob)
				m.OpsetImport = append(m.OpsetImport, o)
			}
		case 14:
			var kb []byte
			if kb, err = r.bytesField(wt); err == nil {
				var k, v string
				k, v, err = decodeStringEntry(kb)
				m.Metadata[k] = v
			}
		default:
			err = r.skip(wt)
		}
		if err != nil {
			return nil, fmt.Errorf("onnx: ModelProto field %d: %w", f, err)
		}
	}
	if m.Graph == nil {
		return nil, fmt.Errorf("onnx: model has no graph")
	}
	return m, nil
}

func decodeOpsetID(b []byte) (OpsetID, error) {
	var o OpsetID
	r := reader{buf: b}
	for !r.eof() {
		f, wt, err := r.tag()
		if err != nil {
			return o, err
		}
		switch f {
		case 1:
			o.Domain, err = r.stringField(wt)
		case 2:
			o.Version, err = r.int64Field(wt)
		default:
			err = r.skip(wt)
		}
		if err != nil {
			return o, err
		}
	}
	return o, nil
}

func decodeStringEntry(b []byte) (k, v string, err error) {
	r := reader{buf: b}
	for !r.eof() {
		f, wt, err := r.tag()
		if err != nil {
			return "", "", err
		}
		switch f {
		case 1:
			k, err = r.stringField(wt)
		case 2:
			v, err = r.stringField(wt)
		default:
			err = r.skip(wt)
		}
		if err != nil {
			return "", "", err
		}
	}
	return k, v, nil
}

func decodeGraph(b []byte) (*Graph, error) {
	g := &Graph{}
	r := reader{buf: b}
	for !r.eof() {
		f, wt, err := r.tag()
		if err != nil {
			return nil, err
		}
		var sub []byte
		switch f {
		case 1:
			if sub, err = r.bytesField(wt); err == nil {
				var n *Node
				n, err = decodeNode(sub)
				g.Nodes = append(g.Nodes, n)
			}
		case 2:
			g.Name, err = r.stringField(wt)
		case 5:
			if sub, err = r.bytesField(wt); err == nil {
				var t *Tensor
				t, err = decodeTensor(sub)
				g.Initializer = append(g.Initializer, t)
			}
		case 10:
			g.DocString, err = r.stringField(wt)
		case 11, 12, 13:
			if sub, err = r.bytesField(wt); err == nil {
				var vi *ValueInfo
				vi, err = decodeValueInfo(sub)
				switch f {
				case 11:
					g.Input = append(g.Input, vi)
				case 12:
					g.Output = append(g.Output, vi)
				default:
					g.ValueInfo = append(g.ValueInfo, vi)
				}
			}
		default:
			err = r.skip(wt)
		}
		if err != nil {
			return nil, fmt.Errorf("GraphProto field %d: %w", f, err)
		}
	}
	return g, nil
}

func decodeNode(b []byte) (*Node, error) {
	n := &Node{}
	r := reader{buf: b}
	for !r.eof() {
		f, wt, err := r.tag()
		if err != nil {
			return nil, err
		}
		var s string
		switch f {
		case 1:
			if s, err = r.stringField(wt); err == nil {
				n.Input = append(n.Input, s)
			}
		case 2:
			if s, err = r.stringField(wt); err == nil {
				n.Output = append(n.Output, s)
			}
		case 3:
			n.Name, err = r.stringField(wt)
		case 4:
			n.OpType, err = r.stringField(wt)
		case 5:
			var sub []byte
			if sub, err = r.bytesField(wt); err == nil {
				var a *Attribute
				a, err = decodeAttribute(sub)
				n.Attribute = append(n.Attribute, a)
			}
		case 6:
			n.DocString, err = r.stringField(wt)
		case 7:
			n.Domain, err = r.stringField(wt)
		default:
			err = r.skip(wt)
		}
		if err != nil {
			return nil, fmt.Errorf("NodeProto field %d: %w", f, err)
		}
	}
	return n, nil
}

func decodeAttribute(b []byte) (*Attribute, error) {
	a := &Attribute{}
	r := reader{buf: b}
	for !r.eof() {
		f, wt, err := r.tag()
		if err != nil {
			return nil, err
		}
		var sub []byte
		switch f {
		case 1:
			a.Name, err = r.stringField(wt)
		case 2:
			a.F, err = r.floatField(wt)
		case 3:
			a.I, err = r.int64Field(wt)
		case 4:
			a.S, err = r.bytesField(wt)
		case 5:
			if sub, err = r.bytesField(wt); err == nil {
				a.T, err = decodeTensor(sub)
			}
		case 6:
			if sub, err = r.bytesField(wt); err == nil {
				a.G, err = decodeGraph(sub)
			}
		case 7:
			a.Floats, err = r.repeatedFloat(wt, a.Floats)
		case 8:
			a.Ints, err = r.repeatedInt64(wt, a.Ints)
		case 9:
			if sub, err = r.bytesField(wt); err == nil {
				a.Strings = append(a.Strings, sub)
			}
		case 10:
			if sub, err = r.bytesField(wt); err == nil {
				var t *Tensor
				t, err = decodeTensor(sub)
				a.Tensors = append(a.Tensors, t)
			}
		case 11:
			if sub, err = r.bytesField(wt); err == nil {
				var g *Graph
				g, err = decodeGraph(sub)
				a.Graphs = append(a.Graphs, g)
			}
		case 20:
			var v int64
			v, err = r.int64Field(wt)
			a.Type = AttrType(v)
		default:
			err = r.skip(wt)
		}
		if err != nil {
			return nil, fmt.Errorf("AttributeProto field %d: %w", f, err)
		}
	}
	return a, nil
}

func decodeTensor(b []byte) (*Tensor, error) {
	t := &Tensor{}
	r := reader{buf: b}
	for !r.eof() {
		f, wt, err := r.tag()
		if err != nil {
			return nil, err
		}
		switch f {
		case 1:
			t.Dims, err = r.repeatedInt64(wt, t.Dims)
		case 2:
			var v int64
			v, err = r.int64Field(wt)
			t.DataType = DataType(v)
		case 4:
			t.FloatData, err = r.repeatedFloat(wt, t.FloatData)
		case 5:
			t.Int32Data, err = r.repeatedInt32(wt, t.Int32Data)
		case 6:
			var s []byte
			if s, err = r.bytesField(wt); err == nil {
				t.StringData = append(t.StringData, s)
			}
		case 7:
			t.Int64Data, err = r.repeatedInt64(wt, t.Int64Data)
		case 8:
			t.Name, err = r.stringField(wt)
		case 9:
			t.RawData, err = r.bytesField(wt)
		case 10:
			t.DoubleData, err = r.repeatedDouble(wt, t.DoubleData)
		case 11:
			t.Uint64Data, err = r.repeatedUint64(wt, t.Uint64Data)
		case 12:
			t.DocString, err = r.stringField(wt)
		case 13:
			var sub []byte
			if sub, err = r.bytesField(wt); err == nil {
				var k, v string
				if k, v, err = decodeStringEntry(sub); err == nil {
					if t.ExternalData == nil {
						t.ExternalData = map[string]string{}
					}
					t.ExternalData[k] = v
				}
			}
		case 14:
			var v int64
			v, err = r.int64Field(wt)
			t.DataLocation = int32(v)
		default:
			err = r.skip(wt)
		}
		if err != nil {
			return nil, fmt.Errorf("TensorProto field %d: %w", f, err)
		}
	}
	return t, nil
}

func decodeValueInfo(b []byte) (*ValueInfo, error) {
	vi := &ValueInfo{}
	r := reader{buf: b}
	for !r.eof() {
		f, wt, err := r.tag()
		if err != nil {
			return nil, err
		}
		switch f {
		case 1:
			vi.Name, err = r.stringField(wt)
		case 2:
			var sub []byte
			if sub, err = r.bytesField(wt); err == nil {
				err = decodeTypeProto(sub, vi)
			}
		case 3:
			vi.DocString, err = r.stringField(wt)
		default:
			err = r.skip(wt)
		}
		if err != nil {
			return nil, fmt.Errorf("ValueInfoProto field %d: %w", f, err)
		}
	}
	return vi, nil
}

// decodeTypeProto handles TypeProto{tensor_type: Tensor{elem_type, shape}}.
func decodeTypeProto(b []byte, vi *ValueInfo) error {
	r := reader{buf: b}
	for !r.eof() {
		f, wt, err := r.tag()
		if err != nil {
			return err
		}
		if f != 1 { // only tensor_type
			if err := r.skip(wt); err != nil {
				return err
			}
			continue
		}
		sub, err := r.bytesField(wt)
		if err != nil {
			return err
		}
		tr := reader{buf: sub}
		for !tr.eof() {
			tf, twt, err := tr.tag()
			if err != nil {
				return err
			}
			switch tf {
			case 1:
				var v int64
				v, err = tr.int64Field(twt)
				vi.ElemType = DataType(v)
			case 2:
				var sb []byte
				if sb, err = tr.bytesField(twt); err == nil {
					err = decodeShape(sb, vi)
				}
			default:
				err = tr.skip(twt)
			}
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func decodeShape(b []byte, vi *ValueInfo) error {
	vi.HasShape = true
	r := reader{buf: b}
	for !r.eof() {
		f, wt, err := r.tag()
		if err != nil {
			return err
		}
		if f != 1 {
			if err := r.skip(wt); err != nil {
				return err
			}
			continue
		}
		db, err := r.bytesField(wt)
		if err != nil {
			return err
		}
		dim, param := int64(-1), ""
		dr := reader{buf: db}
		for !dr.eof() {
			df, dwt, err := dr.tag()
			if err != nil {
				return err
			}
			switch df {
			case 1:
				dim, err = dr.int64Field(dwt)
			case 2:
				param, err = dr.stringField(dwt)
			default:
				err = dr.skip(dwt)
			}
			if err != nil {
				return err
			}
		}
		vi.Shape = append(vi.Shape, dim)
		vi.ShapeParams = append(vi.ShapeParams, param)
	}
	return nil
}
