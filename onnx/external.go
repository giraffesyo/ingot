package onnx

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

// DecodeFile reads and decodes an ONNX model from disk, resolving any
// external-data tensors (weights stored in a sidecar file referenced by the
// TensorProto external_data field). Sidecar paths are resolved relative to the
// model file's directory. Use this instead of Decode for models exported with
// external data (common for large models and the PyTorch dynamo exporter).
func DecodeFile(path string) (*Model, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	m, err := Decode(raw)
	if err != nil {
		return nil, err
	}
	if err := resolveExternal(m, filepath.Dir(path)); err != nil {
		return nil, err
	}
	return m, nil
}

// resolveExternal loads external tensor data into RawData for every tensor in
// the graph (initializers and attribute tensors) that references it. Sidecar
// files are memoised so each is read once.
func resolveExternal(m *Model, dir string) error {
	cache := map[string][]byte{}
	load := func(rel string) ([]byte, error) {
		if b, ok := cache[rel]; ok {
			return b, nil
		}
		b, err := os.ReadFile(filepath.Join(dir, filepath.Clean(rel)))
		if err != nil {
			return nil, err
		}
		cache[rel] = b
		return b, nil
	}
	var visit func(t *Tensor) error
	visit = func(t *Tensor) error {
		if t == nil || t.DataLocation != 1 {
			return nil
		}
		loc := t.ExternalData["location"]
		if loc == "" {
			return fmt.Errorf("tensor %q: external data without location", t.Name)
		}
		buf, err := load(loc)
		if err != nil {
			return fmt.Errorf("tensor %q external data: %w", t.Name, err)
		}
		off, length := 0, len(buf)
		if s := t.ExternalData["offset"]; s != "" {
			if off, err = strconv.Atoi(s); err != nil {
				return fmt.Errorf("tensor %q: bad offset %q", t.Name, s)
			}
		}
		if s := t.ExternalData["length"]; s != "" {
			if length, err = strconv.Atoi(s); err != nil {
				return fmt.Errorf("tensor %q: bad length %q", t.Name, s)
			}
		}
		if off < 0 || off+length > len(buf) {
			return fmt.Errorf("tensor %q: external range [%d,%d) out of file %q (len %d)", t.Name, off, off+length, loc, len(buf))
		}
		t.RawData = buf[off : off+length]
		t.DataLocation = 0
		return nil
	}
	for _, init := range m.Graph.Initializer {
		if err := visit(init); err != nil {
			return err
		}
	}
	// Attribute tensors (e.g. Constant values) can also be external.
	for _, n := range m.Graph.Nodes {
		for _, a := range n.Attribute {
			if err := visit(a.T); err != nil {
				return err
			}
			for _, at := range a.Tensors {
				if err := visit(at); err != nil {
					return err
				}
			}
		}
	}
	return nil
}
