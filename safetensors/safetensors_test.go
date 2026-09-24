package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type entry struct {
	name  string
	dtype string
	shape []int
	data  []byte
}

// writeFile writes a .safetensors file; pad adds spaces to the header so the
// data section starts at a chosen alignment.
func writeFile(t *testing.T, path string, es []entry, pad int) {
	t.Helper()
	hdr := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	var data []byte
	for _, e := range es {
		hdr[e.name] = map[string]any{"dtype": e.dtype, "shape": e.shape,
			"data_offsets": []int{len(data), len(data) + len(e.data)}}
		data = append(data, e.data...)
	}
	h, err := json.Marshal(hdr)
	if err != nil {
		t.Fatal(err)
	}
	h = append(h, []byte(strings.Repeat(" ", pad))...)
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	out = append(append(out, h...), data...)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}
}

func f32Bytes(v ...float32) []byte {
	var b []byte
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(x))
	}
	return b
}

func u16Bytes(v ...uint16) []byte {
	var b []byte
	for _, x := range v {
		b = binary.LittleEndian.AppendUint16(b, x)
	}
	return b
}

func TestReadDTypes(t *testing.T) {
	bf := func(x float32) uint16 { return uint16(math.Float32bits(x) >> 16) }
	for _, pad := range []int{0, 1, 2, 3} { // shifts data alignment
		path := filepath.Join(t.TempDir(), "m.safetensors")
		f64 := binary.LittleEndian.AppendUint64(nil, math.Float64bits(-2.5))
		f64 = binary.LittleEndian.AppendUint64(f64, math.Float64bits(1e-3))
		writeFile(t, path, []entry{
			{"w.f32", "F32", []int{2, 3}, f32Bytes(1, -2, 3.5, 0, 1e-7, -1e30)},
			{"w.bf16", "BF16", []int{4}, u16Bytes(bf(1), bf(-2), bf(0.5), bf(3e38))},
			{"w.f16", "F16", []int{3}, u16Bytes(0x3c00, 0xc000, 0x0001)}, // 1, -2, smallest subnormal
			{"w.f64", "F64", []int{2}, f64},
			{"scalar", "F32", []int{}, f32Bytes(7)},
		}, pad)
		f, err := Open(path)
		if err != nil {
			t.Fatal(err)
		}
		check := func(name string, want ...float32) {
			t.Helper()
			x, err := f.F32(name)
			if err != nil {
				t.Fatalf("pad %d %s: %v", pad, name, err)
			}
			got := x.F32()
			if len(got) != len(want) {
				t.Fatalf("pad %d %s: %d values, want %d", pad, name, len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("pad %d %s[%d] = %g, want %g", pad, name, i, got[i], want[i])
				}
			}
		}
		check("w.f32", 1, -2, 3.5, 0, 1e-7, -1e30)
		check("w.bf16", 1, -2, 0.5, math.Float32frombits(uint32(bf(3e38))<<16))
		check("w.f16", 1, -2, float32(math.Ldexp(1, -24)))
		check("w.f64", -2.5, 1e-3)
		check("scalar", 7)
		if x, _ := f.F32("w.f32"); !x.Shape().Equal([]int{2, 3}) {
			t.Fatalf("shape %v", x.Shape())
		}
		if raw, err := f.Tensor("w.bf16"); err != nil || raw.DType().String() != "bf16" || len(raw.Bytes()) != 8 {
			t.Fatalf("raw bf16 view: %v %v", raw, err)
		}
		if f.Metadata["format"] != "pt" {
			t.Fatalf("metadata %v", f.Metadata)
		}
		if _, err := f.F32("missing"); err == nil {
			t.Fatal("missing tensor: want error")
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestHalfToF32(t *testing.T) {
	for _, c := range []struct {
		h    uint16
		want float64
	}{
		{0x0000, 0}, {0x3c00, 1}, {0xbc00, -1}, {0x7bff, 65504}, {0x0400, math.Ldexp(1, -14)},
		{0x03ff, math.Ldexp(1023, -24)}, {0x0001, math.Ldexp(1, -24)}, {0x3555, 0.333251953125},
	} {
		if got := HalfToF32(c.h); float64(got) != c.want {
			t.Errorf("HalfToF32(%#04x) = %g, want %g", c.h, got, c.want)
		}
	}
	if !math.IsInf(float64(HalfToF32(0x7c00)), 1) || !math.IsNaN(float64(HalfToF32(0x7e00))) {
		t.Error("inf/nan")
	}
	if math.Signbit(float64(HalfToF32(0x8000))) != true {
		t.Error("-0 lost its sign")
	}
}

func TestMalformed(t *testing.T) {
	dir := t.TempDir()
	write := func(name string, b []byte) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	hdr := func(h string, data int) []byte {
		b := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
		return append(append(b, h...), make([]byte, data)...)
	}
	for name, b := range map[string][]byte{
		"short":     {1, 2, 3},
		"hdrlen":    binary.LittleEndian.AppendUint64(nil, 1<<40),
		"json":      hdr("{not json", 0),
		"dtype":     hdr(`{"a":{"dtype":"Q7","shape":[1],"data_offsets":[0,4]}}`, 4),
		"range":     hdr(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, 4),
		"size":      hdr(`{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}`, 8),
		"negdim":    hdr(`{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}}`, 0),
		"backwards": hdr(`{"a":{"dtype":"F32","shape":[0],"data_offsets":[4,0]}}`, 4),
	} {
		if _, err := Open(write(name+".safetensors", b)); err == nil {
			t.Errorf("%s: want error", name)
		}
	}
}

func TestOpenDirSharded(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "m-00001-of-00002.safetensors"), []entry{{"a", "F32", []int{1}, f32Bytes(1)}}, 0)
	writeFile(t, filepath.Join(dir, "m-00002-of-00002.safetensors"), []entry{{"b", "F32", []int{2}, f32Bytes(2, 3)}}, 0)
	idx := `{"metadata":{},"weight_map":{"a":"m-00001-of-00002.safetensors","b":"m-00002-of-00002.safetensors"}}`
	if err := os.WriteFile(filepath.Join(dir, "m.safetensors.index.json"), []byte(idx), 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := OpenDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if got := s.Names(); len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("names %v", got)
	}
	b, err := s.F32("b")
	if err != nil || b.F32()[1] != 3 {
		t.Fatalf("b: %v %v", b, err)
	}

	// Same name in two shards is an error, not a silent pick.
	writeFile(t, filepath.Join(dir, "m-00002-of-00002.safetensors"), []entry{{"a", "F32", []int{1}, f32Bytes(9)}}, 0)
	if _, err := OpenDir(dir); err == nil {
		t.Fatal("duplicate tensor across shards: want error")
	}
}

// TestRealCheckpoint opens the local Qwen-Image-2.1 snapshot when present:
// every header parses and a bf16 and an f32 tensor convert with the shapes
// the configs promise.
func TestRealCheckpoint(t *testing.T) {
	home, _ := os.UserHomeDir()
	snaps, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots/*"))
	if len(snaps) == 0 {
		t.Skip("Qwen-Image-2.1 snapshot not in the HF cache")
	}
	for _, c := range []struct {
		sub, name, dtype string
		shape            []int
	}{
		{"transformer", "img_in.weight", "BF16", []int{4096, 64}},
		{"vae", "decoder.conv_in.weight", "F32", []int{1152, 64, 3, 3}},
		{"text_encoder", "lm_head.weight", "BF16", []int{151936, 4096}},
	} {
		s, err := OpenDir(filepath.Join(snaps[0], c.sub))
		if err != nil {
			t.Fatal(err)
		}
		info, ok := s.Info(c.name)
		if !ok || info.DType != c.dtype {
			t.Fatalf("%s/%s: info %+v", c.sub, c.name, info)
		}
		if c.name != "lm_head.weight" { // 1.2 GB; header check only
			x, err := s.F32(c.name)
			if err != nil || !x.Shape().Equal(c.shape) {
				t.Fatalf("%s/%s: %v %v", c.sub, c.name, x, err)
			}
			var nonzero int
			for _, v := range x.F32() {
				if v != 0 && !math.IsNaN(float64(v)) {
					nonzero++
				}
			}
			if nonzero < len(x.F32())/2 {
				t.Fatalf("%s/%s: only %d/%d nonzero finite values", c.sub, c.name, nonzero, len(x.F32()))
			}
		}
		s.Close()
	}
}

func TestTensorIdentity(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.safetensors")
	writeFile(t, path, []entry{{"w", "BF16", []int{2}, u16Bytes(1, 2)}}, 0)
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	a, _ := f.Tensor("w")
	b, _ := f.Tensor("w")
	if a != b {
		t.Fatal("Tensor must return the same view per name (shared weight caches key on it)")
	}
}
