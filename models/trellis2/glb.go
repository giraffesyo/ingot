package trellis2

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
)

// glTF constants (the 2.0 binary container and its accessor enums).
const (
	glbMagic      = 0x46546C67 // "glTF"
	glbChunkJSON  = 0x4E4F534A
	glbChunkBIN   = 0x004E4942
	gltfFloat     = 5126
	gltfUint      = 5125
	gltfUshort    = 5123
	gltfArrayBuf  = 34962
	gltfElemBuf   = 34963
	gltfTriangles = 4
)

// WriteGLB writes the mesh as a binary glTF 2.0 file: one primitive with
// positions, normals when present, and the material as vertex attributes.
// Base colour and alpha go in COLOR_0 (linear, 16-bit); glTF has no
// per-vertex metallic or roughness, so those become the material's
// factors, averaged over the vertices. The model's z-up frame is rotated
// to glTF's y-up, as the reference exporter does: (x, y, z) → (x, z, −y).
func (m *Mesh) WriteGLB(w io.Writer) error {
	if len(m.Vertices) == 0 || len(m.Faces) == 0 {
		return fmt.Errorf("trellis2: mesh has no geometry to write")
	}
	var bin bytes.Buffer
	type view struct {
		Buffer     int `json:"buffer"`
		ByteOffset int `json:"byteOffset"`
		ByteLength int `json:"byteLength"`
		Target     int `json:"target"`
	}
	type accessor struct {
		BufferView    int       `json:"bufferView"`
		ComponentType int       `json:"componentType"`
		Normalized    bool      `json:"normalized,omitempty"`
		Count         int       `json:"count"`
		Type          string    `json:"type"`
		Min           []float32 `json:"min,omitempty"`
		Max           []float32 `json:"max,omitempty"`
	}
	var views []view
	var accessors []accessor
	add := func(a accessor, target int, write func()) int {
		for bin.Len()%4 != 0 {
			bin.WriteByte(0)
		}
		off := bin.Len()
		write()
		views = append(views, view{ByteOffset: off, ByteLength: bin.Len() - off, Target: target})
		a.BufferView = len(views) - 1
		accessors = append(accessors, a)
		return len(accessors) - 1
	}
	vec3 := func(vs [][3]float32) func() {
		return func() {
			for _, v := range vs {
				_ = binary.Write(&bin, binary.LittleEndian, [3]float32{v[0], v[2], -v[1]})
			}
		}
	}

	lo := [3]float32{float32(math.Inf(1)), float32(math.Inf(1)), float32(math.Inf(1))}
	hi := [3]float32{float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1))}
	for _, v := range m.Vertices {
		for a, x := range [3]float32{v[0], v[2], -v[1]} {
			lo[a], hi[a] = min(lo[a], x), max(hi[a], x)
		}
	}
	attrs := map[string]int{}
	attrs["POSITION"] = add(accessor{ComponentType: gltfFloat, Count: len(m.Vertices), Type: "VEC3", Min: lo[:], Max: hi[:]},
		gltfArrayBuf, vec3(m.Vertices))
	if m.Normals != nil {
		attrs["NORMAL"] = add(accessor{ComponentType: gltfFloat, Count: len(m.Normals), Type: "VEC3"}, gltfArrayBuf, vec3(m.Normals))
	}
	metallic, roughness := 0.0, 1.0
	if m.BaseColor != nil {
		attrs["COLOR_0"] = add(accessor{ComponentType: gltfUshort, Normalized: true, Count: len(m.BaseColor), Type: "VEC4"}, gltfArrayBuf, func() {
			q := func(v float32) uint16 { return uint16(min(1, max(0, v))*65535 + 0.5) }
			for i, c := range m.BaseColor {
				_ = binary.Write(&bin, binary.LittleEndian, [4]uint16{q(srgbToLinear(c[0])), q(srgbToLinear(c[1])), q(srgbToLinear(c[2])), q(m.Alpha[i])})
			}
		})
		metallic, roughness = mean(m.Metallic), mean(m.Roughness)
	}
	indices := add(accessor{ComponentType: gltfUint, Count: 3 * len(m.Faces), Type: "SCALAR"}, gltfElemBuf, func() {
		_ = binary.Write(&bin, binary.LittleEndian, m.Faces)
	})
	for bin.Len()%4 != 0 {
		bin.WriteByte(0)
	}

	doc := map[string]any{
		"asset":  map[string]any{"version": "2.0", "generator": "ingot"},
		"scene":  0,
		"scenes": []any{map[string]any{"nodes": []int{0}}},
		"nodes":  []any{map[string]any{"mesh": 0}},
		"meshes": []any{map[string]any{"primitives": []any{map[string]any{
			"attributes": attrs, "indices": indices, "material": 0, "mode": gltfTriangles,
		}}}},
		"materials": []any{map[string]any{
			"pbrMetallicRoughness": map[string]any{"metallicFactor": metallic, "roughnessFactor": roughness},
			"doubleSided":          true,
		}},
		"buffers":     []any{map[string]any{"byteLength": bin.Len()}},
		"bufferViews": views,
		"accessors":   accessors,
	}
	js, err := json.Marshal(doc)
	if err != nil {
		return fmt.Errorf("trellis2: glb: %w", err)
	}
	for len(js)%4 != 0 {
		js = append(js, ' ')
	}
	total := 12 + 8 + len(js) + 8 + bin.Len()
	if total > math.MaxUint32 {
		return fmt.Errorf("trellis2: mesh needs %d bytes, past the GLB limit", total)
	}
	var out bytes.Buffer
	for _, v := range []uint32{glbMagic, 2, uint32(total), uint32(len(js)), glbChunkJSON} {
		_ = binary.Write(&out, binary.LittleEndian, v)
	}
	out.Write(js)
	_ = binary.Write(&out, binary.LittleEndian, uint32(bin.Len()))
	_ = binary.Write(&out, binary.LittleEndian, uint32(glbChunkBIN))
	if _, err := w.Write(out.Bytes()); err != nil {
		return fmt.Errorf("trellis2: glb: %w", err)
	}
	if _, err := w.Write(bin.Bytes()); err != nil {
		return fmt.Errorf("trellis2: glb: %w", err)
	}
	return nil
}

func srgbToLinear(c float32) float32 {
	if c <= 0.04045 {
		return c / 12.92
	}
	return float32(math.Pow((float64(c)+0.055)/1.055, 2.4))
}

func mean(xs []float32) float64 {
	if len(xs) == 0 {
		return 0
	}
	var s float64
	for _, x := range xs {
		s += float64(x)
	}
	return s / float64(len(xs))
}
