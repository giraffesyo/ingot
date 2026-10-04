package trellis2

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"

	"github.com/giraffesyo/ingot/sparse"
)

// sphereGrid builds the decoder output a perfect dual grid would hold for a
// sphere of radius r (in cells) centred in a res³ grid: the cells any
// crossed edge touches, each owning its far-corner edges, vertices at cell
// centres. splitHi picks the quad diagonal through the weights.
func sphereGrid(res int, r float64, splitHi bool) ([]sparse.Coord, []float32) {
	c := float64(res) / 2
	inside := func(x, y, z int32) bool {
		dx, dy, dz := float64(x)-c, float64(y)-c, float64(z)-c
		return dx*dx+dy*dy+dz*dz < r*r
	}
	// crossed reports whether the edge leaving corner p along axis a
	// changes side.
	crossed := func(p [3]int32, a int) bool {
		q := p
		q[a]++
		return inside(p[0], p[1], p[2]) != inside(q[0], q[1], q[2])
	}
	var coords []sparse.Coord
	var feats []float32
	for x := range int32(res) {
		for y := range int32(res) {
			for z := range int32(res) {
				// The cell's corners are not all on one side.
				mixed, first := false, inside(x, y, z)
				for k := range 8 {
					if inside(x+int32(k&1), y+int32(k>>1&1), z+int32(k>>2&1)) != first {
						mixed = true
					}
				}
				if !mixed {
					continue
				}
				f := make([]float32, 7)
				// The owned edge along axis a starts at the far corner in
				// the other two axes.
				for a := range 3 {
					p := [3]int32{x + 1, y + 1, z + 1}
					p[a]--
					f[3+a] = -1
					if crossed(p, a) {
						f[3+a] = 1
					}
				}
				if splitHi == ((x+y+z)%2 == 0) {
					f[6] = 3
				}
				coords = append(coords, sparse.Coord{x, y, z})
				feats = append(feats, f...)
			}
		}
	}
	return coords, feats
}

// TestExtractMesh: the dual grid of a sphere is a closed 2-manifold (every
// edge on exactly two faces); orienting it gives one piece, consistently
// wound, enclosing about the sphere's volume, with outward normals.
func TestExtractMesh(t *testing.T) {
	const res, r = 24, 8.3
	for _, splitHi := range []bool{false, true} {
		coords, feats := sphereGrid(res, r, splitHi)
		m, err := ExtractMesh(coords, feats, res, 0.5)
		if err != nil {
			t.Fatal(err)
		}
		if len(m.Faces) == 0 {
			t.Fatal("no faces")
		}
		uses := map[edgeKey]int{}
		for _, f := range m.Faces {
			for e := range 3 {
				uses[undirected(f[e], f[(e+1)%3])]++
			}
		}
		for k, n := range uses {
			if n != 2 {
				t.Fatalf("edge %v is on %d faces, want 2", k, n)
			}
		}
		if p := m.OrientFaces(); p != 1 {
			t.Fatalf("%d connected pieces, want 1", p)
		}
		// Consistent winding: each directed edge appears exactly once.
		dir := map[[2]uint32]int{}
		var vol float64
		for _, f := range m.Faces {
			for e := range 3 {
				dir[[2]uint32{f[e], f[(e+1)%3]}]++
			}
			vol += tripleProduct(m.Vertices[f[0]], m.Vertices[f[1]], m.Vertices[f[2]]) / 6
		}
		for k, n := range dir {
			if n != 1 {
				t.Fatalf("directed edge %v appears %d times after orienting", k, n)
			}
		}
		want := 4.0 / 3 * math.Pi * math.Pow(r/res, 3)
		if math.Abs(vol-want)/want > 0.1 {
			t.Fatalf("enclosed volume %.5f, sphere is %.5f", vol, want)
		}
		m.ComputeNormals()
		m.Compact()
		for i, v := range m.Vertices {
			n := m.Normals[i]
			// Cell centres sit within a cell of the sphere, so the normal
			// points away from the grid centre (the origin).
			if d := float64(v[0]*n[0] + v[1]*n[1] + v[2]*n[2]); d <= 0 {
				t.Fatalf("vertex %d: normal %v points inward at %v", i, n, v)
			}
		}
		t.Logf("split=%v: %d vertices, %d faces, volume %.5f (sphere %.5f)", splitHi, len(m.Vertices), len(m.Faces), vol, want)
	}
}

// TestExtractMeshMissingCell: a crossed edge whose four cells are not all
// present makes no quad.
func TestExtractMeshMissingCell(t *testing.T) {
	coords := []sparse.Coord{{0, 0, 0}, {0, 0, 1}, {0, 1, 1}}
	feats := make([]float32, 3*7)
	feats[3] = 1 // cell 0's x-directed edge: needs (0,1,0) too
	m, err := ExtractMesh(coords, feats, 4, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Faces) != 0 {
		t.Fatalf("%d faces from an incomplete quad", len(m.Faces))
	}
	if _, err := ExtractMesh(coords, feats[:7], 4, 0.5); err == nil {
		t.Fatal("short features accepted")
	}
}

// TestWriteGLB parses the container back: header, chunk framing, accessor
// counts and bounds, the axis rotation, and the colour encoding.
func TestWriteGLB(t *testing.T) {
	m := &Mesh{
		Vertices:  [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 2, 0}, {0, 0, 3}},
		Faces:     [][3]uint32{{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2}},
		BaseColor: [][3]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0.5, 0.5, 0.5}},
		Metallic:  []float32{0, 0, 1, 1}, Roughness: []float32{0.2, 0.4, 0.6, 0.8}, Alpha: []float32{1, 1, 1, 1},
	}
	m.ComputeNormals()
	var buf bytes.Buffer
	if err := m.WriteGLB(&buf); err != nil {
		t.Fatal(err)
	}
	raw := buf.Bytes()
	u32 := func(off int) uint32 { return binary.LittleEndian.Uint32(raw[off:]) }
	if u32(0) != glbMagic || u32(4) != 2 || int(u32(8)) != len(raw) {
		t.Fatalf("bad header: magic %#x version %d length %d (file %d)", u32(0), u32(4), u32(8), len(raw))
	}
	jl := int(u32(12))
	if u32(16) != glbChunkJSON || jl%4 != 0 {
		t.Fatalf("bad JSON chunk: type %#x length %d", u32(16), jl)
	}
	bo := 20 + jl
	bl := int(u32(bo))
	if u32(bo+4) != glbChunkBIN || bl%4 != 0 || bo+8+bl != len(raw) {
		t.Fatalf("bad BIN chunk: type %#x length %d", u32(bo+4), bl)
	}
	var doc struct {
		Accessors []struct {
			BufferView, ComponentType, Count int
			Type                             string
			Min, Max                         []float32
		}
		BufferViews []struct{ ByteOffset, ByteLength int }
		Meshes      []struct {
			Primitives []struct {
				Attributes map[string]int
				Indices    int
			}
		}
		Materials []struct {
			PbrMetallicRoughness struct{ MetallicFactor, RoughnessFactor float64 }
		}
	}
	if err := json.Unmarshal(raw[20:20+jl], &doc); err != nil {
		t.Fatal(err)
	}
	prim := doc.Meshes[0].Primitives[0]
	pos := doc.Accessors[prim.Attributes["POSITION"]]
	if pos.Count != 4 || pos.Type != "VEC3" {
		t.Fatalf("position accessor %+v", pos)
	}
	// (x, y, z) → (x, z, −y): bounds x [0,1], y [0,3], z [−2,0].
	if want := []float32{0, 0, -2}; pos.Min[0] != want[0] || pos.Min[1] != want[1] || pos.Min[2] != want[2] {
		t.Fatalf("position min %v, want %v", pos.Min, want)
	}
	if want := []float32{1, 3, 0}; pos.Max[0] != want[0] || pos.Max[1] != want[1] || pos.Max[2] != want[2] {
		t.Fatalf("position max %v, want %v", pos.Max, want)
	}
	bin := raw[bo+8:]
	pv := doc.BufferViews[pos.BufferView]
	f := func(i int) float32 { return math.Float32frombits(binary.LittleEndian.Uint32(bin[pv.ByteOffset+4*i:])) }
	if f(6) != 0 || f(7) != 0 || f(8) != -2 { // vertex 2 = (0, 2, 0)
		t.Fatalf("vertex 2 stored as (%g, %g, %g)", f(6), f(7), f(8))
	}
	if idx := doc.Accessors[prim.Indices]; idx.Count != 12 || idx.ComponentType != gltfUint {
		t.Fatalf("index accessor %+v", idx)
	}
	col := doc.Accessors[prim.Attributes["COLOR_0"]]
	cv := doc.BufferViews[col.BufferView]
	c := func(i int) uint16 { return binary.LittleEndian.Uint16(bin[cv.ByteOffset+2*i:]) }
	if c(0) != 65535 || c(1) != 0 || c(3) != 65535 {
		t.Fatalf("vertex 0 colour (%d, %d, %d, %d)", c(0), c(1), c(2), c(3))
	}
	// sRGB 0.5 is linear ≈ 0.2140.
	if g := float64(c(12)) / 65535; math.Abs(g-0.2140) > 1e-3 {
		t.Fatalf("grey stored as %.4f, want linear 0.2140", g)
	}
	mr := doc.Materials[0].PbrMetallicRoughness
	if math.Abs(mr.MetallicFactor-0.5) > 1e-6 || math.Abs(mr.RoughnessFactor-0.5) > 1e-6 {
		t.Fatalf("material factors %+v", mr)
	}
}
