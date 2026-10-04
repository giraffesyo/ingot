package trellis2

import (
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/sparse"
)

// Mesh is a triangle mesh with per-vertex PBR attributes, in the model's
// frame: the unit cube [-0.5, 0.5]³, z up.
type Mesh struct {
	Vertices [][3]float32
	Normals  [][3]float32 // unit, per vertex; nil until ComputeNormals
	Faces    [][3]uint32
	// Per-vertex material, each in [0, 1]; nil when the mesh carries none.
	BaseColor [][3]float32 // sRGB
	Metallic  []float32
	Roughness []float32
	Alpha     []float32
}

// Each grid edge is shared by four cells; a cell owns the three edges
// meeting at its far corner, one per axis. quadCells lists, per axis, the
// offsets of the four cells around that edge, in the order they wind
// around it.
var quadCells = [3][4]sparse.Coord{
	{{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}}, // x-directed edge
	{{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}}, // y-directed edge
	{{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}, // z-directed edge
}

// ExtractMesh turns the shape decoder's output into a mesh by the flexible
// dual grid: every cell holds one vertex, and every grid edge the surface
// crosses becomes a quad joining the four cells around it. feats is
// [len(coords), 7] flattened — per cell the vertex position within the
// cell (3 logits; sigmoid, stretched by margin on either side), whether
// each of the cell's three owned edges is crossed (3 logits) and a weight
// choosing which diagonal splits its quads (1, through softplus). The grid
// is res cells across the unit cube.
//
// Quads whose four cells are not all present are dropped. Faces come out
// in cell order, then axis; their winding is not yet consistent — see
// OrientFaces.
func ExtractMesh(coords []sparse.Coord, feats []float32, res int, margin float64) (*Mesh, error) {
	const ch = 7
	if len(feats) != ch*len(coords) {
		return nil, fmt.Errorf("trellis2: %d shape features for %d cells (want %d each)", len(feats), len(coords), ch)
	}
	ix, err := sparse.NewIndex(coords)
	if err != nil {
		return nil, err
	}
	m := &Mesh{Vertices: make([][3]float32, len(coords))}
	weight := make([]float32, len(coords))
	size := 1 / float32(res)
	for i, c := range coords {
		f := feats[i*ch : (i+1)*ch]
		for a := range 3 {
			d := float32((1+2*margin)*sigmoid(float64(f[a])) - margin)
			m.Vertices[i][a] = (float32(c[a])+d)*size - 0.5
		}
		weight[i] = float32(softplus(float64(f[6])))
	}
	for i, c := range coords {
		for a := range 3 {
			if feats[i*ch+3+a] <= 0 {
				continue
			}
			var q [4]uint32
			ok := true
			for k, off := range quadCells[a] {
				j := ix.Lookup(sparse.Coord{c[0] + off[0], c[1] + off[1], c[2] + off[2]})
				if j < 0 {
					ok = false
					break
				}
				q[k] = uint32(j)
			}
			if !ok {
				continue
			}
			if weight[q[0]]*weight[q[2]] > weight[q[1]]*weight[q[3]] {
				m.Faces = append(m.Faces, [3]uint32{q[0], q[1], q[2]}, [3]uint32{q[0], q[2], q[3]})
			} else {
				m.Faces = append(m.Faces, [3]uint32{q[0], q[1], q[3]}, [3]uint32{q[3], q[1], q[2]})
			}
		}
	}
	return m, nil
}

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

func softplus(x float64) float64 {
	if x > 20 {
		return x
	}
	return math.Log1p(math.Exp(x))
}

// SetMaterial attaches the texture decoder's output to the mesh: feats is
// [vertices, 6] flattened — base colour (3), metallic, roughness, alpha —
// in [-1, 1], one row per vertex (the texture decoder runs over the shape
// decoder's cells, so rows and vertices correspond).
func (m *Mesh) SetMaterial(feats []float32) error {
	const ch = 6
	n := len(m.Vertices)
	if len(feats) != ch*n {
		return fmt.Errorf("trellis2: %d texture features for %d vertices (want %d each)", len(feats), n, ch)
	}
	unit := func(v float32) float32 { return min(1, max(0, v*0.5+0.5)) }
	m.BaseColor = make([][3]float32, n)
	m.Metallic, m.Roughness, m.Alpha = make([]float32, n), make([]float32, n), make([]float32, n)
	for i := range n {
		f := feats[i*ch : (i+1)*ch]
		m.BaseColor[i] = [3]float32{unit(f[0]), unit(f[1]), unit(f[2])}
		m.Metallic[i], m.Roughness[i], m.Alpha[i] = unit(f[3]), unit(f[4]), unit(f[5])
	}
	return nil
}

type edgeKey struct{ a, b uint32 }

func undirected(a, b uint32) edgeKey {
	if a > b {
		a, b = b, a
	}
	return edgeKey{a, b}
}

// OrientFaces makes the winding consistent: within each connected piece,
// faces sharing an edge traverse it in opposite directions, and the piece
// as a whole winds counter-clockwise seen from outside (positive enclosed
// volume). Edges shared by more than two faces do not propagate
// orientation. It reports the number of connected pieces.
func (m *Mesh) OrientFaces() int {
	type pair struct {
		f [2]int32
		n int8
	}
	edges := make(map[edgeKey]pair, len(m.Faces)*3/2)
	for fi, f := range m.Faces {
		for e := range 3 {
			k := undirected(f[e], f[(e+1)%3])
			p := edges[k]
			if p.n < 2 {
				p.f[p.n] = int32(fi)
			}
			p.n = min(p.n+1, 3)
			edges[k] = p
		}
	}
	// directed reports whether face f traverses a→b in that order.
	directed := func(f [3]uint32, a, b uint32) bool {
		return (f[0] == a && f[1] == b) || (f[1] == a && f[2] == b) || (f[2] == a && f[0] == b)
	}
	seen := make([]bool, len(m.Faces))
	var stack, piece []int32
	pieces := 0
	for start := range m.Faces {
		if seen[start] {
			continue
		}
		pieces++
		seen[start] = true
		stack, piece = append(stack[:0], int32(start)), piece[:0]
		for len(stack) > 0 {
			fi := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			piece = append(piece, fi)
			f := m.Faces[fi]
			for e := range 3 {
				a, b := f[e], f[(e+1)%3]
				p := edges[undirected(a, b)]
				if p.n != 2 {
					continue
				}
				o := p.f[0]
				if o == fi {
					o = p.f[1]
				}
				if seen[o] {
					continue
				}
				seen[o] = true
				if g := m.Faces[o]; directed(g, a, b) {
					m.Faces[o] = [3]uint32{g[0], g[2], g[1]}
				}
				stack = append(stack, o)
			}
		}
		var vol float64
		for _, fi := range piece {
			f := m.Faces[fi]
			vol += tripleProduct(m.Vertices[f[0]], m.Vertices[f[1]], m.Vertices[f[2]])
		}
		if vol < 0 {
			for _, fi := range piece {
				f := m.Faces[fi]
				m.Faces[fi] = [3]uint32{f[0], f[2], f[1]}
			}
		}
	}
	return pieces
}

func tripleProduct(a, b, c [3]float32) float64 {
	ax, ay, az := float64(a[0]), float64(a[1]), float64(a[2])
	bx, by, bz := float64(b[0]), float64(b[1]), float64(b[2])
	cx, cy, cz := float64(c[0]), float64(c[1]), float64(c[2])
	return ax*(by*cz-bz*cy) + ay*(bz*cx-bx*cz) + az*(bx*cy-by*cx)
}

// ComputeNormals sets per-vertex normals: the area-weighted mean of the
// adjacent faces' normals. Vertices on no face get +z.
func (m *Mesh) ComputeNormals() {
	acc := make([][3]float64, len(m.Vertices))
	for _, f := range m.Faces {
		a, b, c := m.Vertices[f[0]], m.Vertices[f[1]], m.Vertices[f[2]]
		ux, uy, uz := float64(b[0]-a[0]), float64(b[1]-a[1]), float64(b[2]-a[2])
		vx, vy, vz := float64(c[0]-a[0]), float64(c[1]-a[1]), float64(c[2]-a[2])
		n := [3]float64{uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx}
		for _, v := range f {
			acc[v][0] += n[0]
			acc[v][1] += n[1]
			acc[v][2] += n[2]
		}
	}
	m.Normals = make([][3]float32, len(m.Vertices))
	for i, n := range acc {
		l := math.Sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
		if l == 0 {
			m.Normals[i] = [3]float32{0, 0, 1}
			continue
		}
		m.Normals[i] = [3]float32{float32(n[0] / l), float32(n[1] / l), float32(n[2] / l)}
	}
}

// Compact drops vertices no face uses, renumbering the faces.
func (m *Mesh) Compact() {
	remap := make([]int32, len(m.Vertices))
	for i := range remap {
		remap[i] = -1
	}
	n := 0
	for fi, f := range m.Faces {
		for e, v := range f {
			if remap[v] < 0 {
				remap[v] = int32(n)
				n++
			}
			m.Faces[fi][e] = uint32(remap[v])
		}
	}
	m.Vertices = compact(m.Vertices, remap, n)
	m.Normals = compact(m.Normals, remap, n)
	m.BaseColor = compact(m.BaseColor, remap, n)
	m.Metallic = compact(m.Metallic, remap, n)
	m.Roughness = compact(m.Roughness, remap, n)
	m.Alpha = compact(m.Alpha, remap, n)
}

func compact[T any](xs []T, remap []int32, n int) []T {
	if xs == nil {
		return nil
	}
	out := make([]T, n)
	for i, j := range remap {
		if j >= 0 {
			out[j] = xs[i]
		}
	}
	return out
}
