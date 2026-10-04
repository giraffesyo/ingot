// Package sparse is the coordinate bookkeeping of sparse voxel tensors: a
// set of occupied integer cells, one feature row per cell. It builds the
// index tables — convolution neighbours, subdivision children — that the
// sparse ops consume as plain integer tensors; the features themselves stay
// ordinary [N, C] tensors.
package sparse

import (
	"fmt"

	"github.com/giraffesyo/ingot/kernels/par"
)

// Coord is a voxel's (x, y, z) cell.
type Coord = [3]int32

// Index maps occupied cells to their rows: an open-addressing table over
// the packed coordinate, read-only once built (safe for concurrent Lookup).
type Index struct {
	keys []uint64 // packed coordinate + 1; 0 = empty slot
	rows []int32
	mask uint64
}

// coordBits is the width of each packed axis: cells in [0, 2²¹).
const coordBits = 21

func pack(c Coord) uint64 {
	return uint64(uint32(c[0]))<<(2*coordBits) | uint64(uint32(c[1]))<<coordBits | uint64(uint32(c[2]))
}

func inRange(c Coord) bool {
	return c[0] >= 0 && c[1] >= 0 && c[2] >= 0 && c[0] < 1<<coordBits && c[1] < 1<<coordBits && c[2] < 1<<coordBits
}

// hash is the 64-bit finaliser of MurmurHash3.
func hash(k uint64) uint64 {
	k ^= k >> 33
	k *= 0xff51afd7ed558ccd
	k ^= k >> 33
	k *= 0xc4ceb9fe1a85ec53
	return k ^ k>>33
}

// NewIndex indexes coords (row i holds coords[i]). It fails on a cell
// outside [0, 2²¹)³ or one listed twice.
func NewIndex(coords []Coord) (*Index, error) {
	size := 16
	for size < 2*len(coords) {
		size *= 2
	}
	ix := &Index{keys: make([]uint64, size), rows: make([]int32, size), mask: uint64(size - 1)}
	for i, c := range coords {
		if !inRange(c) {
			return nil, fmt.Errorf("sparse: coordinate %v of row %d out of range", c, i)
		}
		k := pack(c) + 1
		s := hash(k) & ix.mask
		for ix.keys[s] != 0 {
			if ix.keys[s] == k {
				return nil, fmt.Errorf("sparse: coordinate %v listed twice (rows %d and %d)", c, ix.rows[s], i)
			}
			s = (s + 1) & ix.mask
		}
		ix.keys[s], ix.rows[s] = k, int32(i)
	}
	return ix, nil
}

// Lookup returns the row holding cell c, or -1.
func (ix *Index) Lookup(c Coord) int32 {
	if !inRange(c) {
		return -1
	}
	k := pack(c) + 1
	for s := hash(k) & ix.mask; ix.keys[s] != 0; s = (s + 1) & ix.mask {
		if ix.keys[s] == k {
			return ix.rows[s]
		}
	}
	return -1
}

// Neighbors returns the submanifold-convolution neighbour table of a k³
// kernel with the given dilation: out[i·k³ + v] is the row of the cell at
// coords[i] + offset(v), or -1 when that cell is empty. v enumerates
// offsets with x slowest and z fastest, each axis from −⌊k/2⌋·dilation —
// the order of a [Co, kx, ky, kz, Ci] weight's kernel axes.
func Neighbors(coords []Coord, k, dilation int) ([]int32, error) {
	ix, err := NewIndex(coords)
	if err != nil {
		return nil, err
	}
	V := k * k * k
	out := make([]int32, len(coords)*V)
	half := int32(k / 2 * dilation)
	d := int32(dilation)
	par.For(len(coords), 2048, func(i, _ int) {
		c := coords[i]
		row := out[i*V : (i+1)*V]
		v := 0
		for a := range int32(k) {
			for b := range int32(k) {
				for e := range int32(k) {
					if a*d == half && b*d == half && e*d == half {
						row[v] = int32(i)
					} else {
						row[v] = ix.Lookup(Coord{c[0] - half + a*d, c[1] - half + b*d, c[2] - half + e*d})
					}
					v++
				}
			}
		}
	})
	return out, nil
}

// Subdivide splits each cell into the children its mask selects: mask
// [len(coords), 8] marks sub-cell s of cell i, s = dx + 2·dy + 4·dz. It
// returns the children's coordinates at twice the resolution, in parent
// order with s ascending, and for each child the row parent·8 + s — the
// gather index into a [N·8, C] view of features laid out [N, 8, C].
func Subdivide(coords []Coord, mask []bool) (children []Coord, src []int32, err error) {
	if len(mask) != 8*len(coords) {
		return nil, nil, fmt.Errorf("sparse: subdivision mask has %d entries for %d cells", len(mask), len(coords))
	}
	n := 0
	for _, m := range mask {
		if m {
			n++
		}
	}
	children, src = make([]Coord, 0, n), make([]int32, 0, n)
	for i, c := range coords {
		for s := range int32(8) {
			if mask[i*8+int(s)] {
				children = append(children, Coord{2*c[0] + s&1, 2*c[1] + s>>1&1, 2*c[2] + s>>2&1})
				src = append(src, int32(i)*8+s)
			}
		}
	}
	return children, src, nil
}
