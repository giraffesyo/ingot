package trellis2

import (
	"fmt"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/sparse"
	"github.com/giraffesyo/ingot/tensor"
)

// FlowVelocity adapts a compiled BuildFlow graph to the sampler: cond and
// neg are the image features and the negative condition ([condTokens,
// cond_channels]); x is [T, in_channels] flattened.
func FlowVelocity(r graph.Runner, tokens, channels int, cond, neg *tensor.Tensor) Velocity {
	return func(x []float32, t float32, positive bool) ([]float32, error) {
		c := cond
		if !positive {
			c = neg
		}
		res, err := r.Run(map[string]*tensor.Tensor{
			"x": tensor.FromF32(x, tokens, channels), "t": tensor.FromF32([]float32{t}, 1), "cond": c,
		})
		if err != nil {
			return nil, fmt.Errorf("trellis2: flow step: %w", err)
		}
		v := append([]float32(nil), res["v"].F32()...)
		r.Release(res)
		return v, nil
	}
}

// OccupiedCells turns the structure decoder's logits over an n³ grid
// (row-major x, y, z) into the occupied cells of a res³ grid, res dividing
// n: a cell is occupied when any of the logits it covers is positive. Cells
// come out in row-major order.
func OccupiedCells(logits []float32, n, res int) ([]sparse.Coord, error) {
	if res <= 0 || n%res != 0 || len(logits) != n*n*n {
		return nil, fmt.Errorf("trellis2: %d logits do not form a %d³ grid divisible to %d³", len(logits), n, res)
	}
	r := n / res
	var out []sparse.Coord
	for x := range res {
		for y := range res {
			for z := range res {
				if cellOccupied(logits, n, r, x, y, z) {
					out = append(out, sparse.Coord{int32(x), int32(y), int32(z)})
				}
			}
		}
	}
	return out, nil
}

func cellOccupied(logits []float32, n, r, x, y, z int) bool {
	for a := range r {
		for b := range r {
			row := logits[((x*r+a)*n+y*r+b)*n+z*r:]
			for _, v := range row[:r] {
				if v > 0 {
					return true
				}
			}
		}
	}
	return false
}
