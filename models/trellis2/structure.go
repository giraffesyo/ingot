package trellis2

import (
	"fmt"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/sparse"
	"github.com/giraffesyo/ingot/tensor"
)

// FlowCondition is one condition's cross-attention keys and values for
// every block, keyed by the BuildFlow input they feed.
type FlowCondition map[string]*tensor.Tensor

// FlowConditions evaluates BuildFlowCond over each of conds ([condTokens,
// cond_channels], all the same shape): what a sampling run computes once,
// instead of projecting the condition in every block of every evaluation.
func FlowConditions(cfg FlowConfig, f *safetensors.File, blocks int, conds ...*tensor.Tensor) ([]FlowCondition, error) {
	if len(conds) == 0 {
		return nil, nil
	}
	g, err := BuildFlowCond(cfg, f, conds[0].Shape()[0], blocks)
	if err != nil {
		return nil, err
	}
	s, err := graph.Compile(g)
	if err != nil {
		return nil, fmt.Errorf("trellis2: flow condition: %w", err)
	}
	out := make([]FlowCondition, len(conds))
	for i, c := range conds {
		// The outputs are kept, not released: they live as long as the run.
		res, err := s.Run(map[string]*tensor.Tensor{"cond": c})
		if err != nil {
			return nil, fmt.Errorf("trellis2: flow condition: %w", err)
		}
		out[i] = res
	}
	return out, nil
}

// stager is a runner that reads inputs it has staged in place (the GPU
// session: no copy into its memory on every Run).
type stager interface {
	Stage(t *tensor.Tensor) *tensor.Tensor
}

// FlowVelocity adapts a compiled BuildFlow graph to the sampler: cond and
// neg are the keys and values of the image features and of the negative
// condition (FlowConditions); x is [T, in_channels] flattened. proj, for a
// model with projected image features, is their [T, ProjChannels] tensor;
// the negative condition uses zeros in its place.
func FlowVelocity(r graph.Runner, tokens, channels int, cond, neg FlowCondition, proj *tensor.Tensor) Velocity {
	feeds := [2]map[string]*tensor.Tensor{}
	stage := func(t *tensor.Tensor) *tensor.Tensor {
		if s, ok := r.(stager); ok {
			return s.Stage(t)
		}
		return t
	}
	for i, c := range []FlowCondition{neg, cond} {
		feeds[i] = make(map[string]*tensor.Tensor, len(c)+3)
		for name, t := range c {
			feeds[i][name] = stage(t)
		}
	}
	if proj != nil {
		feeds[0]["proj"] = stage(tensor.New(tensor.F32, proj.Shape()...))
		feeds[1]["proj"] = stage(proj)
	}
	return func(x []float32, t float32, positive bool) ([]float32, error) {
		fd := feeds[0]
		if positive {
			fd = feeds[1]
		}
		fd["x"], fd["t"] = tensor.FromF32(x, tokens, channels), tensor.FromF32([]float32{t}, 1)
		res, err := r.Run(fd)
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
