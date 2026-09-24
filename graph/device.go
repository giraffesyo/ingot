package graph

import (
	"fmt"

	"github.com/giraffesyo/ingot/tensor"
)

// Runner executes a compiled graph: a CPU Session or a GPUSession.
type Runner interface {
	Run(feeds map[string]*tensor.Tensor) (map[string]*tensor.Tensor, error)
	// Release hands a Run's outputs back to the runner's buffer pool (see
	// Session.Release).
	Release(res map[string]*tensor.Tensor)
}

// CompileOn compiles g for a device: "cpu" (or ""), "gpu" (an error where
// no GPU backend exists) or "auto" (the GPU when available, else the CPU).
func CompileOn(g *Graph, device string) (Runner, error) {
	switch device {
	case "", "cpu":
		return Compile(g)
	case "gpu":
		return CompileGPU(g)
	case "auto":
		if s, err := CompileGPU(g); err == nil {
			return s, nil
		}
		return Compile(g)
	}
	return nil, fmt.Errorf("graph: unknown device %q (want cpu, gpu or auto)", device)
}
