//go:build !(darwin && arm64)

package graph

import (
	"errors"

	"github.com/giraffesyo/ingot/tensor"
)

// GPUSession is unavailable off darwin/arm64 (see the darwin build).
type GPUSession struct {
	*Session
	GPUSteps, CPUSteps int
}

// CompileGPU reports that no GPU backend exists on this platform.
func CompileGPU(*Graph) (*GPUSession, error) {
	return nil, errors.New("graph: GPU sessions need darwin/arm64 (Metal)")
}

func (s *GPUSession) Run(map[string]*tensor.Tensor) (map[string]*tensor.Tensor, error) {
	return nil, errors.New("graph: GPU sessions need darwin/arm64 (Metal)")
}

func (s *GPUSession) Close() {}
