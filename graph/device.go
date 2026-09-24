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

// GPUOption configures CompileGPU.
type GPUOption func(*GPUSession)

// GPUBF16 runs matrix products against constant weights (MatMul, Gemm,
// convolution GEMMs) in bf16: weights converted once, activations cast on
// the fly, f32 accumulation and outputs — the GPU's fast matrix path (~2.7x
// f32). Attention stays f32. Accuracy: bf16 operand rounding, relative
// error ~2⁻⁸·√K per product.
func GPUBF16() GPUOption { return func(s *GPUSession) { s.bf16 = true } }

// CompileOn compiles g for a device: "cpu" (or ""), "gpu" (an error where
// no GPU backend exists), "gpu-bf16" (GPU with GPUBF16) or "auto" (the GPU
// when available, else the CPU).
func CompileOn(g *Graph, device string) (Runner, error) {
	switch device {
	case "", "cpu":
		return Compile(g)
	case "gpu":
		return CompileGPU(g)
	case "gpu-bf16":
		return CompileGPU(g, GPUBF16())
	case "auto":
		if s, err := CompileGPU(g); err == nil {
			return s, nil
		}
		return Compile(g)
	}
	return nil, fmt.Errorf("graph: unknown device %q (want cpu, gpu, gpu-bf16 or auto)", device)
}
