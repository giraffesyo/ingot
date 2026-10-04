// Package trellis2 runs TRELLIS.2 (image-to-3D) on the ingot runtime, built
// in Go over the Hugging Face safetensors checkpoints: the DINOv3 image
// encoder, the sparse-structure flow transformer and its dense 3-D conv
// decoder, and the structured-latent flow transformers with their sparse
// conv decoders. Reference: microsoft/TRELLIS.2 (trellis2 package).
package trellis2

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// weights reads tensors from one checkpoint file, panicking with a
// loadError that the Build* entry points recover into an ordinary error —
// model builders then read like the reference module tree, without an
// error check per weight.
type weights struct {
	f      *safetensors.File
	prefix string
	widen  bool // hand Gemm f32 weights even where the file stores bf16
}

type loadError struct{ err error }

func (w weights) scope(p string) weights { return weights{w.f, w.prefix + p + ".", w.widen} }

// f32 returns prefix+name widened to float32 (biases, norms, small tables).
func (w weights) f32(name string) *tensor.Tensor {
	t, err := w.f.F32(w.prefix + name)
	if err != nil {
		panic(loadError{err})
	}
	return t
}

// raw returns prefix+name in its stored dtype, zero-copy: large bf16 Linear
// weights stay views of the mapped file and Gemm packs them straight from
// bf16.
func (w weights) raw(name string) *tensor.Tensor {
	t, err := w.f.Tensor(w.prefix + name)
	if err != nil {
		panic(loadError{err})
	}
	return t
}

// linear returns a Linear's weight in the form Gemm takes — f32 zero-copy,
// bf16 zero-copy unless the weights are to be widened (the CPU Gemm packs
// straight from bf16; the GPU executor places only f32 operands), anything
// else (f16) widened — and its f32 bias.
func (w weights) linear(name string) (wt, bias *tensor.Tensor) {
	info, ok := w.f.Info(w.prefix + name + ".weight")
	if !ok {
		panic(loadError{fmt.Errorf("trellis2: checkpoint has no %s%s.weight", w.prefix, name)})
	}
	if info.DType == "F32" || (info.DType == "BF16" && !w.widen) {
		wt = w.raw(name + ".weight")
	} else {
		wt = w.f32(name + ".weight")
	}
	return wt, w.f32(name + ".bias")
}

// catch converts a loadError panic into *err; other panics propagate.
func catch(err *error) {
	if r := recover(); r != nil {
		le, ok := r.(loadError)
		if !ok {
			panic(r)
		}
		*err = le.err
	}
}

// readJSON decodes path into v.
func readJSON(path string, v any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("trellis2: %w", err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		return fmt.Errorf("trellis2: %s: %w", path, err)
	}
	return nil
}
