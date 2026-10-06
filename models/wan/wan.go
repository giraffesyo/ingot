// Package wan runs Wan 2.2 TI2V-5B (text- and image-to-video) on the ingot
// runtime, built in Go over the Diffusers-layout safetensors checkpoint
// (Wan-AI/Wan2.2-TI2V-5B-Diffusers): the umT5-XXL text encoder, the 5B
// diffusion transformer with 3-D rotary positions and per-token timesteps,
// the Wan 2.2 VAE (16x16 spatial, 4x temporal compression, causal 3-D
// convolutions decoded chunk by chunk) and the UniPC flow-matching
// scheduler. Image-to-video conditions on the input image's latent as the
// first latent frame, held at timestep 0. Reference: diffusers
// WanImageToVideoPipeline with expand_timesteps (0.35+).
package wan

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// weights reads tensors from a checkpoint, panicking with a loadError that
// the Build* entry points recover into an ordinary error — model builders
// then read like the reference module tree, without an error check per
// weight.
type weights struct {
	set    *safetensors.Set
	prefix string
	widen  bool // hand Gemm f32 weights even where the file stores bf16
}

type loadError struct{ err error }

func (w weights) scope(p string) weights { return weights{w.set, w.prefix + p + ".", w.widen} }

// f32 returns prefix+name as float32 (zero-copy for F32 storage).
func (w weights) f32(name string) *tensor.Tensor {
	t, err := w.set.F32(w.prefix + name)
	if err != nil {
		panic(loadError{err})
	}
	return t
}

// raw returns prefix+name in its stored dtype, zero-copy.
func (w weights) raw(name string) *tensor.Tensor {
	t, err := w.set.Tensor(w.prefix + name)
	if err != nil {
		panic(loadError{err})
	}
	return t
}

// matrix returns a Linear weight in the form Gemm takes: f32 zero-copy,
// bf16 zero-copy unless widen (the CPU Gemm packs straight from bf16; the
// GPU executor places only f32 operands), anything else widened.
func (w weights) matrix(name string) *tensor.Tensor {
	info, ok := w.set.Info(w.prefix + name)
	if !ok {
		panic(loadError{fmt.Errorf("wan: checkpoint has no %s%s", w.prefix, name)})
	}
	if info.DType == "F32" || (info.DType == "BF16" && !w.widen) {
		return w.raw(name)
	}
	return w.f32(name)
}

// linear returns a Linear's weight and f32 bias.
func (w weights) linear(name string) (wt, bias *tensor.Tensor) {
	return w.matrix(name + ".weight"), w.f32(name + ".bias")
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

// readJSON decodes dir/name into v.
func readJSON(dir, name string, v any) error {
	raw, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return fmt.Errorf("wan: %w", err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		return fmt.Errorf("wan: %s/%s: %w", dir, name, err)
	}
	return nil
}
