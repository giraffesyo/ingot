// Package qwenimage runs Qwen-Image-2.1 (text-to-image and editing) on the
// ingot runtime, built in Go over the Hugging Face safetensors checkpoint:
// the Qwen3-VL text encoder, the 7B single-stream DiT, and the Wan-style
// VAE. Reference: diffusers QwenImage21Pipeline (0.37.0.dev0).
package qwenimage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// weights reads f32 tensors from a checkpoint, panicking with a loadError
// that the Build* entry points recover into an ordinary error — model
// builders then read like the reference module tree, without an error
// check per weight.
type weights struct {
	set    *safetensors.Set
	prefix string
}

type loadError struct{ err error }

func (w weights) scope(p string) weights { return weights{w.set, w.prefix + p + "."} }

func (w weights) f32(name string) *tensor.Tensor {
	t, err := w.set.F32(w.prefix + name)
	if err != nil {
		panic(loadError{err})
	}
	return t
}

// raw returns prefix+name in its stored dtype, zero-copy: large bf16
// Linear weights stay as views of the mapped file and Gemm packs them
// straight from bf16.
func (w weights) raw(name string) *tensor.Tensor {
	t, err := w.set.Tensor(w.prefix + name)
	if err != nil {
		panic(loadError{err})
	}
	return t
}

// has reports whether the checkpoint holds prefix+name.
func (w weights) has(name string) bool {
	_, ok := w.set.Info(w.prefix + name)
	return ok
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

// readConfig decodes dir/config.json into v.
func readConfig(dir string, v any) error { return readJSON(dir, "config.json", v) }

// readJSON decodes dir/name into v.
func readJSON(dir, name string, v any) error {
	raw, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return fmt.Errorf("qwenimage: %w", err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		return fmt.Errorf("qwenimage: %s/%s: %w", dir, name, err)
	}
	return nil
}

// reshaped returns a view of t with a new shape (same numel).
func reshaped(t *tensor.Tensor, shape ...int) *tensor.Tensor { return t.Reshape(shape...) }

// scaled returns a new tensor t·s.
func scaled(t *tensor.Tensor, s float32) *tensor.Tensor {
	out := tensor.New(tensor.F32, t.Shape()...)
	for i, v := range t.F32() {
		out.F32()[i] = v * s
	}
	return out
}
