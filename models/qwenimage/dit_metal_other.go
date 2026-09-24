//go:build !(darwin && arm64)

package qwenimage

import (
	"errors"

	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// MetalDiT is unavailable off darwin/arm64.
type MetalDiT struct{ Fast bool }

func NewMetalDiT(DiTConfig, *safetensors.Set, *DiTLayout, int) (*MetalDiT, error) {
	return nil, errors.New("qwenimage: Metal DiT needs darwin/arm64")
}
func (*MetalDiT) Prefix(txt, cond *tensor.Tensor) error     { return errors.New("unavailable") }
func (*MetalDiT) SetPrefix(map[string]*tensor.Tensor) error { return errors.New("unavailable") }
func (*MetalDiT) Step(*tensor.Tensor, float32) (*tensor.Tensor, error) {
	return nil, errors.New("unavailable")
}
func (*MetalDiT) Close() {}

// MetalTextEncoder is unavailable off darwin/arm64.
type MetalTextEncoder struct{}

func NewMetalTextEncoder(TextConfig, *safetensors.Set) (*MetalTextEncoder, error) {
	return nil, errors.New("qwenimage: Metal text encoder needs darwin/arm64")
}
func (*MetalTextEncoder) Encode([]int64, int, int) (*tensor.Tensor, error) {
	return nil, errors.New("unavailable")
}
func (*MetalTextEncoder) Close() {}
