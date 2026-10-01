//go:build !(darwin && arm64)

package qwen3tts

import (
	"errors"

	"github.com/giraffesyo/ingot/safetensors"
)

func metalAvailable() bool { return false }

var errNoMetal = errors.New("qwen3tts: GPU decode needs Apple Metal (darwin/arm64)")

type metalLM struct{}

func newMetalLM(*safetensors.Set, LMConfig, string, int, []string, string, int, int, Quant) (*metalLM, error) {
	return nil, errNoMetal
}

func (*metalLM) reset()   {}
func (*metalLM) pos() int { return 0 }
func (*metalLM) run([]float32, int, int) ([]float32, []float32, error) {
	return nil, nil, errNoMetal
}
func (*metalLM) close() {}

type frameSampler struct {
	greedy      bool
	topK        int
	temperature float32
}

func (*metalLM) setTables(*safetensors.Set, []string) error { return errNoMetal }
func (*metalLM) runFrame([]float32, frameSampler, []float32, bool) ([]int64, [][]float32, error) {
	return nil, nil, errNoMetal
}
