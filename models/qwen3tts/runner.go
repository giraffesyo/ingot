package qwen3tts

import (
	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/kernels/vek"
	"github.com/giraffesyo/ingot/tensor"
)

// lmRunner runs one decoder stack (talker or code predictor) with its KV
// cache: the CPU graph (cpuLM) or the Metal decoder (metalLM).
type lmRunner interface {
	reset()
	pos() int
	// run feeds n input rows (x [n, in]) at the next positions and returns
	// the last row's final-normed hidden state and, for head >= 0, that
	// output head's logits. Both alias runner buffers until the next run.
	run(x []float32, n, head int) (hidden, logits []float32, err error)
	close()
}

// cpuLM is a compiled decode graph (buildLM) plus its output heads: in the
// graph (graphHead, the talker's single codec head) or applied on the host
// per call (the code predictor's one head per codebook).
type cpuLM struct {
	sess      *graph.Session
	dec       *graph.Decode
	rope      *rope
	in        int
	graphHead bool
	heads     []*tensor.Tensor   // host heads, bf16 [n, D]
	headsI8   []*gemm.I8GWeights // Int8: quantised host heads
	hid       []float32
	logits    []float32
}

func newCPULM(g *graph.Graph, r *rope, in, maxT, hidden int, graphHead bool, heads []*tensor.Tensor, q Quant) (*cpuLM, error) {
	sess, err := graph.CompileDecode(g)
	if err != nil {
		return nil, err
	}
	c := &cpuLM{sess: sess, dec: sess.NewDecode(maxT), rope: r, in: in, graphHead: graphHead, heads: heads, hid: make([]float32, hidden)}
	for _, h := range heads {
		c.logits = make([]float32, max(len(c.logits), h.Dim(0)))
		if q == Int8 && vek.Q8Fast {
			c.headsI8 = append(c.headsI8, gemm.QuantizeI8GBF16(h.BF16(), h.Dim(1), h.Dim(0), h.Dim(1)))
		}
	}
	return c, nil
}

func (c *cpuLM) reset()   { c.dec.Reset() }
func (c *cpuLM) pos() int { return c.dec.Pos() }
func (c *cpuLM) close()   {}

func (c *cpuLM) run(x []float32, n, head int) ([]float32, []float32, error) {
	cos, sin := c.rope.at(c.dec.Pos(), n)
	out, err := c.sess.RunDecode(c.dec, map[string]*tensor.Tensor{
		"x": tensor.FromF32(x[:n*c.in], n, c.in), "cos": cos, "sin": sin}, n)
	if err != nil {
		return nil, nil, err
	}
	defer c.sess.Release(out)
	copy(c.hid, out["hidden"].F32())
	if head < 0 {
		return c.hid, nil, nil
	}
	if c.graphHead {
		l := out["logits"].F32()
		if len(c.logits) < len(l) {
			c.logits = make([]float32, len(l))
		}
		copy(c.logits, l)
		return c.hid, c.logits[:len(l)], nil
	}
	h := c.heads[head]
	logits := c.logits[:h.Dim(0)]
	if c.headsI8 != nil {
		gemm.GemvI8G(logits, c.headsI8[head], c.hid, 1, 0)
	} else {
		gemvBF16(logits, h, c.hid)
	}
	return c.hid, logits, nil
}
