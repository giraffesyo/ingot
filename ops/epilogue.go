package ops

import (
	"fmt"

	"github.com/giraffesyo/ingot/kernels/vek"
)

// epilogue is the fused tail of Conv/ConvTranspose:
//
//	y = post_scale · act(y + bias) + post_shift
//
// applied per output tile while it is still cache-resident. It is configured by
// the graph optimizer through "ingot_*" attributes (see graph/optimize.go);
// the ONNX loader never produces them.
//
//	ingot_act        string  relu | hardswish | hardsigmoid | sigmoid | silu | clip | leakyrelu
//	ingot_act_alpha  float   hardsigmoid alpha / clip min / leakyrelu alpha
//	ingot_act_beta   float   hardsigmoid beta  / clip max
//	ingot_post_scale float   default 1
//	ingot_post_shift float   default 0
type epilogue struct {
	act          string
	alpha, beta  float32
	scale, shift float32
	post         bool
}

func parseEpilogue(n NodeInfo) (epilogue, error) {
	a := n.Attrs
	e := epilogue{act: a.String("ingot_act", ""), alpha: a.Float("ingot_act_alpha", 0), beta: a.Float("ingot_act_beta", 0)}
	switch e.act {
	case "", "relu", "hardswish", "hardsigmoid", "sigmoid", "silu", "clip", "leakyrelu":
	default:
		return e, n.Errorf("unknown ingot_act %q", e.act)
	}
	if a.Has("ingot_post_scale") || a.Has("ingot_post_shift") {
		e.post = true
		e.scale = a.Float("ingot_post_scale", 1)
		e.shift = a.Float("ingot_post_shift", 0)
	}
	return e, nil
}

func (e *epilogue) active() bool { return e.act != "" || e.post }

// apply runs the epilogue in place on one contiguous output segment.
func (e *epilogue) apply(row []float32) {
	switch e.act {
	case "relu":
		vek.Relu(row, row)
	case "hardswish":
		vek.HardSwish(row, row)
	case "hardsigmoid":
		vek.HardSigmoid(row, row, e.alpha, e.beta)
	case "sigmoid":
		sigmoidVec(row, row)
	case "silu":
		vek.SiLU(row, row)
	case "clip":
		vek.Clip(row, row, e.alpha, e.beta)
	case "leakyrelu":
		vek.LeakyRelu(row, row, e.alpha)
	}
	if e.post {
		if e.scale != 1 {
			vek.MulScalar(row, row, e.scale)
		}
		if e.shift != 0 {
			vek.AddScalar(row, row, e.shift)
		}
	}
}

func (e *epilogue) String() string {
	if !e.active() {
		return ""
	}
	return fmt.Sprintf("act=%s(%g,%g) post=%gx+%g", e.act, e.alpha, e.beta, e.scale, e.shift)
}
