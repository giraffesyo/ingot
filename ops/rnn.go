package ops

import (
	"github.com/giraffesyo/ingot/kernels/gemm"
	"github.com/giraffesyo/ingot/kernels/vek"
	"github.com/giraffesyo/ingot/tensor"
)

// lstmOp: ONNX LSTM, layout 0 ([T,B,I]), default activations (sigmoid, tanh,
// tanh), forward/reverse/bidirectional. sequence_lens, peepholes, clip and
// custom activations error loudly. Gate order is iofc per spec.
//
// The input projection for all timesteps runs as one GEMM (X·Wᵀ); the
// recurrent projection is a per-step GEMM on [B,4H]. tanh runs through
// vek.Sigmoid via tanh(x) = 2σ(2x)−1.
type lstmOp struct {
	n      NodeInfo
	hidden int
	dirs   []bool // per direction: reversed?
}

func rnnDirections(n NodeInfo) ([]bool, error) {
	switch d := n.Attrs.String("direction", "forward"); d {
	case "forward":
		return []bool{false}, nil
	case "reverse":
		return []bool{true}, nil
	case "bidirectional":
		return []bool{false, true}, nil
	default:
		return nil, n.Errorf("direction %q unsupported", d)
	}
}

func sigmoidInPlace(x []float32) { vek.Sigmoid(x, x) }

func tanhInPlace(x []float32) {
	for i, v := range x {
		x[i] = 2 * v
	}
	vek.Sigmoid(x, x)
	for i, v := range x {
		x[i] = 2*v - 1
	}
}

func (o *lstmOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 3 || in[0] == nil || in[1] == nil || in[2] == nil {
		return nil, o.n.Errorf("LSTM: need X, W, R")
	}
	if len(in) > 4 && in[4] != nil {
		return nil, o.n.Errorf("LSTM: sequence_lens is not supported")
	}
	if len(in) > 7 && in[7] != nil {
		return nil, o.n.Errorf("LSTM: peepholes are not supported")
	}
	x, w, r := in[0], in[1], in[2]
	xs := x.Shape()
	if len(xs) != 3 {
		return nil, o.n.Errorf("LSTM: want X [T,B,I], got %v", xs)
	}
	T, B, I := xs[0], xs[1], xs[2]
	H := o.hidden
	D := len(o.dirs)
	if !w.Shape().Equal(tensor.Shape{D, 4 * H, I}) || !r.Shape().Equal(tensor.Shape{D, 4 * H, H}) {
		return nil, o.n.Errorf("LSTM: W %v / R %v vs D=%d H=%d I=%d", w.Shape(), r.Shape(), D, H, I)
	}
	var bias []float32
	if len(in) > 3 && in[3] != nil {
		if in[3].Numel() != D*8*H {
			return nil, o.n.Errorf("LSTM: B has %d elements, want %d", in[3].Numel(), D*8*H)
		}
		bias = in[3].F32()
	}
	var ih, ic []float32
	if len(in) > 5 && in[5] != nil {
		ih = in[5].F32()
	}
	if len(in) > 6 && in[6] != nil {
		ic = in[6].F32()
	}

	y := ctx.NewUninit(tensor.F32, T, D, B, H)
	yh := ctx.NewUninit(tensor.F32, D, B, H)
	yc := ctx.NewUninit(tensor.F32, D, B, H)
	xf, wf, rf := x.F32(), w.F32(), r.F32()
	yf, yhf, ycf := y.F32(), yh.F32(), yc.F32()

	scr := ctx.NewUninit(tensor.F32, T*B*4*H+B*4*H+2*B*H)
	sf := scr.F32()
	G := sf[:T*B*4*H]
	gates := sf[T*B*4*H : T*B*4*H+B*4*H]
	hState := sf[T*B*4*H+B*4*H : T*B*4*H+B*4*H+B*H]
	cState := sf[T*B*4*H+B*4*H+B*H:]

	for d, rev := range o.dirs {
		wd := wf[d*4*H*I:]
		rd := rf[d*4*H*H:]
		// G = X · Wᵀ for every timestep at once.
		gemm.SgemmT(false, true, T*B, 4*H, I, 1, xf, I, wd, I, 0, G, 4*H)
		var bsum []float32
		if bias != nil {
			bsum = make([]float32, 4*H)
			wb := bias[d*8*H:]
			for j := 0; j < 4*H; j++ {
				bsum[j] = wb[j] + wb[4*H+j]
			}
		}
		if ih != nil {
			copy(hState, ih[d*B*H:(d+1)*B*H])
		} else {
			clear(hState)
		}
		if ic != nil {
			copy(cState, ic[d*B*H:(d+1)*B*H])
		} else {
			clear(cState)
		}
		for step := 0; step < T; step++ {
			t := step
			if rev {
				t = T - 1 - step
			}
			copy(gates, G[t*B*4*H:(t+1)*B*4*H])
			gemm.SgemmT(false, true, B, 4*H, H, 1, hState, H, rd, H, 1, gates, 4*H)
			for b := 0; b < B; b++ {
				g := gates[b*4*H:]
				if bsum != nil {
					vek.Add(g[:4*H], g[:4*H], bsum)
				}
				gi, go_, gf_, gc := g[:H], g[H:2*H], g[2*H:3*H], g[3*H:4*H]
				sigmoidInPlace(gi)
				sigmoidInPlace(go_)
				sigmoidInPlace(gf_)
				tanhInPlace(gc)
				cRow := cState[b*H:]
				hRow := hState[b*H:]
				for j := 0; j < H; j++ {
					c := gf_[j]*cRow[j] + gi[j]*gc[j]
					cRow[j] = c
				}
				// h = o * tanh(c): tanh into the gc scratch.
				copy(gc, cRow[:H])
				tanhInPlace(gc[:H])
				for j := 0; j < H; j++ {
					hRow[j] = go_[j] * gc[j]
				}
				copy(yf[((t*D+d)*B+b)*H:], hRow[:H])
			}
		}
		copy(yhf[d*B*H:], hState[:B*H])
		copy(ycf[d*B*H:], cState[:B*H])
	}
	if ctx.Pool != nil {
		ctx.Pool.Put(scr)
	}
	outs := ctx.OutPad(3, y)
	outs[1] = yh
	outs[2] = yc
	return outs, nil
}

// gruOp: ONNX GRU, layout 0, default activations (sigmoid, tanh), gate order
// zrh. Supports both linear_before_reset settings (torch exports 1).
type gruOp struct {
	n      NodeInfo
	hidden int
	dirs   []bool
	lbr    bool
}

func (o *gruOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 3 || in[0] == nil || in[1] == nil || in[2] == nil {
		return nil, o.n.Errorf("GRU: need X, W, R")
	}
	if len(in) > 4 && in[4] != nil {
		return nil, o.n.Errorf("GRU: sequence_lens is not supported")
	}
	x, w, r := in[0], in[1], in[2]
	xs := x.Shape()
	if len(xs) != 3 {
		return nil, o.n.Errorf("GRU: want X [T,B,I], got %v", xs)
	}
	T, B, I := xs[0], xs[1], xs[2]
	H := o.hidden
	D := len(o.dirs)
	if !w.Shape().Equal(tensor.Shape{D, 3 * H, I}) || !r.Shape().Equal(tensor.Shape{D, 3 * H, H}) {
		return nil, o.n.Errorf("GRU: W %v / R %v vs D=%d H=%d I=%d", w.Shape(), r.Shape(), D, H, I)
	}
	var bias []float32
	if len(in) > 3 && in[3] != nil {
		if in[3].Numel() != D*6*H {
			return nil, o.n.Errorf("GRU: B has %d elements, want %d", in[3].Numel(), D*6*H)
		}
		bias = in[3].F32()
	}
	var ih []float32
	if len(in) > 5 && in[5] != nil {
		ih = in[5].F32()
	}

	y := ctx.NewUninit(tensor.F32, T, D, B, H)
	yh := ctx.NewUninit(tensor.F32, D, B, H)
	xf, wf, rf := x.F32(), w.F32(), r.F32()
	yf, yhf := y.F32(), yh.F32()

	scr := ctx.NewUninit(tensor.F32, T*B*3*H+2*B*3*H+B*H)
	sf := scr.F32()
	G := sf[:T*B*3*H]
	gates := sf[T*B*3*H : T*B*3*H+B*3*H]
	rg := sf[T*B*3*H+B*3*H : T*B*3*H+2*B*3*H]
	hState := sf[T*B*3*H+2*B*3*H:]

	for d, rev := range o.dirs {
		wd := wf[d*3*H*I:]
		rd := rf[d*3*H*H:]
		gemm.SgemmT(false, true, T*B, 3*H, I, 1, xf, I, wd, I, 0, G, 3*H)
		var wb, rb []float32
		if bias != nil {
			wb = bias[d*6*H : d*6*H+3*H]
			rb = bias[d*6*H+3*H : d*6*H+6*H]
		}
		if ih != nil {
			copy(hState, ih[d*B*H:(d+1)*B*H])
		} else {
			clear(hState)
		}
		for step := 0; step < T; step++ {
			t := step
			if rev {
				t = T - 1 - step
			}
			copy(gates, G[t*B*3*H:(t+1)*B*3*H])
			// rg = h · Rᵀ (all three recurrent projections).
			gemm.SgemmT(false, true, B, 3*H, H, 1, hState, H, rd, H, 0, rg, 3*H)
			for b := 0; b < B; b++ {
				g := gates[b*3*H:]
				rr := rg[b*3*H:]
				hRow := hState[b*H:]
				gz, grt, gh := g[:H], g[H:2*H], g[2*H:3*H]
				rz, rrt, rh := rr[:H], rr[H:2*H], rr[2*H:3*H]
				for j := 0; j < H; j++ {
					gz[j] += rz[j]
					grt[j] += rrt[j]
				}
				if wb != nil {
					for j := 0; j < H; j++ {
						gz[j] += wb[j] + rb[j]
						grt[j] += wb[H+j] + rb[H+j]
					}
				}
				sigmoidInPlace(gz)
				sigmoidInPlace(grt)
				if o.lbr {
					// h~ = tanh(Wh·x + bWh + r ⊙ (Rh·h + bRh))
					for j := 0; j < H; j++ {
						t := rh[j]
						if rb != nil {
							t += rb[2*H+j]
						}
						gh[j] += grt[j] * t
						if wb != nil {
							gh[j] += wb[2*H+j]
						}
					}
				} else {
					// h~ = tanh(Wh·x + bWh + Rh·(r ⊙ h) + bRh): recompute the
					// recurrent term with the reset applied to h first.
					for j := 0; j < H; j++ {
						rh[j] = grt[j] * hRow[j] // reuse rh as (r ⊙ h)
					}
					// gh += (r ⊙ h) · Rhᵀ
					gemm.SgemmTSerial(false, true, 1, H, H, 1, rh[:H], H, rd[2*H*H:], H, 1, gh[:H], H)
					if wb != nil {
						for j := 0; j < H; j++ {
							gh[j] += wb[2*H+j] + rb[2*H+j]
						}
					}
				}
				tanhInPlace(gh[:H])
				for j := 0; j < H; j++ {
					hRow[j] = (1-gz[j])*gh[j] + gz[j]*hRow[j]
				}
				copy(yf[((t*D+d)*B+b)*H:], hRow[:H])
			}
		}
		copy(yhf[d*B*H:], hState[:B*H])
	}
	if ctx.Pool != nil {
		ctx.Pool.Put(scr)
	}
	outs := ctx.OutPad(2, y)
	outs[1] = yh
	return outs, nil
}

func rnnCommon(n NodeInfo) ([]bool, error) {
	if n.Attrs.Has("activations") {
		return nil, n.Errorf("custom activations are not supported")
	}
	if n.Attrs.Float("clip", 0) != 0 {
		return nil, n.Errorf("clip is not supported")
	}
	if n.Attrs.Int("layout", 0) != 0 {
		return nil, n.Errorf("layout=1 is not supported")
	}
	return rnnDirections(n)
}

func init() {
	Register("", "LSTM", 7, func(n NodeInfo) (Op, error) {
		dirs, err := rnnCommon(n)
		if err != nil {
			return nil, err
		}
		h := int(n.Attrs.Int("hidden_size", 0))
		if h <= 0 {
			return nil, n.Errorf("LSTM: hidden_size required")
		}
		if n.Attrs.Int("input_forget", 0) != 0 {
			return nil, n.Errorf("LSTM: input_forget is not supported")
		}
		return &lstmOp{n: n, hidden: h, dirs: dirs}, nil
	})
	Register("", "GRU", 7, func(n NodeInfo) (Op, error) {
		dirs, err := rnnCommon(n)
		if err != nil {
			return nil, err
		}
		h := int(n.Attrs.Int("hidden_size", 0))
		if h <= 0 {
			return nil, n.Errorf("GRU: hidden_size required")
		}
		return &gruOp{n: n, hidden: h, dirs: dirs, lbr: n.Attrs.Int("linear_before_reset", 0) != 0}, nil
	})
}
