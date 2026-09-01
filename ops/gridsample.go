package ops

import (
	"math"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/tensor"
)

// gridSampleOp: ONNX GridSample (opset 16+), 4-D only. X [N,C,H,W], grid
// [N,Ho,Wo,2] with (x,y) in [-1,1]. Modes: bilinear|nearest; padding:
// zeros|border|reflection; align_corners 0|1. Matches torch grid_sample
// semantics (which the ONNX spec adopts).
type gridSampleOp struct {
	n       NodeInfo
	mode    string
	padding string
	align   bool
}

func (o *gridSampleOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 2 || in[0] == nil || in[1] == nil {
		return nil, o.n.Errorf("GridSample: need X and grid")
	}
	x, grid := in[0], in[1]
	xs, gs := x.Shape(), grid.Shape()
	if len(xs) != 4 || len(gs) != 4 || gs[3] != 2 || gs[0] != xs[0] {
		return nil, o.n.Errorf("GridSample: shapes X=%v grid=%v", xs, gs)
	}
	if x.DType() != tensor.F32 || grid.DType() != tensor.F32 {
		return nil, o.n.Errorf("GridSample: want f32")
	}
	N, C, H, W := xs[0], xs[1], xs[2], xs[3]
	Ho, Wo := gs[1], gs[2]
	out := ctx.NewUninit(tensor.F32, N, C, Ho, Wo)
	xf, gf, of := x.F32(), grid.F32(), out.F32()

	// Unnormalise a [-1,1] coordinate to pixel space.
	unnorm := func(v float32, size int) float64 {
		if o.align {
			return (float64(v) + 1) / 2 * float64(size-1)
		}
		return ((float64(v)+1)*float64(size) - 1) / 2
	}
	// reflect coordinates into range (torch reflection semantics).
	reflect := func(v float64, lo, hi float64) float64 {
		span := hi - lo
		if span <= 0 {
			return lo
		}
		v = math.Abs(v - lo)
		v = math.Mod(v, 2*span)
		if v > span {
			v = 2*span - v
		}
		return v + lo
	}
	resolve := func(v float64, size int) float64 {
		switch o.padding {
		case "border":
			return math.Min(math.Max(v, 0), float64(size-1))
		case "reflection":
			if o.align {
				return reflect(v, 0, float64(size-1))
			}
			v = reflect(v, -0.5, float64(size)-0.5)
			return math.Min(math.Max(v, 0), float64(size-1))
		}
		return v // zeros: out-of-range handled at fetch
	}
	fetch := func(plane []float32, ix, iy int) float32 {
		if ix < 0 || ix >= W || iy < 0 || iy >= H {
			return 0 // zeros padding (border/reflection never land here)
		}
		return plane[iy*W+ix]
	}
	par.For(N*Ho, max(1, unaryChunk/max(Wo*C, 1)), func(nh, _ int) {
		n, oy := nh/Ho, nh%Ho
		for ox := 0; ox < Wo; ox++ {
			g := gf[((n*Ho+oy)*Wo+ox)*2:]
			px := resolve(unnorm(g[0], W), W)
			py := resolve(unnorm(g[1], H), H)
			switch o.mode {
			case "nearest":
				ix, iy := int(math.RoundToEven(px)), int(math.RoundToEven(py))
				for c := 0; c < C; c++ {
					plane := xf[(n*C+c)*H*W:]
					of[((n*C+c)*Ho+oy)*Wo+ox] = fetch(plane, ix, iy)
				}
			default: // bilinear
				x0, y0 := math.Floor(px), math.Floor(py)
				dx, dy := float32(px-x0), float32(py-y0)
				ix, iy := int(x0), int(y0)
				for c := 0; c < C; c++ {
					plane := xf[(n*C+c)*H*W:]
					v00 := fetch(plane, ix, iy)
					v01 := fetch(plane, ix+1, iy)
					v10 := fetch(plane, ix, iy+1)
					v11 := fetch(plane, ix+1, iy+1)
					top := v00 + (v01-v00)*dx
					bot := v10 + (v11-v10)*dx
					of[((n*C+c)*Ho+oy)*Wo+ox] = top + (bot-top)*dy
				}
			}
		}
	})
	return ctx.Out(out), nil
}

func init() {
	Register("", "GridSample", 16, func(n NodeInfo) (Op, error) {
		o := &gridSampleOp{
			n:       n,
			mode:    n.Attrs.String("mode", "bilinear"),
			padding: n.Attrs.String("padding_mode", "zeros"),
			align:   n.Attrs.Int("align_corners", 0) != 0,
		}
		switch o.mode {
		case "bilinear", "nearest", "linear":
			if o.mode == "linear" {
				o.mode = "bilinear" // opset 20 renamed the attr value
			}
		default:
			return nil, n.Errorf("GridSample: unsupported mode %q", o.mode)
		}
		switch o.padding {
		case "zeros", "border", "reflection":
		default:
			return nil, n.Errorf("GridSample: unsupported padding_mode %q", o.padding)
		}
		return o, nil
	})
}
