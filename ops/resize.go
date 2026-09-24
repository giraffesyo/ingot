package ops

import (
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/kernels/vek"
	"github.com/giraffesyo/ingot/tensor"
)

// resizeOp implements 2-D Resize over the trailing (H, W) dims of an NCHW
// tensor. Supports nearest and linear (bilinear) modes and the common
// coordinate-transformation modes. Cubic is not implemented.
type resizeOp struct {
	n       NodeInfo
	mode    string // "nearest" | "linear"
	coord   string // half_pixel | pytorch_half_pixel | align_corners | asymmetric
	nearest string // floor | ceil | round_prefer_floor | round_prefer_ceil
	scales  []float32
	sizes   []int64
}

// srcCoord maps an output index to a source coordinate.
func srcCoord(coord string, outIdx, outSize, inSize int, scale float32) float64 {
	o := float64(outIdx)
	switch coord {
	case "align_corners":
		if outSize == 1 {
			return 0
		}
		return o * float64(inSize-1) / float64(outSize-1)
	case "asymmetric":
		return o / float64(scale)
	case "pytorch_half_pixel":
		if outSize <= 1 {
			return 0
		}
		return (o+0.5)/float64(scale) - 0.5
	default: // half_pixel
		return (o+0.5)/float64(scale) - 0.5
	}
}

// ResizeTaps is a 2-D Resize's separable source mapping: output row i
// reads rows Y0[i] and Y1[i] blended by WY[i] (the upper row's weight),
// likewise for columns. Nearest mode has Y1 = Y0 and zero weights.
type ResizeTaps struct {
	OH, OW         int
	Y0, Y1, X0, X1 []int
	WY, WX         []float32
}

// PlanResize resolves a Resize node's output size and taps for its inputs
// (X [N,C,H,W], roi, scales, sizes) — the mapping the CPU op uses.
func PlanResize(a Attrs, in []*tensor.Tensor) (ResizeTaps, error) {
	o := &resizeOp{mode: a.String("mode", "nearest"), coord: a.String("coordinate_transformation_mode", "half_pixel"),
		nearest: a.String("nearest_mode", "round_prefer_floor")}
	return o.taps(in)
}

func (o *resizeOp) taps(in []*tensor.Tensor) (ResizeTaps, error) {
	var t ResizeTaps
	if len(in) < 1 || in[0] == nil || in[0].DType() != tensor.F32 {
		return t, fmt.Errorf("need f32 input")
	}
	xs := in[0].Shape()
	if len(xs) != 4 {
		return t, fmt.Errorf("only 4-D NCHW resize supported, got %v", xs)
	}
	// Resolve scales/sizes from inputs (opset 11/13/18: X, roi, scales, sizes).
	var scales []float32
	var sizes []int64
	if len(in) > 2 && in[2] != nil && in[2].Numel() > 0 {
		scales = in[2].F32()
	} else if o.scales != nil {
		scales = o.scales
	}
	if len(in) > 3 && in[3] != nil && in[3].Numel() > 0 {
		sizes = asI64(in[3])
	} else if o.sizes != nil {
		sizes = o.sizes
	}
	H, W := xs[2], xs[3]
	var sh, sw float32
	switch {
	case len(sizes) == 4:
		t.OH, t.OW = int(sizes[2]), int(sizes[3])
		sh, sw = float32(t.OH)/float32(H), float32(t.OW)/float32(W)
	case len(scales) == 4:
		sh, sw = scales[2], scales[3]
		t.OH, t.OW = int(math.Floor(float64(sh)*float64(H))), int(math.Floor(float64(sw)*float64(W)))
	default:
		return t, fmt.Errorf("need scales or sizes")
	}
	if t.OH <= 0 || t.OW <= 0 {
		return t, fmt.Errorf("non-positive output %dx%d", t.OH, t.OW)
	}
	t.Y0, t.Y1, t.WY = make([]int, t.OH), make([]int, t.OH), make([]float32, t.OH)
	t.X0, t.X1, t.WX = make([]int, t.OW), make([]int, t.OW), make([]float32, t.OW)
	switch o.mode {
	case "nearest":
		for i := range t.Y0 {
			t.Y0[i] = clampIdx(nearestIdx(o.nearest, srcCoord(o.coord, i, t.OH, H, sh)), H)
			t.Y1[i] = t.Y0[i]
		}
		for j := range t.X0 {
			t.X0[j] = clampIdx(nearestIdx(o.nearest, srcCoord(o.coord, j, t.OW, W, sw)), W)
			t.X1[j] = t.X0[j]
		}
	case "linear":
		for i := range t.Y0 {
			t.Y0[i], t.Y1[i], t.WY[i] = linTaps(srcCoord(o.coord, i, t.OH, H, sh), H)
		}
		for j := range t.X0 {
			t.X0[j], t.X1[j], t.WX[j] = linTaps(srcCoord(o.coord, j, t.OW, W, sw), W)
		}
	default:
		return t, fmt.Errorf("unsupported mode %q (only nearest, linear)", o.mode)
	}
	return t, nil
}

func (o *resizeOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	tp, err := o.taps(in)
	if err != nil {
		return nil, o.n.Errorf("%v", err)
	}
	x := in[0]
	xs := x.Shape()
	N, C, H, W := xs[0], xs[1], xs[2], xs[3]
	OH, OW := tp.OH, tp.OW
	out := ctx.NewUninit(tensor.F32, N, C, OH, OW)
	xf, of := x.F32(), out.F32()

	switch o.mode {
	case "nearest":
		ry, rx := tp.Y0, tp.X0
		// Integer 2× upsample (the FPN case): each output row is either a
		// self-interleave of a source row or a copy of the previous output
		// row — no per-element gather.
		dup2 := OH == 2*H && OW == 2*W
		for i := 0; dup2 && i < OH; i++ {
			dup2 = ry[i] == i>>1
		}
		for j := 0; dup2 && j < OW; j++ {
			dup2 = rx[j] == j>>1
		}
		par.For(N*C, 1, func(nc, _ int) {
			src := xf[nc*H*W : (nc+1)*H*W]
			dst := of[nc*OH*OW : (nc+1)*OH*OW]
			if dup2 {
				for ih := 0; ih < H; ih++ {
					srow := src[ih*W : (ih+1)*W]
					drow := dst[2*ih*OW : (2*ih+1)*OW]
					vek.Zip2(drow, srow, srow, 0)
					copy(dst[(2*ih+1)*OW:(2*ih+2)*OW], drow)
				}
				return
			}
			for i := 0; i < OH; i++ {
				srow := src[ry[i]*W:]
				drow := dst[i*OW : (i+1)*OW]
				for j := 0; j < OW; j++ {
					drow[j] = srow[rx[j]]
				}
			}
		})
	case "linear":
		y0, y1, wy, x0, x1, wx := tp.Y0, tp.Y1, tp.WY, tp.X0, tp.X1, tp.WX
		par.For(N*C, 1, func(nc, _ int) {
			src := xf[nc*H*W : (nc+1)*H*W]
			dst := of[nc*OH*OW : (nc+1)*OH*OW]
			for i := 0; i < OH; i++ {
				r0 := src[y0[i]*W:]
				r1 := src[y1[i]*W:]
				a := wy[i]
				drow := dst[i*OW : (i+1)*OW]
				for j := 0; j < OW; j++ {
					b := wx[j]
					top := r0[x0[j]]*(1-b) + r0[x1[j]]*b
					bot := r1[x0[j]]*(1-b) + r1[x1[j]]*b
					drow[j] = top*(1-a) + bot*a
				}
			}
		})
	}
	return ctx.Out(out), nil
}

func nearestIdx(mode string, c float64) int {
	switch mode {
	case "floor":
		return int(math.Floor(c))
	case "ceil":
		return int(math.Ceil(c))
	case "round_prefer_ceil":
		return int(math.Floor(c + 0.5))
	default: // round_prefer_floor
		return int(math.Ceil(c - 0.5))
	}
}

func clampIdx(i, n int) int {
	if i < 0 {
		return 0
	}
	if i >= n {
		return n - 1
	}
	return i
}

// linTaps returns the two source indices and the weight of the upper one.
func linTaps(c float64, n int) (i0, i1 int, w float32) {
	if c < 0 {
		c = 0
	}
	f := math.Floor(c)
	i0 = int(f)
	w = float32(c - f)
	i1 = i0 + 1
	if i0 >= n-1 {
		i0, i1, w = n-1, n-1, 0
	}
	if i1 >= n {
		i1 = n - 1
	}
	return
}

func init() {
	build := func(n NodeInfo) (Op, error) {
		return &resizeOp{
			n:       n,
			mode:    n.Attrs.String("mode", "nearest"),
			coord:   n.Attrs.String("coordinate_transformation_mode", "half_pixel"),
			nearest: n.Attrs.String("nearest_mode", "round_prefer_floor"),
		}, nil
	}
	Register("", "Resize", 10, build)
	Register("", "Resize", 11, build)
	Register("", "Resize", 13, build)
	Register("", "Resize", 18, build)
}
