package ops

import (
	"math"

	"github.com/giraffesyo/ocr/kernels/par"
	"github.com/giraffesyo/ocr/tensor"
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

func (o *resizeOp) Run(ctx *Ctx, in []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(in) < 1 || in[0] == nil || in[0].DType() != tensor.F32 {
		return nil, o.n.Errorf("need f32 input")
	}
	x := in[0]
	xs := x.Shape()
	if len(xs) != 4 {
		return nil, o.n.Errorf("only 4-D NCHW resize supported, got %v", xs)
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
	N, C, H, W := xs[0], xs[1], xs[2], xs[3]
	var OH, OW int
	var sh, sw float32
	switch {
	case len(sizes) == 4:
		OH, OW = int(sizes[2]), int(sizes[3])
		sh, sw = float32(OH)/float32(H), float32(OW)/float32(W)
	case len(scales) == 4:
		sh, sw = scales[2], scales[3]
		OH, OW = int(math.Floor(float64(sh)*float64(H))), int(math.Floor(float64(sw)*float64(W)))
	default:
		return nil, o.n.Errorf("need scales or sizes")
	}
	if OH <= 0 || OW <= 0 {
		return nil, o.n.Errorf("non-positive output %dx%d", OH, OW)
	}
	out := ctx.NewUninit(tensor.F32, N, C, OH, OW)
	xf, of := x.F32(), out.F32()
	coord := o.coord

	// Precompute per-output-row/col source mapping.
	switch o.mode {
	case "nearest":
		ry := make([]int, OH)
		rx := make([]int, OW)
		for i := range ry {
			ry[i] = clampIdx(nearestIdx(o.nearest, srcCoord(coord, i, OH, H, sh)), H)
		}
		for j := range rx {
			rx[j] = clampIdx(nearestIdx(o.nearest, srcCoord(coord, j, OW, W, sw)), W)
		}
		par.For(N*C, 1, func(nc, _ int) {
			src := xf[nc*H*W : (nc+1)*H*W]
			dst := of[nc*OH*OW : (nc+1)*OH*OW]
			for i := 0; i < OH; i++ {
				srow := src[ry[i]*W:]
				drow := dst[i*OW : (i+1)*OW]
				for j := 0; j < OW; j++ {
					drow[j] = srow[rx[j]]
				}
			}
		})
	case "linear":
		y0 := make([]int, OH)
		y1 := make([]int, OH)
		wy := make([]float32, OH)
		for i := range y0 {
			c := srcCoord(coord, i, OH, H, sh)
			y0[i], y1[i], wy[i] = linTaps(c, H)
		}
		x0 := make([]int, OW)
		x1 := make([]int, OW)
		wx := make([]float32, OW)
		for j := range x0 {
			c := srcCoord(coord, j, OW, W, sw)
			x0[j], x1[j], wx[j] = linTaps(c, W)
		}
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
	default:
		return nil, o.n.Errorf("unsupported mode %q (only nearest, linear)", o.mode)
	}
	return []*tensor.Tensor{out}, nil
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
