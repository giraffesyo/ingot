package ocr

import (
	"github.com/giraffesyo/ingot/kernels/par"
	"image"
	"math"

	"github.com/giraffesyo/ingot/tensor"
)

// DetNormMean/Std are the PP-OCR detection normalisation constants (ImageNet,
// in 0-255 scale, RGB).
var (
	detMean = [3]float32{0.485 * 255, 0.456 * 255, 0.406 * 255}
	detStd  = [3]float32{0.229 * 255, 0.224 * 255, 0.225 * 255}
)

// preprocessDet resizes img so its longest side is <= limit and both sides are
// multiples of 32 (DBNet requirement), normalises to an NCHW f32 tensor, and
// returns the tensor plus the width/height scale factors (resized/original) to
// map detections back to original coordinates.
func preprocessDet(img image.Image, limit int) (*tensor.Tensor, float64, float64) {
	b := img.Bounds()
	W, H := b.Dx(), b.Dy()
	scale := math.Min(float64(limit)/float64(max(W, H)), 1.0)
	nw := int(math.Max(32, math.Round(float64(W)*scale/32)*32))
	nh := int(math.Max(32, math.Round(float64(H)*scale/32)*32))
	rgb := resizeBilinear(img, nw, nh)
	t := tensor.New(tensor.F32, 1, 3, nh, nw)
	f := t.F32()
	plane := nh * nw
	for y := 0; y < nh; y++ {
		for x := 0; x < nw; x++ {
			i := (y*nw + x) * 3
			r, g, bl := rgb[i], rgb[i+1], rgb[i+2]
			o := y*nw + x
			f[o] = (float32(r) - detMean[0]) / detStd[0]
			f[plane+o] = (float32(g) - detMean[1]) / detStd[1]
			f[2*plane+o] = (float32(bl) - detMean[2]) / detStd[2]
		}
	}
	return t, float64(nw) / float64(W), float64(nh) / float64(H)
}

// toRGB returns img as a packed RGB (3 bytes/pixel) buffer plus its width and
// height. *image.RGBA / *image.NRGBA are read straight from Pix; other image
// types go through image.At once (not once per resampled tap).
func toRGB(img image.Image) ([]uint8, int, int) {
	b := img.Bounds()
	W, H := b.Dx(), b.Dy()
	out := make([]uint8, W*H*3)
	var pix []uint8
	var stride int
	switch im := img.(type) {
	case *image.RGBA:
		pix, stride = im.Pix, im.Stride
	case *image.NRGBA:
		pix, stride = im.Pix, im.Stride
	}
	if pix != nil {
		for y := 0; y < H; y++ {
			row := pix[y*stride : y*stride+W*4]
			o := out[y*W*3 : (y+1)*W*3]
			for x := 0; x < W; x++ {
				o[x*3], o[x*3+1], o[x*3+2] = row[x*4], row[x*4+1], row[x*4+2]
			}
		}
		return out, W, H
	}
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			o := (y*W + x) * 3
			out[o], out[o+1], out[o+2] = uint8(r>>8), uint8(g>>8), uint8(bl>>8)
		}
	}
	return out, W, H
}

// resizeBilinear returns a packed RGB (3 bytes/pixel) buffer of size nw*nh,
// bilinearly resampled from img. align_corners=false convention. Rows are
// resampled in parallel from a packed copy of the source.
func resizeBilinear(img image.Image, nw, nh int) []uint8 {
	src, W, H := toRGB(img)
	out := make([]uint8, nw*nh*3)
	sx := float64(W) / float64(nw)
	sy := float64(H) / float64(nh)
	// Precompute x taps.
	xs0 := make([]int, nw)
	xs1 := make([]int, nw)
	wxs := make([]float32, nw)
	for ox := 0; ox < nw; ox++ {
		fx := (float64(ox)+0.5)*sx - 0.5
		x0 := int(math.Floor(fx))
		wxs[ox] = float32(fx - float64(x0))
		xs0[ox] = clampI(x0, 0, W-1) * 3
		xs1[ox] = clampI(x0+1, 0, W-1) * 3
	}
	par.For(nh, max(1, 4096/max(nw, 1)), func(oy, _ int) {
		fy := (float64(oy)+0.5)*sy - 0.5
		y0 := int(math.Floor(fy))
		wy := float32(fy - float64(y0))
		r0 := src[clampI(y0, 0, H-1)*W*3:]
		r1 := src[clampI(y0+1, 0, H-1)*W*3:]
		dst := out[oy*nw*3 : (oy+1)*nw*3]
		for ox := 0; ox < nw; ox++ {
			x0, x1, wx := xs0[ox], xs1[ox], wxs[ox]
			for c := 0; c < 3; c++ {
				top := float32(r0[x0+c])*(1-wx) + float32(r0[x1+c])*wx
				bot := float32(r1[x0+c])*(1-wx) + float32(r1[x1+c])*wx
				dst[ox*3+c] = u8(float64(top*(1-wy) + bot*wy))
			}
		}
	})
	return out
}

func clampI(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func u8(v float64) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v + 0.5)
}
