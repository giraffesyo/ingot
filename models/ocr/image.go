package ocr

import (
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

// resizeBilinear returns a packed RGB (3 bytes/pixel) buffer of size nw*nh,
// bilinearly resampled from img. align_corners=false convention.
func resizeBilinear(img image.Image, nw, nh int) []uint8 {
	b := img.Bounds()
	W, H := b.Dx(), b.Dy()
	out := make([]uint8, nw*nh*3)
	sx := float64(W) / float64(nw)
	sy := float64(H) / float64(nh)
	at := func(x, y int) (float64, float64, float64) {
		r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
		return float64(r >> 8), float64(g >> 8), float64(bl >> 8)
	}
	for oy := 0; oy < nh; oy++ {
		fy := (float64(oy)+0.5)*sy - 0.5
		y0 := int(math.Floor(fy))
		wy := fy - float64(y0)
		y1 := y0 + 1
		y0 = clampI(y0, 0, H-1)
		y1 = clampI(y1, 0, H-1)
		for ox := 0; ox < nw; ox++ {
			fx := (float64(ox)+0.5)*sx - 0.5
			x0 := int(math.Floor(fx))
			wx := fx - float64(x0)
			x1 := x0 + 1
			x0 = clampI(x0, 0, W-1)
			x1 = clampI(x1, 0, W-1)
			r00, g00, b00 := at(x0, y0)
			r10, g10, b10 := at(x1, y0)
			r01, g01, b01 := at(x0, y1)
			r11, g11, b11 := at(x1, y1)
			lerp := func(a, b, c, d float64) float64 {
				top := a*(1-wx) + b*wx
				bot := c*(1-wx) + d*wx
				return top*(1-wy) + bot*wy
			}
			o := (oy*nw + ox) * 3
			out[o] = u8(lerp(r00, r10, r01, r11))
			out[o+1] = u8(lerp(g00, g10, g01, g11))
			out[o+2] = u8(lerp(b00, b10, b01, b11))
		}
	}
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
