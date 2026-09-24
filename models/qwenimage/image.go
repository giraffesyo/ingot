package qwenimage

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"

	"github.com/giraffesyo/ingot/tensor"
)

// Vision-language image geometry (Qwen3-VL): 16-px patches, merged 2×2 into
// one language-model token, each patch duplicated over a temporal pair.
const (
	visPatch    = 16
	visMerge    = 2
	visTemporal = 2
	visPatchDim = 3 * visTemporal * visPatch * visPatch // 1536
)

// ConditionSize is calculate_dimensions: the size a condition image is
// resized to — area ≈ resolution², the image's aspect ratio, sides rounded to
// multiples of 32 (one vision-language token and a 2×2 latent block).
func ConditionSize(resolution, w, h int) (int, int) {
	ratio := float64(w) / float64(h)
	fw := math.Sqrt(float64(resolution*resolution) * ratio)
	fh := fw / ratio
	return int(math.RoundToEven(fw/32)) * 32, int(math.RoundToEven(fh/32)) * 32
}

// toNRGBA converts img to NRGBA (a no-op copy for NRGBA input).
func toNRGBA(img image.Image) *image.NRGBA {
	if n, ok := img.(*image.NRGBA); ok {
		return n
	}
	b := img.Bounds()
	out := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(out, out.Bounds(), img, b.Min, draw.Src)
	return out
}

// VAEPixels is VaeImageProcessor.preprocess for an RGBA condition image
// already at its target size: [4, H, W] in [-1, 1] (x/255 · 2 − 1).
func VAEPixels(img image.Image) *tensor.Tensor {
	n := toNRGBA(img)
	W, H := n.Rect.Dx(), n.Rect.Dy()
	out := tensor.New(tensor.F32, 4, H, W)
	f := out.F32()
	for y := range H {
		for x := range W {
			c := n.NRGBAAt(x, y)
			for ch, v := range [4]uint8{c.R, c.G, c.B, c.A} {
				f[(ch*H+y)*W+x] = float32(v)/255*2 - 1
			}
		}
	}
	return out
}

// VisionPatches is the Qwen3-VL image processor for an image already at a
// multiple of 32 on each side: alpha composited over white (as the
// pipeline flattens RGBA for the vision encoder, PIL's paste rounding),
// scaled to [-1, 1] (mean = std = 0.5), cut into 16-px patches in 2×2
// merge-block order, each flattened as (channel, temporal copy, row, col):
// [gridH·gridW, 1536]. Returns the patch grid too.
func VisionPatches(img image.Image) (*tensor.Tensor, int, int, error) {
	n := toNRGBA(img)
	W, H := n.Rect.Dx(), n.Rect.Dy()
	if W%(visPatch*visMerge) != 0 || H%(visPatch*visMerge) != 0 {
		return nil, 0, 0, fmt.Errorf("qwenimage: vision image %dx%d not a multiple of %d", W, H, visPatch*visMerge)
	}
	gh, gw := H/visPatch, W/visPatch
	rgb := make([]float32, 3*H*W)
	for y := range H {
		for x := range W {
			c := n.NRGBAAt(x, y)
			for ch, v := range [3]uint8{c.R, c.G, c.B} {
				rgb[(ch*H+y)*W+x] = (float32(blendWhite(v, c.A))*(1.0/255) - 0.5) / 0.5
			}
		}
	}
	out := tensor.New(tensor.F32, gh*gw, visPatchDim)
	f := out.F32()
	p := 0
	for br := 0; br < gh/visMerge; br++ {
		for bc := 0; bc < gw/visMerge; bc++ {
			for ir := range visMerge {
				for ic := range visMerge {
					py, px := (br*visMerge+ir)*visPatch, (bc*visMerge+ic)*visPatch
					row := f[p*visPatchDim : (p+1)*visPatchDim]
					i := 0
					for ch := range 3 {
						for range visTemporal {
							for yy := range visPatch {
								copy(row[i:i+visPatch], rgb[(ch*H+py+yy)*W+px:(ch*H+py+yy)*W+px+visPatch])
								i += visPatch
							}
						}
					}
					p++
				}
			}
		}
	}
	return out, gh, gw, nil
}

// blendWhite is PIL's paste-with-mask over white for one channel:
// (v·a + 255·(255−a)) / 255 with PIL's rounding.
func blendWhite(v, a uint8) uint8 {
	tmp := uint32(v)*uint32(a) + 255*uint32(255-a) + 128
	return uint8((tmp + tmp>>8) >> 8)
}

// ResizeLanczos resamples img to w×h with a Lanczos-3 filter (PIL's
// LANCZOS), per channel on straight (non-premultiplied) RGBA.
func ResizeLanczos(img image.Image, w, h int) *image.NRGBA {
	src := toNRGBA(img)
	sw, sh := src.Rect.Dx(), src.Rect.Dy()
	if sw == w && sh == h {
		return src
	}
	// Horizontal then vertical pass (separable), float accumulation.
	tmp := make([]float64, 4*sh*w)
	resample1D(sw, w, func(x int, taps []int, wt []float64) {
		for y := range sh {
			for ch := range 4 {
				var acc float64
				for i, t := range taps {
					acc += wt[i] * float64(src.Pix[y*src.Stride+4*t+ch])
				}
				tmp[(y*w+x)*4+ch] = acc
			}
		}
	})
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	resample1D(sh, h, func(y int, taps []int, wt []float64) {
		for x := range w {
			var c [4]float64
			for i, t := range taps {
				for ch := range 4 {
					c[ch] += wt[i] * tmp[(t*w+x)*4+ch]
				}
			}
			out.SetNRGBA(x, y, color.NRGBA{clamp8(c[0]), clamp8(c[1]), clamp8(c[2]), clamp8(c[3])})
		}
	})
	return out
}

func clamp8(v float64) uint8 { return uint8(math.Max(0, math.Min(255, math.Round(v)))) }

// resample1D computes, for each output index, the Lanczos-3 source taps and
// normalised weights (PIL's support scaling when downsampling).
func resample1D(in, out int, fn func(o int, taps []int, wt []float64)) {
	scale := float64(in) / float64(out)
	support := 3 * math.Max(scale, 1)
	fscale := math.Max(scale, 1)
	for o := range out {
		center := (float64(o) + 0.5) * scale
		lo := int(math.Max(0, math.Floor(center-support+0.5)))
		hi := int(math.Min(float64(in), math.Floor(center+support+0.5)))
		var taps []int
		var wt []float64
		var sum float64
		for t := lo; t < hi; t++ {
			x := (float64(t) - center + 0.5) / fscale
			v := lanczos3(x)
			taps, wt = append(taps, t), append(wt, v)
			sum += v
		}
		for i := range wt {
			wt[i] /= sum
		}
		fn(o, taps, wt)
	}
}

func lanczos3(x float64) float64 {
	if x == 0 {
		return 1
	}
	if x <= -3 || x >= 3 {
		return 0
	}
	px := math.Pi * x
	return 3 * math.Sin(px) * math.Sin(px/3) / (px * px)
}
