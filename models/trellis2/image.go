package trellis2

import (
	"fmt"
	"image"
	"image/color"
	"math"

	"github.com/giraffesyo/ingot/tensor"
)

// rgba8 is an image as interleaved 8-bit samples, ch per pixel.
type rgba8 struct {
	pix  []uint8
	w, h int
	ch   int
}

// Preprocess prepares a photo the way the pipeline does: the subject —
// the pixels more than 80% opaque — is cropped to a centred square and
// composited on black. An image without transparency is taken as all
// subject; segment the background out first for anything but a clean
// cut-out (the reference runs a matting model here, which this package
// does not include). Images wider than 1024 pixels are reduced first.
// The result is square RGB.
func Preprocess(img image.Image) (*image.NRGBA, error) { return PreprocessMargin(img, 1) }

// PreprocessMargin is Preprocess with the square crop scaled about the
// subject's centre: 1 crops tight; Pixal3D's pipeline uses 1.1.
func PreprocessMargin(img image.Image, scale float64) (*image.NRGBA, error) {
	b := img.Bounds()
	src := rgba8{pix: make([]uint8, 4*b.Dx()*b.Dy()), w: b.Dx(), h: b.Dy(), ch: 4}
	for y := range src.h {
		for x := range src.w {
			c := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
			copy(src.pix[4*(y*src.w+x):], []uint8{c.R, c.G, c.B, c.A})
		}
	}
	if m := max(src.w, src.h); m > 1024 {
		// Resampled with colour premultiplied by alpha, so transparent
		// pixels do not bleed their colour into the subject's edge.
		s := 1024 / float64(m)
		for i := 0; i < len(src.pix); i += 4 {
			a := float64(src.pix[i+3]) / 255
			for c := range 3 {
				src.pix[i+c] = uint8(float64(src.pix[i+c])*a + 0.5)
			}
		}
		src = resizeLanczos(src, int(float64(src.w)*s), int(float64(src.h)*s))
		for i := 0; i < len(src.pix); i += 4 {
			if a := float64(src.pix[i+3]); a > 0 {
				for c := range 3 {
					src.pix[i+c] = uint8(min(255, float64(src.pix[i+c])*255/a+0.5))
				}
			}
		}
	}
	x0, y0, x1, y1 := src.w, src.h, -1, -1
	for y := range src.h {
		for x := range src.w {
			if float64(src.pix[4*(y*src.w+x)+3]) > 0.8*255 {
				x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
			}
		}
	}
	if x1 < 0 {
		return nil, fmt.Errorf("trellis2: image has no opaque subject (alpha above 80%% nowhere)")
	}
	cx, cy := float64(x0+x1)/2, float64(y0+y1)/2
	half := int(float64(max(x1-x0, y1-y0))*scale) / 2
	// The crop box rounds half to even, as the reference's does.
	left, top := int(math.RoundToEven(cx-float64(half))), int(math.RoundToEven(cy-float64(half)))
	side := int(math.RoundToEven(cx+float64(half))) - left
	if side <= 0 {
		return nil, fmt.Errorf("trellis2: subject is a single pixel")
	}
	out := image.NewNRGBA(image.Rect(0, 0, side, side))
	for y := range side {
		for x := range side {
			sx, sy := left+x, top+y
			o := out.Pix[4*(y*side+x):]
			o[3] = 255
			if sx < 0 || sy < 0 || sx >= src.w || sy >= src.h {
				continue
			}
			p := src.pix[4*(sy*src.w+sx):]
			a := float32(p[3]) / 255
			for c := range 3 {
				o[c] = uint8(float32(p[c]) / 255 * a * 255)
			}
		}
	}
	return out, nil
}

// ImageTensor resizes a preprocessed image to size×size (Lanczos) and
// normalises it for the encoder: [1, 3, size, size].
func ImageTensor(img *image.NRGBA, size int) *tensor.Tensor { return imageTensor(img, size, true) }

// imageTensor is ImageTensor, or with normalize false the plain [0, 1] RGB
// the feature upsampler is guided by.
func imageTensor(img *image.NRGBA, size int, normalize bool) *tensor.Tensor {
	b := img.Bounds()
	src := rgba8{pix: make([]uint8, 3*b.Dx()*b.Dy()), w: b.Dx(), h: b.Dy(), ch: 3}
	for y := range src.h {
		for x := range src.w {
			copy(src.pix[3*(y*src.w+x):], img.Pix[img.PixOffset(b.Min.X+x, b.Min.Y+y):][:3])
		}
	}
	if src.w != size || src.h != size {
		src = resizeLanczos(src, size, size)
	}
	t := tensor.New(tensor.F32, 1, 3, size, size)
	f := t.F32()
	for i := range size * size {
		for c := range 3 {
			v := float32(src.pix[3*i+c]) / 255
			if normalize {
				v = (v - ImageMean[c]) / ImageStd[c]
			}
			f[c*size*size+i] = v
		}
	}
	return t
}

// resizeLanczos resamples to w×h with a Lanczos-3 filter widened by the
// reduction factor (antialiased), horizontally then vertically with the
// intermediate rounded to 8 bits — the scheme of the reference's image
// library, so features match to rounding.
func resizeLanczos(src rgba8, w, h int) rgba8 {
	mid := rgba8{pix: make([]uint8, src.ch*w*src.h), w: w, h: src.h, ch: src.ch}
	resample1D(src.pix, mid.pix, src.w, w, src.h, src.ch, src.ch, src.ch*src.w, src.ch*w)
	out := rgba8{pix: make([]uint8, src.ch*w*h), w: w, h: h, ch: src.ch}
	// Vertical: each column is a line of stride ch·w.
	resample1D(mid.pix, out.pix, src.h, h, w, src.ch, src.ch*w, src.ch, src.ch)
	return out
}

// resample1D filters `lines` lines of n samples to m samples; a sample is
// ch channels, consecutive samples are step apart and lines are lineIn /
// lineOut apart in src / dst.
func resample1D(src, dst []uint8, n, m, lines, ch, step, lineIn, lineOut int) {
	scale := float64(n) / float64(m)
	fscale := max(scale, 1)
	support := 3 * fscale
	weights := make([]float64, int(math.Ceil(support))*2+2)
	for i := range m {
		center := (float64(i) + 0.5) * scale
		lo := max(0, int(center-support+0.5))
		hi := min(n, int(center+support+0.5))
		var sum float64
		for k := lo; k < hi; k++ {
			wk := lanczos3((float64(k) - center + 0.5) / fscale)
			weights[k-lo] = wk
			sum += wk
		}
		for l := range lines {
			for c := range ch {
				var acc float64
				for k := lo; k < hi; k++ {
					acc += weights[k-lo] * float64(src[l*lineIn+k*step+c])
				}
				dst[l*lineOut+i*step+c] = uint8(min(255, max(0, math.Floor(acc/sum+0.5))))
			}
		}
	}
}

func lanczos3(x float64) float64 {
	if x < 0 {
		x = -x
	}
	if x >= 3 {
		return 0
	}
	if x == 0 {
		return 1
	}
	px := math.Pi * x
	return 3 * math.Sin(px) * math.Sin(px/3) / (px * px)
}
