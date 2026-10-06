package wan

import (
	"image"
	"image/color"
	"image/draw"
	"math"

	"github.com/giraffesyo/ingot/models/qwenimage"
	"github.com/giraffesyo/ingot/tensor"
)

// DefaultArea is the pixel area TI2V-5B was trained at (1280×704).
const DefaultArea = 1280 * 704

// SizeFor picks an output size for an image of w×h: the image's aspect
// ratio, about area pixels, sides multiples of 32 (one latent patch).
func SizeFor(w, h, area int) (int, int) {
	ratio := float64(w) / float64(h)
	fw := math.Sqrt(float64(area) * ratio)
	fh := fw / ratio
	return max(32, int(math.Round(fw/32))*32), max(32, int(math.Round(fh/32))*32)
}

// Fit scales img to cover w×h (Lanczos-3) and crops the centre — the
// original Wan 2.2 image-to-video preprocessing. An image already w×h is
// returned unchanged.
func Fit(img image.Image, w, h int) *image.NRGBA {
	b := img.Bounds()
	sw, sh := b.Dx(), b.Dy()
	// Crop the source to the target aspect first, then resample.
	cw, ch := sw, sh
	if sw*h > sh*w {
		cw = int(math.Round(float64(sh) * float64(w) / float64(h)))
	} else {
		ch = int(math.Round(float64(sw) * float64(h) / float64(w)))
	}
	x0, y0 := b.Min.X+(sw-cw)/2, b.Min.Y+(sh-ch)/2
	crop := image.NewNRGBA(image.Rect(0, 0, cw, ch))
	draw.Draw(crop, crop.Bounds(), img, image.Pt(x0, y0), draw.Src)
	return qwenimage.ResizeLanczos(crop, w, h)
}

// Pixels is VideoProcessor.preprocess for an image at its target size:
// [3, H, W] in [-1, 1] (x/255·2 − 1), alpha ignored.
func Pixels(img *image.NRGBA) *tensor.Tensor {
	W, H := img.Rect.Dx(), img.Rect.Dy()
	out := tensor.New(tensor.F32, 3, H, W)
	f := out.F32()
	for y := range H {
		for x := range W {
			c := img.NRGBAAt(img.Rect.Min.X+x, img.Rect.Min.Y+y)
			for ch, v := range [3]uint8{c.R, c.G, c.B} {
				f[(ch*H+y)*W+x] = float32(v)/255*2 - 1
			}
		}
	}
	return out
}

// FrameImage converts one decoded frame [3, H, W] in [-1, 1] to 8-bit RGB
// as the pipeline's postprocess does: (x/2 + 0.5) clamped, ×255, rounded.
func FrameImage(f []float32, H, W int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, W, H))
	q := func(v float32) uint8 {
		u := min(max(float64(v)/2+0.5, 0), 1)
		return uint8(math.Round(u * 255))
	}
	for y := range H {
		for x := range W {
			img.SetNRGBA(x, y, color.NRGBA{q(f[y*W+x]), q(f[(H+y)*W+x]), q(f[(2*H+y)*W+x]), 255})
		}
	}
	return img
}

// KeepRegion composites the source image back over a frame where mask is
// set: out = m·src + (1−m)·frame per channel, m = mask's gray level / 255.
// Where the mask is white the frame is pixel-identical to src (both at the
// output size); soft mask edges blend. This is how a region (a face, a
// body) stays exactly the input while the rest animates: the model
// conditions on the first frame only, so a pixel-exact hold is a
// post-process, not a sampling constraint.
func KeepRegion(frame, src *image.NRGBA, mask image.Image) {
	b := frame.Rect
	mb := mask.Bounds()
	for y := range b.Dy() {
		for x := range b.Dx() {
			mx := mb.Min.X + x*mb.Dx()/b.Dx()
			my := mb.Min.Y + y*mb.Dy()/b.Dy()
			g := color.GrayModel.Convert(mask.At(mx, my)).(color.Gray).Y
			if g == 0 {
				continue
			}
			s := src.NRGBAAt(src.Rect.Min.X+x, src.Rect.Min.Y+y)
			if g == 255 {
				frame.SetNRGBA(b.Min.X+x, b.Min.Y+y, s)
				continue
			}
			f := frame.NRGBAAt(b.Min.X+x, b.Min.Y+y)
			m := float64(g) / 255
			mix := func(a, c uint8) uint8 { return uint8(math.Round(m*float64(a) + (1-m)*float64(c))) }
			frame.SetNRGBA(b.Min.X+x, b.Min.Y+y, color.NRGBA{mix(s.R, f.R), mix(s.G, f.G), mix(s.B, f.B), 255})
		}
	}
}
