package wan

import (
	"image"
	"image/color"
	"testing"
)

func TestSizeFor(t *testing.T) {
	for _, c := range []struct{ w, h, area, ww, wh int }{
		{1280, 704, DefaultArea, 1280, 704},
		{1920, 1080, DefaultArea, 1280, 704},
		{1024, 1024, DefaultArea, 960, 960},
		{768, 1344, DefaultArea, 704, 1248},
		{832, 480, 832 * 480, 832, 480},
	} {
		if w, h := SizeFor(c.w, c.h, c.area); w != c.ww || h != c.wh {
			t.Errorf("SizeFor(%d, %d) = %dx%d, want %dx%d", c.w, c.h, w, h, c.ww, c.wh)
		}
	}
}

func TestKeepRegion(t *testing.T) {
	const W, H = 8, 6
	src, frame := image.NewNRGBA(image.Rect(0, 0, W, H)), image.NewNRGBA(image.Rect(0, 0, W, H))
	for y := range H {
		for x := range W {
			src.SetNRGBA(x, y, color.NRGBA{uint8(10 * x), uint8(20 * y), 7, 255})
			frame.SetNRGBA(x, y, color.NRGBA{200, 100, 50, 255})
		}
	}
	// Mask at half resolution: left half kept, one soft pixel column.
	mask := image.NewGray(image.Rect(0, 0, W/2, H/2))
	for y := range H / 2 {
		mask.SetGray(0, y, color.Gray{255})
		mask.SetGray(1, y, color.Gray{255})
		mask.SetGray(2, y, color.Gray{128})
	}
	KeepRegion(frame, src, mask)
	for y := range H {
		for x := range W {
			got := frame.NRGBAAt(x, y)
			switch {
			case x < 4:
				if got != src.NRGBAAt(x, y) {
					t.Fatalf("(%d,%d) kept %v, want source %v", x, y, got, src.NRGBAAt(x, y))
				}
			case x < 6:
				s := src.NRGBAAt(x, y)
				want := uint8((128*int(s.R) + 127*200 + 127) / 255)
				if d := int(got.R) - int(want); d < -1 || d > 1 {
					t.Fatalf("(%d,%d) blend R %d, want ~%d", x, y, got.R, want)
				}
			default:
				if got != (color.NRGBA{200, 100, 50, 255}) {
					t.Fatalf("(%d,%d) changed to %v", x, y, got)
				}
			}
		}
	}
}

func TestFitCrops(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 300, 100))
	for x := range 300 {
		for y := range 100 {
			c := uint8(0)
			if x >= 100 && x < 200 {
				c = 255
			}
			img.SetNRGBA(x, y, color.NRGBA{c, c, c, 255})
		}
	}
	out := Fit(img, 64, 64) // the centre square, all white
	for y := 8; y < 56; y++ {
		for x := 8; x < 56; x++ {
			if v := out.NRGBAAt(x, y).R; v < 250 {
				t.Fatalf("(%d,%d) = %d: crop not centred", x, y, v)
			}
		}
	}
}
