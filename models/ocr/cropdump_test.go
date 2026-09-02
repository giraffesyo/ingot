package ocr

import (
	"image"
	"image/color"
	"image/png"
	"os"
	"testing"
)

// TestDumpCrop writes the crop of the first detected box on img_006 under
// both sampling modes (OCR_DUMP_CROP=dir), for eyeballing.
func TestDumpCrop(t *testing.T) {
	dir := os.Getenv("OCR_DUMP_CROP")
	if dir == "" {
		t.Skip("set OCR_DUMP_CROP=dir")
	}
	d, err := NewDetector(corpusDir + "/det.onnx")
	if err != nil {
		t.Skip(err)
	}
	f, err := os.Open(corpusDir + "/corpus/img_006.png")
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	img, _, _ := image.Decode(f)
	boxes, _ := d.Detect(img)
	for bi, b := range boxes[:3] {
		W := cropWidth(b, 48)
		for _, mode := range []bool{true, false} {
			cropNearest = mode
			buf := make([]float32, 3*48*W)
			cropInto(buf, img, b, 48, W, W, cropUpMax)
			out := image.NewRGBA(image.Rect(0, 0, W, 48))
			for y := 0; y < 48; y++ {
				for x := 0; x < W; x++ {
					o := y*W + x
					out.Set(x, y, color.RGBA{u8(float64(buf[o]+1) / 2 * 255), u8(float64(buf[48*W+o]+1) / 2 * 255), u8(float64(buf[2*48*W+o]+1) / 2 * 255), 255})
				}
			}
			name := "bilinear"
			if mode {
				name = "nearest"
			}
			of, _ := os.Create(dir + "/crop" + string(rune('0'+bi)) + "_" + name + ".png")
			png.Encode(of, out)
			of.Close()
			t.Logf("box %d pts=%v W=%d", bi, b.Pts, W)
		}
	}
}
