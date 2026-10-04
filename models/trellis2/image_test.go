package trellis2

import (
	"image"
	"testing"
)

// TestPreprocessParity: the subject crop of an example cut-out matches the
// reference pipeline's exactly, and the Lanczos resize the encoder's image
// library's to one 8-bit step.
func TestPreprocessParity(t *testing.T) {
	ref := loadRef(t, "preprocess")
	rgba := ref.tensor(t, "rgba")
	h, w := rgba.Shape()[0], rgba.Shape()[1]
	img := &image.NRGBA{Pix: rgba.U8(), Stride: 4 * w, Rect: image.Rect(0, 0, w, h)}
	got, err := Preprocess(img)
	if err != nil {
		t.Fatal(err)
	}
	want := ref.tensor(t, "subject")
	side := want.Shape()[0]
	if got.Bounds().Dx() != side || got.Bounds().Dy() != side {
		t.Fatalf("subject is %v, reference %d×%d", got.Bounds(), side, side)
	}
	src := rgba8{pix: make([]uint8, 3*side*side), w: side, h: side, ch: 3}
	for i := range side * side {
		copy(src.pix[3*i:], got.Pix[4*i:4*i+3])
	}
	for i, v := range want.U8() {
		if src.pix[i] != v {
			t.Fatalf("subject sample %d = %d, reference %d", i, src.pix[i], v)
		}
	}
	size := int(ref.Meta["size"].(float64))
	res := resizeLanczos(src, size, size)
	worst, off := 0, 0
	for i, v := range ref.tensor(t, "resized").U8() {
		d := int(res.pix[i]) - int(v)
		if d < 0 {
			d = -d
		}
		if d > 0 {
			off++
		}
		worst = max(worst, d)
	}
	t.Logf("resize %d→%d: %d of %d samples differ, by at most %d", side, size, off, len(res.pix), worst)
	if worst > 1 {
		t.Fatalf("resized image differs from the reference by %d steps", worst)
	}
}

// TestPreprocessOpaque: an image without transparency is used whole.
func TestPreprocessOpaque(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 9, 5))
	for i := range img.Pix {
		img.Pix[i] = 200
		if i%4 == 3 {
			img.Pix[i] = 255
		}
	}
	got, err := Preprocess(img)
	if err != nil {
		t.Fatal(err)
	}
	if got.Bounds().Dx() != got.Bounds().Dy() || got.Bounds().Dx() != 8 {
		t.Fatalf("square crop of a 9×5 frame is %v, want 8×8", got.Bounds())
	}
	if _, err := Preprocess(image.NewNRGBA(image.Rect(0, 0, 4, 4))); err == nil {
		t.Fatal("fully transparent image accepted")
	}
}
