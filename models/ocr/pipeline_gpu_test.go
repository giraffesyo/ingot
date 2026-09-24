package ocr

import (
	"fmt"
	"image"
	"image/draw"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/kernels/metal"
)

// corpusImages loads testdata's OCR corpus.
func corpusImages(tb testing.TB) []image.Image {
	tb.Helper()
	paths, _ := filepath.Glob(detDir + "/corpus/*.png")
	if len(paths) == 0 {
		tb.Skip("OCR corpus not present")
	}
	var imgs []image.Image
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			tb.Fatal(err)
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			tb.Fatal(err)
		}
		imgs = append(imgs, img)
	}
	return imgs
}

// tiled pastes the corpus into one w×h page, row by row: a stand-in for a
// full-page image at the detector's size limit.
func tiled(imgs []image.Image, w, h int) image.Image {
	page := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(page, page.Bounds(), image.White, image.Point{}, draw.Src)
	x, y, rowH := 0, 0, 0
	for _, im := range imgs {
		b := im.Bounds()
		if x+b.Dx() > w {
			x, y, rowH = 0, y+rowH, 0
		}
		if y+b.Dy() > h {
			break
		}
		draw.Draw(page, image.Rect(x, y, x+b.Dx(), y+b.Dy()), im, b.Min, draw.Src)
		x += b.Dx()
		rowH = max(rowH, b.Dy())
	}
	return page
}

func pipelineOn(tb testing.TB, detDev, recDev string) *Pipeline {
	tb.Helper()
	if (detDev == "gpu" || recDev == "gpu") && !metal.Available() {
		tb.Skip("no GPU backend")
	}
	if _, err := os.Stat(detDir + "/det.onnx"); err != nil {
		tb.Skip("PP-OCR models not present")
	}
	p, err := NewPipelineOn(detDir+"/det.onnx", detDir+"/rec.onnx", detDir+"/rec_dict.txt", detDev, recDev)
	if err != nil {
		tb.Fatal(err)
	}
	return p
}

// TestPipelineGPU: the all-GPU pipeline reads the corpus (and the tiled
// page) like the CPU one: same boxes, same text.
func TestPipelineGPU(t *testing.T) {
	cpu, gpu := pipelineOn(t, "cpu", "cpu"), pipelineOn(t, "gpu", "gpu")
	imgs := corpusImages(t)
	imgs = append(imgs, tiled(imgs, 1920, 1080))
	var diffs int
	var maxConf float64
	for i, img := range imgs {
		want, err := cpu.Run(img)
		if err != nil {
			t.Fatal(err)
		}
		got, err := gpu.Run(img)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(want) {
			t.Errorf("image %d: %d results, cpu %d", i, len(got), len(want))
			diffs++
			continue
		}
		for k := range want {
			if got[k].Text != want[k].Text {
				t.Errorf("image %d box %d: %q, cpu %q", i, k, got[k].Text, want[k].Text)
				diffs++
			}
			maxConf = math.Max(maxConf, math.Abs(got[k].Conf-want[k].Conf))
		}
	}
	t.Logf("%d images, %d differences, max confidence delta %.2g", len(imgs), diffs, maxConf)
}

// BenchmarkPipelineDevice: end-to-end OCR (detect + recognise) per device
// placement, over the whole corpus and over one tiled 1920×1080 page.
func BenchmarkPipelineDevice(b *testing.B) {
	imgs := corpusImages(b)
	page := tiled(imgs, 1920, 1080)
	for _, dev := range [][2]string{{"cpu", "cpu"}, {"gpu", "cpu"}, {"gpu", "gpu"}} {
		p := pipelineOn(b, dev[0], dev[1])
		name := fmt.Sprintf("det=%s,rec=%s", dev[0], dev[1])
		b.Run("corpus/"+name, func(b *testing.B) {
			for b.Loop() {
				for _, img := range imgs {
					if _, err := p.Run(img); err != nil {
						b.Fatal(err)
					}
				}
			}
		})
		b.Run("page1920/"+name, func(b *testing.B) {
			for b.Loop() {
				if _, err := p.Run(page); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
