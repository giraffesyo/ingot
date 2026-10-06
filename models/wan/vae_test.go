package wan

import (
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

func testVAEEncoder(t *testing.T, dir, ref string, tol float64) {
	c := loadRef(t, ref)
	cfg, err := LoadVAEConfig(filepath.Join(dir, "vae"))
	if err != nil {
		t.Fatal(err)
	}
	set := openSet(t, filepath.Join(dir, "vae"))
	img := c.tensor(t, "image")
	s := img.Shape()
	for _, dev := range devices() {
		t.Run(dev, func(t *testing.T) {
			g, err := BuildVAEEncoder(cfg, set, s[1], s[2])
			r := compile(t, g, err, dev)
			out, err := r.Run(map[string]*tensor.Tensor{"x": PatchifyImage(img)})
			if err != nil {
				t.Fatal(err)
			}
			compare(t, "mean", out["z"].F32(), c.tensor(t, "mean").F32(), tol)
		})
	}
}

func testVAEDecoder(t *testing.T, dir, ref string, tol float64) {
	c := loadRef(t, ref)
	cfg, err := LoadVAEConfig(filepath.Join(dir, "vae"))
	if err != nil {
		t.Fatal(err)
	}
	set := openSet(t, filepath.Join(dir, "vae"))
	z := c.tensor(t, "z")
	s := z.Shape()
	want := c.tensor(t, "video") // [3, F, H, W]
	ws := want.Shape()
	for _, dev := range devices() {
		t.Run(dev, func(t *testing.T) {
			d, err := BuildVAEDecoder(cfg, set, s[2], s[3])
			if err != nil {
				t.Fatal(err)
			}
			dc, err := NewDecoder(d, dev)
			if err != nil {
				t.Skip(err)
			}
			got, err := dc.Decode(z, nil)
			if err != nil {
				t.Fatal(err)
			}
			// [F, 3, H, W] → the reference's [3, F, H, W].
			gs := got.Shape()
			if gs[0] != ws[1] || gs[2] != ws[2] || gs[3] != ws[3] {
				t.Fatalf("decoded %v, want %v", gs, ws)
			}
			hw := gs[2] * gs[3]
			perm := make([]float32, got.Numel())
			for f := range gs[0] {
				for ch := range 3 {
					copy(perm[(ch*gs[0]+f)*hw:(ch*gs[0]+f+1)*hw], got.F32()[(f*3+ch)*hw:(f*3+ch+1)*hw])
				}
			}
			compare(t, "video", perm, want.F32(), tol)
		})
	}
}

func TestVAEEncoderTiny(t *testing.T) { testVAEEncoder(t, tinyDir(t), "vae_enc", 1e-5) }
func TestVAEDecoderTiny(t *testing.T) { testVAEDecoder(t, tinyDir(t), "vae_dec", 1e-5) }
func TestVAEEncoderReal(t *testing.T) { testVAEEncoder(t, realDir(t), "real_vae_enc", 1e-4) }
func TestVAEDecoderReal(t *testing.T) { testVAEDecoder(t, realDir(t), "real_vae_dec", 1e-4) }
