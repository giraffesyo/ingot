package qwenimage

import (
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// TestConditionPreprocess: the Go vision patches and VAE pixels of the
// reference condition image equal the transformers/diffusers tensors.
func TestConditionPreprocess(t *testing.T) {
	ref := loadRef(t, "edit")
	f, err := os.Open(filepath.Join(refDir, "edit_cond.png"))
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	if w, h := ConditionSize(256, 256, 256); w != 256 || h != 256 {
		t.Fatalf("ConditionSize(256, 256x256) = %dx%d", w, h)
	}
	pix, gh, gw, err := VisionPatches(img)
	if err != nil {
		t.Fatal(err)
	}
	grid := ref.tensor(t, "image_grid_thw").I64()
	if grid[0] != 1 || int(grid[1]) != gh || int(grid[2]) != gw {
		t.Fatalf("grid %dx%d, want %v", gh, gw, grid)
	}
	compare(t, "vision pixel_values", pix.F32(), ref.tensor(t, "pixel_values").F32(), 1e-6)
	compare(t, "vae pixels", VAEPixels(img).F32(), ref.tensor(t, "vae_pixels").F32(), 1e-6)
}

// TestVAEEncoderParity encodes the reference condition pixels and compares
// the normalised latents with the pipeline's.
func TestVAEEncoderParity(t *testing.T) {
	ref := loadRef(t, "edit")
	dir := filepath.Join(snapshotDir(t), "vae")
	cfg, err := LoadVAEConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	px := ref.tensor(t, "vae_pixels") // [4, H, W]
	s := px.Shape()
	g, err := BuildVAEEncoder(cfg, set, s[1], s[2])
	if err != nil {
		t.Fatal(err)
	}
	sess, err := graph.Compile(g)
	if err != nil {
		t.Fatal(err)
	}
	out, err := sess.Run(map[string]*tensor.Tensor{"x": px.Reshape(1, s[0], s[1], s[2])})
	if err != nil {
		t.Fatal(err)
	}
	got := PackLatents(cfg, out["z"])
	compare(t, "cond latents", got.F32(), ref.tensor(t, "cond_latents").F32(), 2e-3)
}
