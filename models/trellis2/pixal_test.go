package trellis2

import (
	"image"
	"os"
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/sparse"
	"github.com/giraffesyo/ingot/tensor"
)

func pixalDir(t testing.TB) string { return hubSnapshot(t, "PIXAL3D_DIR", "TencentARC/Pixal3D") }

// nafFile opens the converted upsampler checkpoint or skips.
func nafFile(t testing.TB) string {
	t.Helper()
	path := os.Getenv("NAF_WEIGHTS")
	if path == "" {
		hub := os.Getenv("HF_HUB_CACHE")
		if hub == "" {
			home, _ := os.UserHomeDir()
			hub = filepath.Join(home, ".cache/huggingface/hub")
		}
		path = filepath.Join(hub, "naf", "naf.safetensors")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("upsampler weights not converted (tools/export/naf_convert.py): %v", err)
	}
	return path
}

// TestProjectParity: cells of a grid projected through the camera, and
// image features sampled there, against the reference's ProjGrid — at the
// default field of view and at a wide one with an off-unit scale.
func TestProjectParity(t *testing.T) {
	for _, name := range []string{"pixal_proj", "pixal_proj_wide"} {
		ref := loadRef(t, name)
		f := func(k string) float64 { return ref.Meta[k].(float64) }
		R := int(f("grid"))
		cam := Camera{FovX: f("fov"), Distance: f("distance"), MeshScale: f("mesh_scale")}
		got, err := ProjectFeatures(cam, GridCoords(R), R, int(f("image")), ref.tensor(t, "features"), nil)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, name, got.F32(), ref.tensor(t, "projected").F32(), 2e-4)
	}
}

// TestCameraFromFov: the derived distance puts the cube's near edge on the
// image border — the grid's corner cell projects to the first pixel
// column's left edge (u = −1 plus the reference's half-pixel offset).
func TestCameraFromFov(t *testing.T) {
	const res = 512
	cam := CameraFromFov(DefaultFovX, 1)
	u, _ := cam.project(sparse.Coord{0, 0, 0}, 16, res)
	// Corner (−1, ·, −1): x = −0.5, depth = distance + 0.5.
	want := float32(-0.5/((cam.Distance+0.5)*0.4571) + 1.0/res)
	if d := u - want; d > 2e-3 || d < -2e-3 {
		t.Fatalf("corner projects to u = %g, want about %g", u, want)
	}
	// The centre of the near face's edge, at depth = distance, is the cell
	// the distance is derived from: exactly the image border.
	near := Camera{FovX: cam.FovX, Distance: cam.Distance + 0.5, MeshScale: 1}
	if u, _ := near.project(sparse.Coord{0, 0, 15}, 16, res); u < -1-1e-3 || u > -1+1.0/res+1e-3 {
		t.Fatalf("border cell projects to u = %g, want −1", u)
	}
}

// TestUpsamplerParity: the NAF upsampler with its released weights, at
// every output pixel, against the reference — with the encoder at the
// output's resolution and pooled down from twice it.
func TestUpsamplerParity(t *testing.T) {
	path := nafFile(t)
	for _, name := range []string{"pixal_naf", "pixal_naf_pool"} {
		ref := loadRef(t, name)
		size, out := int(ref.Meta["size"].(float64)), int(ref.Meta["out"].(float64))
		g, err := BuildNAFEncoder(openFile(t, path), size, out)
		s := compile(t, g, err)
		res, err := s.Run(map[string]*tensor.Tensor{"image": ref.tensor(t, "image")})
		if err != nil {
			t.Fatal(err)
		}
		up, err := Upsample(res["q"], ref.tensor(t, "features"), out)
		if err != nil {
			t.Fatal(err)
		}
		want := ref.tensor(t, "upsampled")
		C := want.Shape()[1]
		got := make([]float32, out*out*C)
		for y := range out {
			for x := range out {
				up.at(x, y, got[(y*out+x)*C:(y*out+x+1)*C])
			}
		}
		compare(t, name, got, want.F32(), 2e-3)
	}
}

// TestProjFlowParity: Pixal3D's flow transformers — cross-attention over
// the image's global tokens plus projected features per token — truncated
// to the reference's block count: the dense structure model and the
// sparse shape model.
func TestProjFlowParity(t *testing.T) {
	dir := filepath.Join(pixalDir(t), "ckpts")
	for _, c := range []struct{ ref, ckpt string }{
		{"pixal_ss_flow", "ss_flow_img_dit_1_3B_64_bf16"},
		{"pixal_slat_flow", "slat_flow_img2shape_dit_1_3B_512_bf16"},
	} {
		ref := loadRef(t, c.ref)
		cfg, err := LoadFlowConfig(filepath.Join(dir, c.ckpt+".json"))
		if err != nil {
			t.Fatal(err)
		}
		coords := GridCoords(cfg.Resolution)
		if c.ref == "pixal_slat_flow" {
			coords = cells(ref.tensor(t, "coords"))
		}
		cond := ref.tensor(t, "cond")
		f := openFile(t, filepath.Join(dir, c.ckpt+".safetensors"))
		blocks := int(ref.Meta["blocks"].(float64))
		g, err := BuildFlow(cfg, f, coords, cond.Shape()[0], blocks, false)
		s := compile(t, g, err)
		feeds := flowFeeds(t, cfg, f, blocks, ref.tensor(t, "x"), cond, float32(ref.Meta["t"].(float64)))
		feeds["proj"] = ref.tensor(t, "proj")
		out, err := s.Run(feeds)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, c.ref, out["v"].F32(), ref.tensor(t, "v").F32(), 2e-3)
	}
}

// TestPreprocessMarginParity: Pixal3D's crop — the subject's square
// widened by 1.1, transparent where it leaves the frame — matches the
// reference exactly.
func TestPreprocessMarginParity(t *testing.T) {
	ref := loadRef(t, "pixal_preprocess")
	rgba := ref.tensor(t, "rgba")
	h, w := rgba.Shape()[0], rgba.Shape()[1]
	got, err := PreprocessMargin(&image.NRGBA{Pix: rgba.U8(), Stride: 4 * w, Rect: image.Rect(0, 0, w, h)}, 1.1)
	if err != nil {
		t.Fatal(err)
	}
	want := ref.tensor(t, "subject")
	side := want.Shape()[0]
	if got.Bounds().Dx() != side || got.Bounds().Dy() != side {
		t.Fatalf("subject is %v, reference %d×%d", got.Bounds(), side, side)
	}
	for i, v := range want.U8() {
		if g := got.Pix[4*(i/3)+i%3]; g != v {
			t.Fatalf("subject sample %d = %d, reference %d", i, g, v)
		}
	}
}
