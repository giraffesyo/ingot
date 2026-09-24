package qwenimage

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// TestVAEDecoderParity decodes the reference 16×16 latent with the real
// weights and compares against the diffusers output (CPU, f32).
func TestVAEDecoderParity(t *testing.T) {
	dir := filepath.Join(snapshotDir(t), "vae")
	ref := loadRef(t, "vae_dec")
	cfg, err := LoadVAEConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	z := ref.tensor(t, "z") // [1, 64, 1, 16, 16]
	zs := z.Shape()
	g, err := BuildVAEDecoder(cfg, set, zs[3], zs[4])
	if err != nil {
		t.Fatal(err)
	}
	s, err := graph.Compile(g)
	if err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	out, err := s.Run(map[string]*tensor.Tensor{"z": z.Reshape(zs[0], zs[1], zs[3], zs[4])})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("decode %dx%d latent: %v (first run, packs weights)", zs[3], zs[4], time.Since(t0))
	s.Release(out)
	s.Profile = testing.Verbose()
	t0 = time.Now()
	if out, err = s.Run(map[string]*tensor.Tensor{"z": z.Reshape(zs[0], zs[1], zs[3], zs[4])}); err != nil {
		t.Fatal(err)
	}
	t.Logf("decode %dx%d latent: %v (warm)", zs[3], zs[4], time.Since(t0))
	if testing.Verbose() {
		for _, st := range s.Stats() {
			t.Logf("  %-22s n=%3d %8.1f ms", st.OpType, st.Count, float64(st.Total.Microseconds())/1e3)
		}
	}
	want := ref.tensor(t, "image") // [1, 4, 1, 256, 256]
	got := out["image"]
	ws := want.Shape()
	if !got.Shape().Equal([]int{ws[0], ws[1], ws[3], ws[4]}) {
		t.Fatalf("shape %v, want %v", got.Shape(), ws)
	}
	compare(t, "image", got.F32(), want.F32(), 1e-3)
}
