package wan

import (
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// runDiT evaluates the transformer once: the text condition graph, then
// the main graph over x [C, F, H, W]; returns the velocity [C, F, H, W].
func runDiT(t testing.TB, cfg DiTConfig, dir string, x, text *tensor.Tensor, ts Timesteps, blocks int, device string) *tensor.Tensor {
	t.Helper()
	set := openSet(t, filepath.Join(dir, "transformer"))
	widen := device != "cpu"
	cg, err := BuildTextCond(cfg, set, blocks, widen)
	cr := compile(t, cg, err, device)
	kv, err := cr.Run(map[string]*tensor.Tensor{"text": text})
	if err != nil {
		t.Fatal(err)
	}
	s := x.Shape()
	grid := Grid{F: s[1], H: s[2], W: s[3]}
	g, err := BuildDiT(cfg, set, grid, len(ts.Values), blocks, widen)
	r := compile(t, g, err, device)
	feeds := map[string]*tensor.Tensor{"x": Patchify(x, grid), "t": tensor.FromF32(ts.Values, len(ts.Values))}
	if ts.Seg != nil {
		feeds["seg"] = tensor.FromI64(ts.Seg, len(ts.Seg))
	}
	for k, v := range kv {
		feeds[k] = v
	}
	out, err := r.Run(feeds)
	if err != nil {
		t.Fatal(err)
	}
	return Unpatchify(out["v"], grid)
}

func TestPatchifyRoundTrip(t *testing.T) {
	g := Grid{F: 2, H: 4, W: 6}
	x := tensor.New(tensor.F32, 3, g.F, g.H, g.W)
	for i := range x.F32() {
		x.F32()[i] = float32(i)
	}
	// Patchify's feature order (c, kh, kw) differs from Unpatchify's
	// (kh, kw, c); permute between them to round-trip.
	p := Patchify(x, g)
	q := tensor.New(tensor.F32, p.Shape()...)
	C := 3
	for tok := range g.Tokens() {
		for c := range C {
			for k := range 4 {
				q.F32()[tok*4*C+k*C+c] = p.F32()[tok*4*C+c*4+k]
			}
		}
	}
	y := Unpatchify(q, g)
	for i, v := range y.F32() {
		if v != x.F32()[i] {
			t.Fatalf("element %d: %g != %g", i, v, x.F32()[i])
		}
	}
}

func testDiT(t *testing.T, dir, ref string, tol float64) {
	c := loadRef(t, ref)
	cfg, err := LoadDiTConfig(filepath.Join(dir, "transformer"))
	if err != nil {
		t.Fatal(err)
	}
	x, text := c.tensor(t, "x"), c.tensor(t, "text")
	tv := float32(c.metaFloat("t"))
	s := x.Shape()
	grid := Grid{F: s[1], H: s[2], W: s[3]}
	blocks := int(c.metaFloat("layers"))
	devs := devices()
	if len(devs) > 1 {
		devs = append(devs, "gpu-bf16")
	}
	for _, dev := range devs {
		t.Run(dev, func(t *testing.T) {
			tol := tol
			if dev == "gpu-bf16" {
				tol = 2e-2 // bf16 operands, f32 accumulation
			}
			got := runDiT(t, cfg, dir, x, text, FirstFrameFixed(grid, tv), blocks, dev)
			compare(t, "ti2v", got.F32(), c.tensor(t, "out_ti2v").F32(), tol)
			got = runDiT(t, cfg, dir, x, text, Timesteps{Values: []float32{tv}}, blocks, dev)
			compare(t, "scalar t", got.F32(), c.tensor(t, "out_scalar").F32(), tol)
		})
	}
}

func TestDiTTiny(t *testing.T)    { testDiT(t, tinyDir(t), "dit", 1e-5) }
func TestDiTRealOne(t *testing.T) { testDiT(t, realDir(t), "real_dit_l1", 1e-4) }
