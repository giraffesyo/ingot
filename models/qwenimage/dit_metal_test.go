//go:build darwin && arm64

package qwenimage

import (
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/kernels/metal"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// TestMetalDiTParity: the GPU target step against the 1-layer diffusers
// reference (step 1 target rows and the cached step 2), with the prefix
// K/V from the CPU prefix graph.
func TestMetalDiTParity(t *testing.T) {
	if !metal.Available() {
		t.Skip("no Metal device")
	}
	dir := filepath.Join(snapshotDir(t), "transformer")
	ref := loadRef(t, "dit_l1")
	cfg, err := LoadDiTConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	toBools := func(x *tensor.Tensor) []bool {
		out := make([]bool, x.Numel())
		for i, v := range x.U8() {
			out[i] = v != 0
		}
		return out
	}
	hw := ref.Meta["latent_hw"].([]any)
	l, err := NewDiTLayout(cfg, toBools(ref.tensor(t, "img_mask")), toBools(ref.tensor(t, "encoder_hidden_states_mask")),
		[][3]int{{1, int(hw[0].(float64)), int(hw[1].(float64))}})
	if err != nil {
		t.Fatal(err)
	}
	g, err := BuildDiTPrefix(cfg, set, l, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	pre, err := graph.Compile(g)
	if err != nil {
		t.Fatal(err)
	}
	enc := ref.tensor(t, "encoder_hidden_states")
	kv, err := pre.Run(l.PrefixFeeds(enc.Reshape(enc.Shape()[1:]...), nil))
	if err != nil {
		t.Fatal(err)
	}
	m, err := NewMetalDiT(cfg, set, l, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	C := cfg.OutChannels
	// Prefix K/V two ways: from the CPU prefix graph, and computed on the
	// GPU (masked prefix attention, valid-row gather).
	for _, mode := range []string{"cpu prefix", "gpu prefix"} {
		if mode == "cpu prefix" {
			err = m.SetPrefix(kv)
		} else {
			err = m.Prefix(enc.Reshape(enc.Shape()[1:]...), nil)
		}
		if err != nil {
			t.Fatal(err)
		}
		if mode == "gpu prefix" {
			compare(t, "prefix k0 (GPU vs CPU)", f32view(m.kp[0]), kv["k0"].F32(), 1e-4)
			compare(t, "prefix v0 (GPU vs CPU)", f32view(m.vp[0]), kv["v0"].F32(), 1e-4)
		}
		step := func(name string, t0 float32) []float32 {
			x := ref.tensor(t, name)
			out, err := m.Step(x.Reshape(x.Shape()[1:]...), t0)
			if err != nil {
				t.Fatal(err)
			}
			return out.F32()
		}
		compare(t, mode+": step1 target rows", step("hidden_states", 0.7), ref.tensor(t, "out_step1").F32()[l.Prefix*C:], 5e-4)
		compare(t, mode+": step2 cached", step("hidden_states_step2", 0.4), ref.tensor(t, "out_step2_cached").F32(), 5e-4)
	}

	// Fast mode: bf16 GEMM inputs. Error grows to bf16's ~3 significant
	// digits (tolerance 1% of the output range).
	f, err := NewMetalDiT(cfg, set, l, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	f.Fast = true
	if err := f.Prefix(enc.Reshape(enc.Shape()[1:]...), nil); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		in, want string
		t0       float32
		rows     int
	}{{"hidden_states", "out_step1", 0.7, l.Prefix * C}, {"hidden_states_step2", "out_step2_cached", 0.4, 0}} {
		x := ref.tensor(t, c.in)
		out, err := f.Step(x.Reshape(x.Shape()[1:]...), c.t0)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "fast (bf16): "+c.want, out.F32(), ref.tensor(t, c.want).F32()[c.rows:], 0.15)
	}
}

// TestMetalTextEncoder: GPU vs CPU language model on a 2-layer truncation
// over the reference prompt ids (no full-model run needed).
func TestMetalTextEncoder(t *testing.T) {
	if !metal.Available() {
		t.Skip("no Metal device")
	}
	dir := filepath.Join(snapshotDir(t), "text_encoder")
	ref := loadRef(t, "pipeline")
	cfg, err := LoadTextConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	ids := ref.tensor(t, "input_ids").I64()
	const layers, drop = 2, 14
	te, err := NewTextEncoder(cfg, set)
	if err != nil {
		t.Fatal(err)
	}
	g, err := te.Build(len(ids), drop, layers)
	if err != nil {
		t.Fatal(err)
	}
	s, err := graph.Compile(g)
	if err != nil {
		t.Fatal(err)
	}
	x, _ := te.Embed(ids)
	cpu, err := s.Run(map[string]*tensor.Tensor{"x": x})
	if err != nil {
		t.Fatal(err)
	}
	m, err := NewMetalTextEncoder(cfg, set)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	gpu, err := m.Encode(ids, drop, layers)
	if err != nil {
		t.Fatal(err)
	}
	var maxw float64
	for _, v := range cpu["hidden"].F32() {
		maxw = max(maxw, float64(abs32(v)))
	}
	compare(t, "text encoder, 2 layers (GPU vs CPU)", gpu.F32(), cpu["hidden"].F32(), 1e-4*maxw)
}

// TestMetalVAEParity decodes the reference latent on the GPU.
func TestMetalVAEParity(t *testing.T) {
	if !metal.Available() {
		t.Skip("no Metal device")
	}
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
	z := ref.tensor(t, "z") // [1, C, 1, h, w] → NHWC [h·w, C]
	zs := z.Shape()
	C, h, w := zs[1], zs[3], zs[4]
	nhwc := tensor.New(tensor.F32, h*w, C)
	for c := range C {
		for p := range h * w {
			nhwc.F32()[p*C+c] = z.F32()[c*h*w+p]
		}
	}
	v, err := NewMetalVAE(cfg, set)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	img, err := v.Decode(nhwc, h, w)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "image (GPU VAE)", img.F32(), ref.tensor(t, "image").F32(), 1e-3)
}

// TestMetalTextEncoderMM: the GPU language model over the reference edit
// prompt (image placeholders filled by the Go vision tower, deepstack,
// M-RoPE) against the pipeline's prompt embeddings. Full model.
func TestMetalTextEncoderMM(t *testing.T) {
	fullModel(t)
	if !metal.Available() {
		t.Skip("no Metal device")
	}
	ref := loadRef(t, "edit")
	dir := filepath.Join(snapshotDir(t), "text_encoder")
	vcfg, err := LoadVisionConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadTextConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	grid := ref.tensor(t, "image_grid_thw").I64()
	gh, gw := int(grid[1]), int(grid[2])
	vg, err := BuildVision(vcfg, set, gh, gw)
	if err != nil {
		t.Fatal(err)
	}
	vs, err := graph.Compile(vg)
	if err != nil {
		t.Fatal(err)
	}
	vis, err := vs.Run(map[string]*tensor.Tensor{"pixels": ref.tensor(t, "pixel_values")})
	if err != nil {
		t.Fatal(err)
	}
	ids := ref.tensor(t, "input_ids").I64()
	const imageToken = 151655
	var imgPos []int
	for i, id := range ids {
		if id == imageToken {
			imgPos = append(imgPos, i)
		}
	}
	pos, err := MRoPEPositions(ids, imageToken, [][2]int{{gh / 2, gw / 2}})
	if err != nil {
		t.Fatal(err)
	}
	m, err := NewMetalTextEncoder(cfg, set)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const drop = 14
	got, err := m.EncodeMM(TextInputs{IDs: ids, ImagePositions: imgPos, ImageEmbeds: vis["merged"],
		Deepstack: []*tensor.Tensor{vis["deep0"], vis["deep1"], vis["deep2"]}, Positions: pos}, drop, cfg.NumHiddenLayers)
	if err != nil {
		t.Fatal(err)
	}
	want := ref.tensor(t, "prompt_embeds")
	var maxw float64
	for _, v := range want.F32() {
		maxw = max(maxw, float64(abs32(v)))
	}
	compare(t, "edit prompt embeds (GPU LM + Go vision)", got.F32(), want.F32(), 3e-4*maxw)
}

// TestMetalVisionParity: the GPU vision tower against the reference.
func TestMetalVisionParity(t *testing.T) {
	if !metal.Available() {
		t.Skip("no Metal device")
	}
	ref := loadRef(t, "edit")
	dir := filepath.Join(snapshotDir(t), "text_encoder")
	cfg, err := LoadVisionConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	grid := ref.tensor(t, "image_grid_thw").I64()
	v, err := NewMetalVision(cfg, set)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	merged, deep, err := v.Encode(ref.tensor(t, "pixel_values"), int(grid[1]), int(grid[2]))
	if err != nil {
		t.Fatal(err)
	}
	for name, got := range map[string]*tensor.Tensor{"vision_merged": merged, "vision_deepstack0": deep[0],
		"vision_deepstack1": deep[1], "vision_deepstack2": deep[2]} {
		want := ref.tensor(t, name).F32()
		var maxw float64
		for _, x := range want {
			maxw = max(maxw, float64(abs32(x)))
		}
		compare(t, name+" (GPU)", got.F32(), want, 2e-4*maxw)
	}
}

// TestMetalVAEEncoderParity: GPU encoder vs the reference condition latents.
func TestMetalVAEEncoderParity(t *testing.T) {
	if !metal.Available() {
		t.Skip("no Metal device")
	}
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
	v, err := NewMetalVAE(cfg, set)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	got, err := v.Encode(ref.tensor(t, "vae_pixels"))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "cond latents (GPU encoder)", got.F32(), ref.tensor(t, "cond_latents").F32(), 2e-3)
}
