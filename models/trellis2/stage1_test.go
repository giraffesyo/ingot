package trellis2

import (
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	"github.com/giraffesyo/ingot/tensor"
)

// TestDinoParity runs the full 24-layer conditioning encoder on the
// reference image and compares the features with transformers.
func TestDinoParity(t *testing.T) {
	dir := dinoDir(t)
	ref := loadRef(t, "dino")
	cfg, err := LoadDinoConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	g, err := BuildDino(cfg, openFile(t, filepath.Join(dir, "model.safetensors")), int(ref.Meta["size"].(float64)))
	s := compile(t, g, err)
	out, err := s.Run(map[string]*tensor.Tensor{"image": ref.tensor(t, "image")})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "features", out["features"].F32(), ref.tensor(t, "features").F32(), 2e-3)
}

// TestStructureFlowParity runs the sparse-structure flow transformer,
// truncated to the reference's block count, for one velocity evaluation.
func TestStructureFlowParity(t *testing.T) {
	base := filepath.Join(trellisDir(t), "ckpts", "ss_flow_img_dit_1_3B_64_bf16")
	ref := loadRef(t, "ss_flow")
	cfg, err := LoadFlowConfig(base + ".json")
	if err != nil {
		t.Fatal(err)
	}
	cond := ref.tensor(t, "cond")
	f, blocks := openFile(t, base+".safetensors"), int(ref.Meta["blocks"].(float64))
	g, err := BuildFlow(cfg, f, GridCoords(cfg.Resolution), cond.Shape()[0], blocks, false)
	s := compile(t, g, err)
	out, err := s.Run(flowFeeds(t, cfg, f, blocks, ref.tensor(t, "x"), cond, float32(ref.Meta["t"].(float64))))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "v", out["v"].F32(), ref.tensor(t, "v").F32(), 2e-3)
}

// TestStructureDecoderParity decodes the reference latent to occupancy
// logits through the dense 3-D conv decoder.
func TestStructureDecoderParity(t *testing.T) {
	base := filepath.Join(structureDir(t), "ckpts", "ss_dec_conv3d_16l8_fp16")
	ref := loadRef(t, "ss_dec")
	cfg, err := LoadStructureDecoderConfig(base + ".json")
	if err != nil {
		t.Fatal(err)
	}
	z := ref.tensor(t, "z")
	g, err := BuildStructureDecoder(cfg, openFile(t, base+".safetensors"), z.Shape()[2])
	s := compile(t, g, err)
	out, err := s.Run(map[string]*tensor.Tensor{"z": z})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "logits", out["logits"].F32(), ref.tensor(t, "logits").F32(), 2e-3)
}

// TestStructureSampling runs the whole sparse-structure stage — the
// pipeline's guided sampler over the 30-block flow model from the
// reference noise, then the decoder — and compares the latent and the
// occupied cells with the reference pipeline. Long: 22 evaluations of a
// 1.3B-parameter transformer over 4096 tokens.
func TestStructureSampling(t *testing.T) {
	if testing.Short() {
		t.Skip("full structure sampling is slow")
	}
	base := filepath.Join(trellisDir(t), "ckpts", "ss_flow_img_dit_1_3B_64_bf16")
	decBase := filepath.Join(structureDir(t), "ckpts", "ss_dec_conv3d_16l8_fp16")
	ref := loadRef(t, "ss_sample")
	cfg, err := LoadFlowConfig(base + ".json")
	if err != nil {
		t.Fatal(err)
	}
	cond, noise := ref.tensor(t, "cond"), ref.tensor(t, "noise")
	coords := GridCoords(cfg.Resolution)
	f := openFile(t, base+".safetensors")
	g, err := BuildFlow(cfg, f, coords, cond.Shape()[0], cfg.NumBlocks, false)
	s := compile(t, g, err)

	params := SamplerParams{SigmaMin: ref.Meta["sigma_min"].(float64)}
	raw, _ := json.Marshal(ref.Meta["params"])
	if err := json.Unmarshal(raw, &params); err != nil {
		t.Fatal(err)
	}
	kv, err := FlowConditions(cfg, f, cfg.NumBlocks, cond, tensor.New(tensor.F32, cond.Shape()...))
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	z, err := Sample(FlowVelocity(s, len(coords), cfg.InChannels, kv[0], kv[1], nil), noise.F32(), params, func(i, n int) {
		t.Logf("step %d/%d (%.1fs)", i, n, time.Since(start).Seconds())
	})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "z", z, ref.tensor(t, "z").F32(), 5e-2)

	dcfg, err := LoadStructureDecoderConfig(decBase + ".json")
	if err != nil {
		t.Fatal(err)
	}
	g, err = BuildStructureDecoder(dcfg, openFile(t, decBase+".safetensors"), cfg.Resolution)
	dec := compile(t, g, err)
	// Tokens are [T, C]; the decoder takes [1, C, n, n, n].
	n, C := cfg.Resolution, cfg.InChannels
	zt := tensor.New(tensor.F32, 1, C, n, n, n)
	for tok := range n * n * n {
		for c := range C {
			zt.F32()[c*n*n*n+tok] = z[tok*C+c]
		}
	}
	out, err := dec.Run(map[string]*tensor.Tensor{"z": zt})
	if err != nil {
		t.Fatal(err)
	}
	logits := out["logits"]
	cells, err := OccupiedCells(logits.F32(), logits.Shape()[2], 32)
	if err != nil {
		t.Fatal(err)
	}
	want := ref.tensor(t, "coords").I32()
	if len(cells) != len(want)/3 {
		t.Fatalf("%d occupied cells, reference has %d", len(cells), len(want)/3)
	}
	for i, c := range cells {
		if c != [3]int32{want[3*i], want[3*i+1], want[3*i+2]} {
			t.Fatalf("cell %d = %v, reference %v", i, c, want[3*i:3*i+3])
		}
	}
	t.Logf("%d occupied cells match the reference", len(cells))
}
