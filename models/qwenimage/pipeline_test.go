package qwenimage

import (
	"fmt"
	"image"
	"math"
	"image/png"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
	"github.com/giraffesyo/ingot/tokenizer"
)

// fullModel skips unless QWENIMAGE_FULL=1: the full-size tests pack the
// 7.1B DiT (~28.5 GB resident) and read ~14 GB of weights.
func fullModel(t *testing.T) {
	t.Helper()
	if os.Getenv("QWENIMAGE_FULL") != "1" {
		t.Skip("set QWENIMAGE_FULL=1 to run the full-size model (~30 GB RAM)")
	}
}

func rssGB() float64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return float64(m.Sys) / 1e9
}

// TestDenoiseParity runs the full 32-layer DiT over the reference prompt
// embeddings and noise for all 4 steps of the reference schedule, checking
// the latents after every step, then decodes and checks the image.
func TestDenoiseParity(t *testing.T) {
	fullModel(t)
	snap := snapshotDir(t)
	ref := loadRef(t, "pipeline")
	cfg, err := LoadDiTConfig(filepath.Join(snap, "transformer"))
	if err != nil {
		t.Fatal(err)
	}
	scfg, err := LoadSchedulerConfig(filepath.Join(snap, "scheduler"))
	if err != nil {
		t.Fatal(err)
	}
	set, err := safetensors.OpenDir(filepath.Join(snap, "transformer"))
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()

	embeds := ref.tensor(t, "prompt_embeds") // [T, 4096]
	x := ref.tensor(t, "latents0").Clone()   // [h·w, 64]
	hw := ref.Meta["hw"].([]any)
	lh, lw := int(hw[0].(float64))/16, int(hw[1].(float64))/16
	txt := embeds.Shape()[0]
	mask := make([]bool, txt+lh*lw/imgTokensPerSlot)
	for i := txt; i < len(mask); i++ {
		mask[i] = true
	}
	l, err := NewDiTLayout(cfg, mask, nil, [][3]int{{1, lh, lw}})
	if err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	build := func(g *graph.Graph, err error) *graph.Session {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		s, err := graph.Compile(g)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	pre := build(BuildDiTPrefix(cfg, set, l, cfg.NumLayers, false))
	tgt := build(BuildDiTTarget(cfg, set, l, cfg.NumLayers))
	t.Logf("built + compiled in %v", time.Since(t0))

	t0 = time.Now()
	kv, err := pre.Run(l.PrefixFeeds(embeds, nil))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("prefix pass (%d tokens): %v, sys %.1f GB", l.Prefix, time.Since(t0), rssGB())

	sched := NewSchedule(scfg, len(ref.tensor(t, "timesteps").F32()), l.Target)
	for i := range len(sched.Timesteps) {
		t0 = time.Now()
		out, err := tgt.Run(l.TargetFeeds(x, sched.ModelTime(i), kv))
		if err != nil {
			t.Fatal(err)
		}
		sched.Step(i, x.F32(), out["out"].F32())
		tgt.Release(out)
		t.Logf("step %d: %v, sys %.1f GB", i, time.Since(t0), rssGB())
		compare(t, fmt.Sprintf("latents after step %d", i+1), x.F32(), ref.tensor(t, fmt.Sprintf("latents%d", i+1)).F32(), 2e-3)
	}

	// Decode: unpack [h·w, C] → [1, C, h, w], denormalise, VAE.
	vdir := filepath.Join(snap, "vae")
	vcfg, err := LoadVAEConfig(vdir)
	if err != nil {
		t.Fatal(err)
	}
	vset, err := safetensors.OpenDir(vdir)
	if err != nil {
		t.Fatal(err)
	}
	defer vset.Close()
	vae := build(BuildVAEDecoder(vcfg, vset, lh, lw))
	img, err := vae.Run(map[string]*tensor.Tensor{"z": UnpackLatents(vcfg, x, lh, lw)})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "image", img["image"].F32(), ref.tensor(t, "image").F32(), 5e-3)
}

// TestGenerateParity is the whole pipeline from the prompt text: tokenizer →
// text encoder → DiT (reference noise) → VAE, against diffusers' image.
func TestGenerateParity(t *testing.T) {
	fullModel(t)
	for _, dev := range []string{"cpu", "gpu"} {
		if dev == "gpu" && !metalAvailable() {
			continue
		}
		t.Run(dev, func(t *testing.T) { generateParity(t, dev) })
	}
}

func generateParity(t *testing.T, device string) {
	snap := snapshotDir(t)
	ref := loadRef(t, "pipeline")
	hw := ref.Meta["hw"].([]any)
	res, err := Generate(snap, Options{
		Device:  device,
		Prompt:  ref.Meta["prompt"].(string),
		Width:   int(hw[1].(float64)),
		Height:  int(hw[0].(float64)),
		Steps:   int(ref.Meta["steps"].(float64)),
		Latents: ref.tensor(t, "latents0"),
		Log:     t.Logf,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("peak RSS %.1f GB", peakRSSGB())
	compare(t, "image", res.Float.F32(), ref.tensor(t, "image").F32(), 5e-3)
}

// TestEditPromptIDs: the edit chat template, pad-expanded for the
// reference image, tokenises to the processor's input_ids.
func TestEditPromptIDs(t *testing.T) {
	ref := loadRef(t, "edit")
	tok, err := tokenizer.Load(filepath.Join(snapshotDir(t), "processor", "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	grid := ref.tensor(t, "image_grid_thw").I64()
	ids, err := tok.Encode(promptText(ref.Meta["prompt"].(string), []condition{{gh: int(grid[1]), gw: int(grid[2])}}))
	if err != nil {
		t.Fatal(err)
	}
	want := ref.tensor(t, "input_ids").I64()
	if len(ids) != len(want) {
		t.Fatalf("%d ids, want %d", len(ids), len(want))
	}
	for i := range ids {
		if ids[i] != want[i] {
			t.Fatalf("ids[%d] = %d, want %d", i, ids[i], want[i])
		}
	}
}

// TestEditParity: full image editing from the prompt and the condition PNG
// (GPU), with the reference noise, against diffusers' output.
func TestEditParity(t *testing.T) {
	fullModel(t)
	if !metalAvailable() {
		t.Skip("edit mode needs the GPU text encoder")
	}
	ref := loadRef(t, "edit")
	f, err := os.Open(filepath.Join(refDir, "edit_cond.png"))
	if err != nil {
		t.Fatal(err)
	}
	img, err := png.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	// f32, and fast (bf16 GEMM inputs, the fused prefix with per-row key
	// limits): the image stays within a few 8-bit levels.
	for _, fast := range []bool{false, true} {
		res, err := Generate(snapshotDir(t), Options{
			Prompt:     ref.Meta["prompt"].(string),
			Images:     []image.Image{img},
			Resolution: int(ref.Meta["resolution"].(float64)),
			Steps:      int(ref.Meta["steps"].(float64)),
			Latents:    ref.tensor(t, "latents0"),
			Device:     "gpu",
			Fast:       fast,
			Log:        t.Logf,
		})
		if err != nil {
			t.Fatal(err)
		}
		want := ref.tensor(t, "image").F32()
		if !fast {
			compare(t, "edit image (f32)", res.Float.F32(), want, 1e-2)
			continue
		}
		// bf16 drifts pixel by pixel over the trajectory; bound the mean.
		var sum, mx float64
		for i, v := range res.Float.F32() {
			d := math.Abs(float64(v - want[i]))
			sum, mx = sum+d, max(mx, d)
		}
		mean := sum / float64(len(want))
		t.Logf("edit image (fast): mean |Δ| %.4f, max %.3f (on [-1, 1])", mean, mx)
		if mean > 0.01 || mx > 0.5 {
			t.Fatalf("fast edit image drifted: mean %.4f max %.3f", mean, mx)
		}
	}
}
