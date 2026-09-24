package qwenimage

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"math/rand/v2"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"time"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
	"github.com/giraffesyo/ingot/tokenizer"
)

// Options configures one text-to-image generation.
type Options struct {
	Prompt string
	Width  int // pixels, rounded down to a multiple of 32; default 1024
	Height int
	Steps  int // default 40
	Seed   uint64

	// Device runs the DiT on "cpu", "gpu" (Metal, darwin/arm64) or "auto"
	// (the default: gpu when available).
	Device string
	// Fast (GPU only) runs the DiT's GEMMs with bf16 activations: ~2.6x the
	// matrix throughput, outputs within ~1% (the reference pipeline itself
	// runs in bf16). Off by default: f32 inputs match diffusers' f32 run.
	Fast bool

	// Latents, when set, replaces the seeded noise: packed [h·w, 64]
	// (parity tests inject the reference pipeline's torch noise).
	Latents *tensor.Tensor
	// Log receives one line per stage with timings; nil is silent.
	Log func(format string, args ...any)
}

// Result is a generation's output and timings.
type Result struct {
	Image  *image.NRGBA
	Float  *tensor.Tensor // [4, H, W] decoder output in [-1, 1]
	Stages map[string]time.Duration
}

// The pipeline's chat template (QwenImage21Pipeline.prompt_template_t2i).
const (
	sysPrompt  = "Comprehend and analyze the provided prompt."
	sysMessage = "<|im_start|>system\n" + sysPrompt + "<|im_end|>\n"
	t2iPrefix  = sysMessage + "<|im_start|>user\n"
	t2iSuffix  = "<|im_end|>\n<|im_start|>assistant\n"
)

// Generate runs text-to-image from the Hugging Face snapshot dir. The stages
// run one after another and each releases its weights before the next loads
// (text encoder ~30 GB packed, DiT ~28.5 GB), so peak memory is one model.
// Sampling follows the reference pipeline: no CFG (true_cfg_scale = 1, the
// model's intended use), flow-matching Euler, prefix-KV reuse.
func Generate(dir string, opt Options) (*Result, error) {
	if opt.Width == 0 {
		opt.Width = 1024
	}
	if opt.Height == 0 {
		opt.Height = 1024
	}
	if opt.Steps == 0 {
		opt.Steps = 40
	}
	logf := opt.Log
	if logf == nil {
		logf = func(string, ...any) {}
	}
	res := &Result{Stages: map[string]time.Duration{}}
	stage := func(name string, t0 time.Time) {
		res.Stages[name] = time.Since(t0)
		logf("%-13s %8.2fs", name, time.Since(t0).Seconds())
	}
	W, H := opt.Width/32*32, opt.Height/32*32
	lh, lw := H/16, W/16

	// 1. Tokenise the chat-templated prompt.
	t0 := time.Now()
	tok, err := tokenizer.Load(filepath.Join(dir, "processor", "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	prompt := opt.Prompt
	if prompt == "" {
		prompt = " " // Qwen has no BOS: the encoder needs something to read
	}
	ids, err := tok.Encode(t2iPrefix + prompt + t2iSuffix)
	if err != nil {
		return nil, err
	}
	sysIDs, err := tok.Encode(sysMessage)
	if err != nil {
		return nil, err
	}
	drop := len(sysIDs)
	stage("tokenize", t0)

	// 2. Text encoder → conditioning hidden states.
	t0 = time.Now()
	embeds, err := encodeText(dir, ids, drop)
	if err != nil {
		return nil, err
	}
	release()
	stage("text encoder", t0)

	// 3. DiT: prefix pass once, then the target steps.
	t0 = time.Now()
	x, err := denoise(dir, embeds, lh, lw, opt, logf)
	if err != nil {
		return nil, err
	}
	release()
	stage("denoise", t0)

	// 4. VAE decode.
	t0 = time.Now()
	img, err := decode(dir, x, lh, lw)
	if err != nil {
		return nil, err
	}
	release()
	stage("decode", t0)
	res.Float = img
	res.Image = toImage(img)
	return res, nil
}

func encodeText(dir string, ids []int64, drop int) (*tensor.Tensor, error) {
	tdir := filepath.Join(dir, "text_encoder")
	cfg, err := LoadTextConfig(tdir)
	if err != nil {
		return nil, err
	}
	set, err := safetensors.OpenDir(tdir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	te, err := NewTextEncoder(cfg, set)
	if err != nil {
		return nil, err
	}
	s, err := compile(te.Build(len(ids), drop, cfg.NumHiddenLayers))
	if err != nil {
		return nil, err
	}
	x, err := te.Embed(ids)
	if err != nil {
		return nil, err
	}
	out, err := s.Run(map[string]*tensor.Tensor{"x": x})
	if err != nil {
		return nil, err
	}
	return out["hidden"].Clone(), nil
}

func denoise(dir string, embeds *tensor.Tensor, lh, lw int, opt Options, logf func(string, ...any)) (*tensor.Tensor, error) {
	ddir := filepath.Join(dir, "transformer")
	cfg, err := LoadDiTConfig(ddir)
	if err != nil {
		return nil, err
	}
	scfg, err := LoadSchedulerConfig(filepath.Join(dir, "scheduler"))
	if err != nil {
		return nil, err
	}
	set, err := safetensors.OpenDir(ddir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	txt := embeds.Shape()[0]
	mask := make([]bool, txt+lh*lw/imgTokensPerSlot)
	for i := txt; i < len(mask); i++ {
		mask[i] = true
	}
	l, err := NewDiTLayout(cfg, mask, nil, [][3]int{{1, lh, lw}})
	if err != nil {
		return nil, err
	}
	x := opt.Latents
	if x == nil {
		x = gaussian(opt.Seed, l.Target, cfg.InChannels)
	} else {
		if !x.Shape().Equal([]int{l.Target, cfg.InChannels}) {
			return nil, fmt.Errorf("qwenimage: latents %v, want [%d %d]", x.Shape(), l.Target, cfg.InChannels)
		}
		x = x.Clone()
	}
	sched := NewSchedule(scfg, opt.Steps, l.Target)
	gpu := opt.Device == "gpu" || ((opt.Device == "" || opt.Device == "auto") && metalAvailable())
	if gpu {
		return denoiseMetal(cfg, set, l, embeds, x, sched, opt.Steps, opt.Fast, logf)
	}
	if opt.Device != "" && opt.Device != "auto" && opt.Device != "cpu" {
		return nil, fmt.Errorf("qwenimage: unknown device %q (cpu, gpu, auto)", opt.Device)
	}
	pre, err := compile(BuildDiTPrefix(cfg, set, l, cfg.NumLayers, false))
	if err != nil {
		return nil, err
	}
	tgt, err := compile(BuildDiTTarget(cfg, set, l, cfg.NumLayers))
	if err != nil {
		return nil, err
	}

	t0 := time.Now()
	kv, err := pre.Run(l.PrefixFeeds(embeds, nil))
	if err != nil {
		return nil, err
	}
	logf("  prefix      %8.2fs (%d tokens)", time.Since(t0).Seconds(), l.Prefix)
	for i := range opt.Steps {
		t0 = time.Now()
		out, err := tgt.Run(l.TargetFeeds(x, sched.ModelTime(i), kv))
		if err != nil {
			return nil, err
		}
		sched.Step(i, x.F32(), out["out"].F32())
		tgt.Release(out)
		logf("  step %2d/%d  %8.2fs", i+1, opt.Steps, time.Since(t0).Seconds())
	}
	return x, nil
}

// denoiseMetal runs the prefix pass and every step on the GPU.
func denoiseMetal(cfg DiTConfig, set *safetensors.Set, l *DiTLayout, embeds, x *tensor.Tensor, sched Schedule, steps int,
	fast bool, logf func(string, ...any)) (*tensor.Tensor, error) {
	t0 := time.Now()
	m, err := NewMetalDiT(cfg, set, l, cfg.NumLayers)
	if err != nil {
		return nil, err
	}
	defer m.Close()
	m.Fast = fast
	if err := m.Prefix(embeds, nil); err != nil {
		return nil, err
	}
	logf("  prefix      %8.2fs (%d tokens, gpu)", time.Since(t0).Seconds(), l.Prefix)
	for i := range steps {
		t0 = time.Now()
		v, err := m.Step(x, sched.ModelTime(i))
		if err != nil {
			return nil, err
		}
		sched.Step(i, x.F32(), v.F32())
		logf("  step %2d/%d  %8.2fs (gpu)", i+1, steps, time.Since(t0).Seconds())
	}
	return x, nil
}

func decode(dir string, x *tensor.Tensor, lh, lw int) (*tensor.Tensor, error) {
	vdir := filepath.Join(dir, "vae")
	cfg, err := LoadVAEConfig(vdir)
	if err != nil {
		return nil, err
	}
	set, err := safetensors.OpenDir(vdir)
	if err != nil {
		return nil, err
	}
	defer set.Close()
	s, err := compile(BuildVAEDecoder(cfg, set, lh, lw))
	if err != nil {
		return nil, err
	}
	out, err := s.Run(map[string]*tensor.Tensor{"z": UnpackLatents(cfg, x, lh, lw)})
	if err != nil {
		return nil, err
	}
	img := out["image"]
	return img.Reshape(img.Shape()[1:]...).Clone(), nil
}

func compile(g *graph.Graph, err error) (*graph.Session, error) {
	if err != nil {
		return nil, err
	}
	return graph.Compile(g)
}

// release returns a finished stage's memory to the OS: once its sessions and
// mapped files are unreachable, a GC runs the weak-keyed pack cache's
// cleanups and a second collects the packs.
func release() {
	runtime.GC()
	runtime.GC()
	debug.FreeOSMemory()
}

// gaussian draws standard-normal packed latents from seed. (The reference
// pipeline draws with torch's generator; images differ from it per seed.)
func gaussian(seed uint64, n, c int) *tensor.Tensor {
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	x := tensor.New(tensor.F32, n, c)
	for i := range x.F32() {
		x.F32()[i] = float32(r.NormFloat64())
	}
	return x
}

// toImage is VaeImageProcessor.postprocess for 4 channels: (x/2 + 1/2)
// clamped to [0, 1], ×255 rounded, as RGBA (the checkpoint's alpha
// channel is real — transparent generations).
func toImage(x *tensor.Tensor) *image.NRGBA {
	s := x.Shape()
	C, H, W := s[0], s[1], s[2]
	img := image.NewNRGBA(image.Rect(0, 0, W, H))
	f := x.F32()
	q := func(v float32) uint8 {
		v = v*0.5 + 0.5
		v = min(max(v, 0), 1)
		return uint8(math.Round(float64(v * 255)))
	}
	for y := range H {
		for xx := range W {
			p := y*W + xx
			c := color.NRGBA{A: 255}
			c.R, c.G, c.B = q(f[p]), q(f[H*W+p]), q(f[2*H*W+p])
			if C > 3 {
				c.A = q(f[3*H*W+p])
			}
			img.SetNRGBA(xx, y, c)
		}
	}
	return img
}
