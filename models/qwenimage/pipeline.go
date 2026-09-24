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
	"strings"
	"time"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
	"github.com/giraffesyo/ingot/tokenizer"
)

// Options configures one generation: text-to-image, or image editing when
// Images are given.
type Options struct {
	Prompt string
	// Images are condition images for editing (up to 10; the last one sets
	// the default output aspect). Each is resized to about Resolution² pixels
	// at its aspect ratio.
	Images []image.Image
	// Resolution is the pipeline's output_resolution: the side of the square
	// area outputs and condition images target. Default 1024.
	Resolution int
	Width      int // pixels, rounded down to a multiple of 32; default from Resolution
	Height     int
	Steps      int // default 40
	Seed       uint64

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

// The pipeline's chat templates (QwenImage21Pipeline.prompt_template_t2i /
// _ti2i): condition images precede the prompt as "<imageN>" +
// vision-start + image-pad (expanded to the image's token count by the
// processor) + vision-end, joined by spaces.
const (
	sysPrompt    = "Comprehend and analyze the provided prompt."
	sysMessage   = "<|im_start|>system\n" + sysPrompt + "<|im_end|>\n"
	t2iPrefix    = sysMessage + "<|im_start|>user\n"
	t2iSuffix    = "<|im_end|>\n<|im_start|>assistant\n"
	imagePad     = "<|image_pad|>"
	imageTokenID = 151655
	maxImages    = 10
)

// condition is one prepared condition image.
type condition struct {
	img    *image.NRGBA // resized to its condition size
	gh, gw int          // vision patch grid (16 px)
}

// promptText builds the chat-templated prompt, expanding each image's pad
// token to its merged-token count.
func promptText(prompt string, conds []condition) string {
	if len(conds) == 0 {
		return t2iPrefix + prompt + t2iSuffix
	}
	var b strings.Builder
	b.WriteString(t2iPrefix)
	for i, c := range conds {
		if i > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprintf(&b, "<image%d><|vision_start|>%s<|vision_end|>", i+1,
			strings.Repeat(imagePad, c.gh*c.gw/(visMerge*visMerge)))
	}
	b.WriteString(prompt)
	b.WriteString(t2iSuffix)
	return b.String()
}

// Generate runs text-to-image from the Hugging Face snapshot dir. The stages
// run one after another and each releases its weights before the next loads
// (text encoder ~30 GB packed, DiT ~28.5 GB), so peak memory is one model.
// Sampling follows the reference pipeline: no CFG (true_cfg_scale = 1, the
// model's intended use), flow-matching Euler, prefix-KV reuse.
func Generate(dir string, opt Options) (*Result, error) {
	if opt.Resolution == 0 {
		opt.Resolution = 1024
	}
	if len(opt.Images) > maxImages {
		return nil, fmt.Errorf("qwenimage: %d condition images (max %d)", len(opt.Images), maxImages)
	}
	var conds []condition
	for _, img := range opt.Images {
		b := img.Bounds()
		cw, ch := ConditionSize(opt.Resolution, b.Dx(), b.Dy())
		r := ResizeLanczos(img, cw, ch)
		conds = append(conds, condition{img: r, gh: ch / visPatch, gw: cw / visPatch})
	}
	if opt.Width == 0 && opt.Height == 0 && len(conds) > 0 {
		last := conds[len(conds)-1].img.Rect
		opt.Width, opt.Height = ConditionSize(opt.Resolution, last.Dx(), last.Dy())
	}
	if opt.Width == 0 {
		opt.Width = opt.Resolution
	}
	if opt.Height == 0 {
		opt.Height = opt.Resolution
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
	ids, err := tok.Encode(promptText(prompt, conds))
	if err != nil {
		return nil, err
	}
	sysIDs, err := tok.Encode(sysMessage)
	if err != nil {
		return nil, err
	}
	drop := len(sysIDs)
	stage("tokenize", t0)

	gpu := opt.Device == "gpu" || ((opt.Device == "" || opt.Device == "auto") && metalAvailable())
	if !gpu && opt.Device != "" && opt.Device != "auto" && opt.Device != "cpu" {
		return nil, fmt.Errorf("qwenimage: unknown device %q (cpu, gpu, auto)", opt.Device)
	}
	if len(conds) > 0 && !gpu {
		return nil, fmt.Errorf("qwenimage: image editing needs the GPU text encoder (device cpu)")
	}

	// 2. Text encoder (+ vision tower) → conditioning hidden states.
	t0 = time.Now()
	embeds, imgPad, err := encodeText(dir, ids, drop, conds, gpu)
	if err != nil {
		return nil, err
	}
	release()
	stage("text encoder", t0)

	// 3. Condition latents (VAE encoder).
	var cond *tensor.Tensor
	if len(conds) > 0 {
		t0 = time.Now()
		if cond, err = encodeConditions(dir, conds); err != nil {
			return nil, err
		}
		release()
		stage("vae encode", t0)
	}

	// 4. DiT: prefix pass once, then the target steps.
	t0 = time.Now()
	x, err := denoise(dir, embeds, imgPad, cond, conds, lh, lw, opt, logf)
	if err != nil {
		return nil, err
	}
	release()
	stage("denoise", t0)

	// 5. VAE decode.
	t0 = time.Now()
	img, err := decode(dir, x, lh, lw, gpu)
	if err != nil {
		return nil, err
	}
	release()
	stage("decode", t0)
	res.Float = img
	res.Image = toImage(img)
	return res, nil
}

// encodeText returns the conditioning hidden states [T-drop, hidden] and,
// with condition images, which of their rows are image tokens.
func encodeText(dir string, ids []int64, drop int, conds []condition, gpu bool) (*tensor.Tensor, []bool, error) {
	tdir := filepath.Join(dir, "text_encoder")
	cfg, err := LoadTextConfig(tdir)
	if err != nil {
		return nil, nil, err
	}
	set, err := safetensors.OpenDir(tdir)
	if err != nil {
		return nil, nil, err
	}
	defer set.Close()
	imgPad := make([]bool, len(ids)-drop)
	if gpu {
		in := TextInputs{IDs: ids}
		if len(conds) > 0 {
			if in, err = visionInputs(tdir, set, ids, conds); err != nil {
				return nil, nil, err
			}
			for _, p := range in.ImagePositions {
				if p < drop {
					return nil, nil, fmt.Errorf("qwenimage: image token inside the system prompt")
				}
				imgPad[p-drop] = true
			}
		}
		m, err := NewMetalTextEncoder(cfg, set)
		if err != nil {
			return nil, nil, err
		}
		defer m.Close()
		h, err := m.EncodeMM(in, drop, cfg.NumHiddenLayers)
		return h, imgPad, err
	}
	te, err := NewTextEncoder(cfg, set)
	if err != nil {
		return nil, nil, err
	}
	s, err := compile(te.Build(len(ids), drop, cfg.NumHiddenLayers))
	if err != nil {
		return nil, nil, err
	}
	x, err := te.Embed(ids)
	if err != nil {
		return nil, nil, err
	}
	out, err := s.Run(map[string]*tensor.Tensor{"x": x})
	if err != nil {
		return nil, nil, err
	}
	return out["hidden"].Clone(), imgPad, nil
}

// visionInputs runs the vision tower over each condition image and
// assembles the language model's multimodal inputs.
func visionInputs(tdir string, set *safetensors.Set, ids []int64, conds []condition) (TextInputs, error) {
	in := TextInputs{IDs: ids}
	vcfg, err := LoadVisionConfig(tdir)
	if err != nil {
		return in, err
	}
	var merged []*tensor.Tensor
	deep := make([][]*tensor.Tensor, len(vcfg.DeepstackVisualIndexes))
	var grids [][2]int
	mv, err := NewMetalVision(vcfg, set)
	if err != nil {
		return in, err
	}
	defer mv.Close()
	for _, c := range conds {
		pix, gh, gw, err := VisionPatches(c.img)
		if err != nil {
			return in, err
		}
		m, d, err := mv.Encode(pix, gh, gw)
		if err != nil {
			return in, err
		}
		merged = append(merged, m)
		for i := range deep {
			deep[i] = append(deep[i], d[i])
		}
		grids = append(grids, [2]int{gh / visMerge, gw / visMerge})
	}
	in.ImageEmbeds = concatRows(merged)
	for i := range deep {
		in.Deepstack = append(in.Deepstack, concatRows(deep[i]))
	}
	for i, id := range ids {
		if id == imageTokenID {
			in.ImagePositions = append(in.ImagePositions, i)
		}
	}
	if in.ImageEmbeds.Shape()[0] != len(in.ImagePositions) {
		return in, fmt.Errorf("qwenimage: %d image tokens in the prompt, %d from the vision tower",
			len(in.ImagePositions), in.ImageEmbeds.Shape()[0])
	}
	in.Positions, err = MRoPEPositions(ids, imageTokenID, grids)
	return in, err
}

// encodeConditions VAE-encodes each condition image into normalised,
// packed latents, concatenated [Σ h·w, z_dim].
func encodeConditions(dir string, conds []condition) (*tensor.Tensor, error) {
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
	var parts []*tensor.Tensor
	if metalAvailable() { // edit mode runs on the GPU
		v, err := NewMetalVAE(cfg, set)
		if err != nil {
			return nil, err
		}
		defer v.Close()
		for _, c := range conds {
			z, err := v.Encode(VAEPixels(c.img))
			if err != nil {
				return nil, err
			}
			parts = append(parts, z)
		}
		return concatRows(parts), nil
	}
	for _, c := range conds {
		px := VAEPixels(c.img)
		ps := px.Shape()
		s, err := compile(BuildVAEEncoder(cfg, set, ps[1], ps[2]))
		if err != nil {
			return nil, err
		}
		out, err := s.Run(map[string]*tensor.Tensor{"x": px.Reshape(1, ps[0], ps[1], ps[2])})
		if err != nil {
			return nil, err
		}
		parts = append(parts, PackLatents(cfg, out["z"]))
	}
	return concatRows(parts), nil
}

// concatRows stacks 2-D tensors along rows.
func concatRows(ts []*tensor.Tensor) *tensor.Tensor {
	rows, cols := 0, ts[0].Shape()[1]
	for _, t := range ts {
		rows += t.Shape()[0]
	}
	out := tensor.New(tensor.F32, rows, cols)
	off := 0
	for _, t := range ts {
		off += copy(out.F32()[off:], t.F32())
	}
	return out
}

func denoise(dir string, embeds *tensor.Tensor, imgPad []bool, cond *tensor.Tensor, conds []condition, lh, lw int,
	opt Options, logf func(string, ...any)) (*tensor.Tensor, error) {
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
	copy(mask, imgPad)
	for i := txt; i < len(mask); i++ {
		mask[i] = true
	}
	var shapes [][3]int
	for _, c := range conds {
		shapes = append(shapes, [3]int{1, c.gh, c.gw}) // latent grid = vision patch grid (both 16 px)
	}
	shapes = append(shapes, [3]int{1, lh, lw})
	l, err := NewDiTLayout(cfg, mask, nil, shapes)
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
	if gpu := opt.Device == "gpu" || ((opt.Device == "" || opt.Device == "auto") && metalAvailable()); gpu {
		return denoiseMetal(cfg, set, l, embeds, cond, x, sched, opt.Steps, opt.Fast, logf)
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
	kv, err := pre.Run(l.PrefixFeeds(embeds, cond))
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
func denoiseMetal(cfg DiTConfig, set *safetensors.Set, l *DiTLayout, embeds, cond, x *tensor.Tensor, sched Schedule, steps int,
	fast bool, logf func(string, ...any)) (*tensor.Tensor, error) {
	t0 := time.Now()
	m, err := NewMetalDiT(cfg, set, l, cfg.NumLayers)
	if err != nil {
		return nil, err
	}
	defer m.Close()
	m.Fast = fast
	if err := m.Prefix(embeds, cond); err != nil {
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

func decode(dir string, x *tensor.Tensor, lh, lw int, gpu bool) (*tensor.Tensor, error) {
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
	if gpu {
		v, err := NewMetalVAE(cfg, set)
		if err != nil {
			return nil, err
		}
		defer v.Close()
		z := x.Clone() // packed [h·w, C] is NHWC; denormalise per channel
		c := cfg.ZDim
		for i, val := range z.F32() {
			z.F32()[i] = val*cfg.LatentsStd[i%c] + cfg.LatentsMean[i%c]
		}
		return v.Decode(z, lh, lw)
	}
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
