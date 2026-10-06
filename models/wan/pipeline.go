package wan

import (
	"fmt"
	"image"
	"math/rand/v2"
	"path/filepath"
	"time"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
	"github.com/giraffesyo/ingot/tokenizer"
)

// DefaultNegative is the original Wan 2.2 repository's default negative
// prompt (it is Chinese; it names over-saturation, static frames, blur,
// subtitles, bad hands and faces, cluttered backgrounds and the like).
const DefaultNegative = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，" +
	"丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，" +
	"杂乱的背景，三条腿，背景人很多，倒着走"

// Options configures one generation.
type Options struct {
	// Image is the first frame for image-to-video; nil generates from the
	// prompt alone. It is scaled to cover Width×Height and centre-cropped.
	Image    image.Image
	Prompt   string
	Negative string // default DefaultNegative; "-" for none
	// Width and Height are the output size in pixels, multiples of 32;
	// zero picks the image's aspect at DefaultArea (1280×704 without one).
	Width, Height int
	Frames        int     // default 121 (5 s at 24 fps); rounded to 4k+1
	Steps         int     // default 50
	Guidance      float64 // classifier-free guidance scale, default 5; ≤ 1 runs no negative pass
	Shift         float64 // flow shift, default the scheduler's (5)
	Seed          uint64

	// Device runs the transformer and the VAE: "cpu", "gpu", "gpu-bf16"
	// or "auto" (graph.CompileOn; default "auto"). TextDevice runs the
	// text encoder, default "cpu" (bf16 weights in place; on the GPU they
	// are widened to f32, 22 GB for umT5-XXL).
	Device, TextDevice string

	// Latents replaces the seeded noise: [48, T, h, w] (parity tests
	// replay the reference's torch noise).
	Latents *tensor.Tensor
	// Step receives the latents after each denoising step (the
	// scheduler's output, before the first frame is re-imposed).
	Step func(i int, latents *tensor.Tensor)
	// Frame receives each decoded frame as soon as its chunk is decoded.
	Frame func(i int, img *image.NRGBA) error
	// Log receives one line per stage with timings; nil is silent.
	Log func(format string, args ...any)
}

// Result is a generation's output.
type Result struct {
	Video   *tensor.Tensor // [F, 3, H, W] in [-1, 1]
	Latents *tensor.Tensor // denoised, normalised [48, T, h, w]
	Input   *image.NRGBA   // the conditioning image at the output size, nil for text-to-video
	Stages  map[string]time.Duration
	// StepTimes are the wall times of each denoising step (all its
	// transformer evaluations).
	StepTimes []time.Duration
}

// Pipeline is an opened Wan 2.2 TI2V checkpoint directory (the Diffusers
// layout: tokenizer/, text_encoder/, transformer/, vae/, scheduler/).
type Pipeline struct {
	dir   string
	tok   *tokenizer.Unigram
	text  TextConfig
	dit   DiTConfig
	vae   VAEConfig
	sched SchedulerConfig
}

// Open reads dir's configs and tokenizer; weights are mapped per stage.
func Open(dir string) (*Pipeline, error) {
	p := &Pipeline{dir: dir}
	var err error
	if p.tok, err = tokenizer.LoadUnigram(filepath.Join(dir, "tokenizer", "tokenizer.json")); err != nil {
		return nil, err
	}
	if p.text, err = LoadTextConfig(filepath.Join(dir, "text_encoder")); err != nil {
		return nil, err
	}
	if p.dit, err = LoadDiTConfig(filepath.Join(dir, "transformer")); err != nil {
		return nil, err
	}
	if p.vae, err = LoadVAEConfig(filepath.Join(dir, "vae")); err != nil {
		return nil, err
	}
	if p.sched, err = LoadSchedulerConfig(filepath.Join(dir, "scheduler")); err != nil {
		return nil, err
	}
	if p.dit.InChannels != p.vae.ZDim || p.dit.TextDim != p.text.DModel {
		return nil, fmt.Errorf("wan: transformer (in %d, text %d) does not match VAE z_dim %d / text encoder d_model %d",
			p.dit.InChannels, p.dit.TextDim, p.vae.ZDim, p.text.DModel)
	}
	return p, nil
}

// VAEConfig returns the checkpoint's VAE configuration.
func (p *Pipeline) VAEConfig() VAEConfig { return p.vae }

func (o *Options) defaults(p *Pipeline) error {
	if o.Negative == "" {
		o.Negative = DefaultNegative
	}
	if o.Negative == "-" {
		o.Negative = ""
	}
	if o.Width == 0 || o.Height == 0 {
		w, h := 1280, 704
		if o.Image != nil {
			b := o.Image.Bounds()
			w, h = SizeFor(b.Dx(), b.Dy(), DefaultArea)
		}
		o.Width, o.Height = w, h
	}
	s := p.vae.Spatial() * p.dit.PatchSize[1]
	if o.Width%s != 0 || o.Height%s != 0 {
		return fmt.Errorf("wan: size %dx%d is not a multiple of %d", o.Width, o.Height, s)
	}
	if o.Frames == 0 {
		o.Frames = 121
	}
	t := p.vae.temporal()
	o.Frames = max(1, (o.Frames-1)/t*t+1)
	if o.Steps == 0 {
		o.Steps = 50
	}
	if o.Guidance == 0 {
		o.Guidance = 5
	}
	if o.Device == "" {
		o.Device = "auto"
	}
	if o.TextDevice == "" {
		o.TextDevice = "cpu"
	}
	return nil
}

func (p *Pipeline) logf(o *Options, format string, args ...any) {
	if o.Log != nil {
		o.Log(format, args...)
	}
}

// closeRunner releases a GPU session's device memory.
func closeRunner(r graph.Runner) {
	if c, ok := r.(interface{ Close() }); ok {
		c.Close()
	}
}

// EncodePrompts runs the text encoder over each prompt and returns the
// transformer's text input per prompt: [MaxTextTokens, d_model], the
// prompt's embeddings followed by zero rows.
func (p *Pipeline) EncodePrompts(device string, prompts ...string) ([]*tensor.Tensor, error) {
	set, err := safetensors.OpenDir(filepath.Join(p.dir, "text_encoder"))
	if err != nil {
		return nil, err
	}
	defer set.Close()
	te, err := NewTextEncoder(p.text, set, device != "cpu")
	if err != nil {
		return nil, err
	}
	var out []*tensor.Tensor
	for _, s := range prompts {
		ids := p.tok.Encode(CleanPrompt(s), true)
		if len(ids) > MaxTextTokens { // truncation keeps the closing </s>
			ids = append(ids[:MaxTextTokens-1], ids[len(ids)-1])
		}
		x, err := te.Embed(ids)
		if err != nil {
			return nil, err
		}
		g, err := te.Build(len(ids), p.text.NumLayers)
		if err != nil {
			return nil, err
		}
		r, err := graph.CompileOn(g, device)
		if err != nil {
			return nil, err
		}
		res, err := r.Run(map[string]*tensor.Tensor{"x": x})
		if err != nil {
			closeRunner(r)
			return nil, fmt.Errorf("wan: text encoder: %w", err)
		}
		emb := tensor.New(tensor.F32, MaxTextTokens, p.text.DModel)
		copy(emb.F32(), res["hidden"].F32())
		closeRunner(r)
		out = append(out, emb)
	}
	return out, nil
}

// EncodeImage runs the VAE encoder on an image at the output size and
// returns its normalised latent frame [z_dim, h, w].
func (p *Pipeline) EncodeImage(img *image.NRGBA, device string) (*tensor.Tensor, error) {
	set, err := safetensors.OpenDir(filepath.Join(p.dir, "vae"))
	if err != nil {
		return nil, err
	}
	defer set.Close()
	H, W := img.Rect.Dy(), img.Rect.Dx()
	g, err := BuildVAEEncoder(p.vae, set, H, W)
	if err != nil {
		return nil, err
	}
	r, err := graph.CompileOn(g, device)
	if err != nil {
		return nil, err
	}
	defer closeRunner(r)
	res, err := r.Run(map[string]*tensor.Tensor{"x": PatchifyImage(Pixels(img))})
	if err != nil {
		return nil, fmt.Errorf("wan: VAE encode: %w", err)
	}
	z := res["z"]
	s := z.Shape()
	C, hw := s[1], s[2]*s[3]
	out := tensor.New(tensor.F32, C, s[2], s[3])
	for c := range C {
		for i := range hw {
			out.F32()[c*hw+i] = (z.F32()[c*hw+i] - p.vae.LatentsMean[c]) / p.vae.LatentsStd[c]
		}
	}
	return out, nil
}

// Run generates a clip.
func (p *Pipeline) Run(o Options) (*Result, error) {
	if err := o.defaults(p); err != nil {
		return nil, err
	}
	res := &Result{Stages: map[string]time.Duration{}}
	sp := p.vae.Spatial()
	grid := Grid{F: p.vae.LatentFrames(o.Frames), H: o.Height / sp, W: o.Width / sp}
	C := p.vae.ZDim
	cfgOn := o.Guidance > 1
	p.logf(&o, "wan: %dx%d, %d frames (%d latent frames, %d tokens), %d steps, guidance %g, seed %d",
		o.Width, o.Height, o.Frames, grid.F, grid.Tokens(), o.Steps, o.Guidance, o.Seed)

	// 1. Text.
	t0 := time.Now()
	prompts := []string{o.Prompt}
	if cfgOn {
		prompts = append(prompts, o.Negative)
	}
	texts, err := p.EncodePrompts(o.TextDevice, prompts...)
	if err != nil {
		return nil, err
	}
	res.Stages["text"] = time.Since(t0)
	p.logf(&o, "wan: text encoder %.1fs", res.Stages["text"].Seconds())

	// 2. The image's latent (image-to-video).
	var cond *tensor.Tensor
	if o.Image != nil {
		t0 = time.Now()
		res.Input = Fit(o.Image, o.Width, o.Height)
		if cond, err = p.EncodeImage(res.Input, o.Device); err != nil {
			return nil, err
		}
		res.Stages["image"] = time.Since(t0)
		p.logf(&o, "wan: image encoder %.1fs", res.Stages["image"].Seconds())
	}

	// 3. Denoise.
	latents := o.Latents
	if latents == nil {
		latents = tensor.New(tensor.F32, C, grid.F, grid.H, grid.W)
		r := rand.New(rand.NewPCG(o.Seed, o.Seed^0x77616e32))
		for i := range latents.F32() {
			latents.F32()[i] = float32(r.NormFloat64())
		}
	} else if s := latents.Shape(); len(s) != 4 || s[0] != C || s[1] != grid.F || s[2] != grid.H || s[3] != grid.W {
		return nil, fmt.Errorf("wan: latents %v, want [%d %d %d %d]", s, C, grid.F, grid.H, grid.W)
	} else {
		latents = latents.Clone()
	}
	if err := p.denoise(&o, res, grid, texts, cond, latents); err != nil {
		return nil, err
	}
	if cond != nil {
		setFirstFrame(latents, cond, grid)
	}
	res.Latents = latents

	// 4. Decode.
	t0 = time.Now()
	video, err := p.Decode(latents, o.Device, o.Frame)
	if err != nil {
		return nil, err
	}
	res.Video = video
	res.Stages["decode"] = time.Since(t0)
	p.logf(&o, "wan: VAE decode %.1fs", res.Stages["decode"].Seconds())
	return res, nil
}

// setFirstFrame writes the latent frame cond [C, h, w] into latents [C, F,
// h, w] at frame 0.
func setFirstFrame(latents, cond *tensor.Tensor, g Grid) {
	hw := g.H * g.W
	for c := range cond.Shape()[0] {
		copy(latents.F32()[c*g.F*hw:c*g.F*hw+hw], cond.F32()[c*hw:(c+1)*hw])
	}
}

// denoise runs the sampling loop in place on latents.
func (p *Pipeline) denoise(o *Options, res *Result, grid Grid, texts []*tensor.Tensor, cond, latents *tensor.Tensor) error {
	t0 := time.Now()
	set, err := safetensors.OpenDir(filepath.Join(p.dir, "transformer"))
	if err != nil {
		return err
	}
	defer set.Close()
	widen := o.Device != "cpu"
	cg, err := BuildTextCond(p.dit, set, p.dit.NumLayers, widen)
	if err != nil {
		return err
	}
	cr, err := graph.CompileOn(cg, o.Device)
	if err != nil {
		return err
	}
	kvs := make([]map[string]*tensor.Tensor, len(texts))
	for i, t := range texts {
		kv, err := cr.Run(map[string]*tensor.Tensor{"text": t})
		if err != nil {
			closeRunner(cr)
			return fmt.Errorf("wan: text condition: %w", err)
		}
		kvs[i] = make(map[string]*tensor.Tensor, len(kv))
		for k, v := range kv {
			kvs[i][k] = v.Clone()
		}
		cr.Release(kv)
	}
	closeRunner(cr)

	S := 1
	if cond != nil {
		S = 2
	}
	g, err := BuildDiT(p.dit, set, grid, S, p.dit.NumLayers, widen)
	if err != nil {
		return err
	}
	r, err := graph.CompileOn(g, o.Device)
	if err != nil {
		return err
	}
	defer closeRunner(r)
	res.Stages["load"] = time.Since(t0)
	p.logf(o, "wan: transformer ready %.1fs", res.Stages["load"].Seconds())

	sched := p.sched
	if o.Shift != 0 {
		sched.FlowShift = o.Shift
	}
	u := NewUniPC(sched, o.Steps)
	t0 = time.Now()
	for i, t := range u.Timesteps {
		ts := time.Now()
		input := latents
		var steps Timesteps
		if cond != nil {
			input = latents.Clone()
			setFirstFrame(input, cond, grid)
			steps = FirstFrameFixed(grid, float32(t))
		} else {
			steps = Timesteps{Values: []float32{float32(t)}}
		}
		feeds := map[string]*tensor.Tensor{"x": Patchify(input, grid), "t": tensor.FromF32(steps.Values, len(steps.Values))}
		if steps.Seg != nil {
			feeds["seg"] = tensor.FromI64(steps.Seg, len(steps.Seg))
		}
		var preds [2][]float32
		for j, kv := range kvs {
			for k, v := range kv {
				feeds[k] = v
			}
			out, err := r.Run(feeds)
			if err != nil {
				return fmt.Errorf("wan: step %d: %w", i, err)
			}
			preds[j] = Unpatchify(out["v"], grid).F32()
			r.Release(out)
		}
		v := preds[0]
		if len(kvs) > 1 {
			gs := float32(o.Guidance)
			for k := range v {
				v[k] = preds[1][k] + gs*(v[k]-preds[1][k])
			}
		}
		copy(latents.F32(), u.Step(v, latents.F32()))
		if o.Step != nil {
			o.Step(i, latents)
		}
		res.StepTimes = append(res.StepTimes, time.Since(ts))
		p.logf(o, "wan: step %d/%d (t=%d) %.2fs", i+1, len(u.Timesteps), t, time.Since(ts).Seconds())
	}
	res.Stages["denoise"] = time.Since(t0)
	return nil
}

// Decode turns normalised latents [z_dim, T, h, w] into video frames [F,
// 3, H, W] in [-1, 1], chunk by chunk; frame (optional) receives each
// frame as 8-bit RGB as soon as it is decoded.
func (p *Pipeline) Decode(latents *tensor.Tensor, device string, frame func(int, *image.NRGBA) error) (*tensor.Tensor, error) {
	set, err := safetensors.OpenDir(filepath.Join(p.dir, "vae"))
	if err != nil {
		return nil, err
	}
	defer set.Close()
	s := latents.Shape()
	C, T, h, w := s[0], s[1], s[2], s[3]
	z := tensor.New(tensor.F32, C, T, h, w)
	n := T * h * w
	for c := range C {
		for i := range n {
			z.F32()[c*n+i] = latents.F32()[c*n+i]*p.vae.LatentsStd[c] + p.vae.LatentsMean[c]
		}
	}
	d, err := BuildVAEDecoder(p.vae, set, h, w)
	if err != nil {
		return nil, err
	}
	dc, err := NewDecoder(d, device)
	if err != nil {
		return nil, err
	}
	defer dc.Close()
	idx := 0
	var cb func(*tensor.Tensor) error
	if frame != nil {
		cb = func(f *tensor.Tensor) error {
			fs := f.Shape()
			per := fs[1] * fs[2] * fs[3]
			for k := range fs[0] {
				if err := frame(idx, FrameImage(f.F32()[k*per:(k+1)*per], fs[2], fs[3])); err != nil {
					return err
				}
				idx++
			}
			return nil
		}
	}
	return dc.Decode(z, cb)
}
