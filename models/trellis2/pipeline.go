package trellis2

import (
	"fmt"
	"image"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/sparse"
	"github.com/giraffesyo/ingot/tensor"
)

// Normalization is a latent's per-channel statistics.
type Normalization struct {
	Mean []float32 `json:"mean"`
	Std  []float32 `json:"std"`
}

type samplerConfig struct {
	Args struct {
		SigmaMin float64 `json:"sigma_min"`
	} `json:"args"`
	Params SamplerParams `json:"params"`
}

func (s samplerConfig) params() SamplerParams {
	p := s.Params
	p.SigmaMin = s.Args.SigmaMin
	return p
}

// pipelineConfig is pipeline.json's "args".
type pipelineConfig struct {
	Models           map[string]string `json:"models"`
	StructureSampler samplerConfig     `json:"sparse_structure_sampler"`
	ShapeSampler     samplerConfig     `json:"shape_slat_sampler"`
	TexSampler       samplerConfig     `json:"tex_slat_sampler"`
	ShapeNorm        Normalization     `json:"shape_slat_normalization"`
	TexNorm          Normalization     `json:"tex_slat_normalization"`
	ImageCond        struct {
		Args struct {
			ModelName string `json:"model_name"`
		} `json:"args"`
	} `json:"image_cond_model"`
}

// Pipeline is TRELLIS.2's image-to-3D pipeline over local checkpoints.
type Pipeline struct {
	cfg  pipelineConfig
	dir  string            // the TRELLIS.2 snapshot
	deps map[string]string // other repositories ("org/name") → snapshot

	// Upsampler is the NAF feature upsampler's checkpoint (safetensors,
	// from tools/export/naf_convert.py), needed by pipelines whose models
	// take projected image features (Pixal3D).
	Upsampler string
	// Device runs the models: "cpu", "gpu", "gpu-bf16" or "auto"
	// (graph.CompileOn). Default "auto".
	Device string
	// Log, if set, receives progress lines.
	Log func(format string, args ...any)
}

// Open reads dir/pipeline.json. deps maps the other Hugging Face
// repositories the pipeline names ("org/name") to their local snapshots:
// the image encoder, and the repository holding the sparse-structure
// decoder.
func Open(dir string, deps map[string]string) (*Pipeline, error) {
	var c struct {
		Name string         `json:"name"`
		Args pipelineConfig `json:"args"`
	}
	if err := readJSON(filepath.Join(dir, "pipeline.json"), &c); err != nil {
		return nil, err
	}
	if c.Name != "Trellis2ImageTo3DPipeline" {
		return nil, fmt.Errorf("trellis2: %s: pipeline %q not implemented", dir, c.Name)
	}
	p := &Pipeline{cfg: c.Args, dir: dir, deps: deps, Device: "auto"}
	for _, need := range p.Repositories() {
		if _, ok := deps[need]; !ok {
			return nil, fmt.Errorf("trellis2: pipeline needs repository %s; no local snapshot given", need)
		}
	}
	return p, nil
}

// PipelineRepositories reads dir/pipeline.json and lists the other
// repositories it names, for locating their snapshots before Open.
func PipelineRepositories(dir string) ([]string, error) {
	var c struct {
		Args pipelineConfig `json:"args"`
	}
	if err := readJSON(filepath.Join(dir, "pipeline.json"), &c); err != nil {
		return nil, err
	}
	return (&Pipeline{cfg: c.Args}).Repositories(), nil
}

// Repositories lists the other repositories the pipeline reads, sorted.
func (p *Pipeline) Repositories() []string {
	set := map[string]bool{p.cfg.ImageCond.Args.ModelName: true}
	for _, path := range p.cfg.Models {
		if repo, _, ok := splitRepo(path); ok {
			set[repo] = true
		}
	}
	out := make([]string, 0, len(set))
	for r := range set {
		out = append(out, r)
	}
	sort.Strings(out)
	return out
}

// splitRepo splits "org/name/ckpts/x" into its repository and the path
// within it; checkpoints of the pipeline's own repository are "ckpts/x".
func splitRepo(path string) (repo, rest string, ok bool) {
	parts := strings.Split(path, "/")
	if len(parts) < 4 || parts[0] == "ckpts" {
		return "", path, false
	}
	return parts[0] + "/" + parts[1], strings.Join(parts[2:], "/"), true
}

// checkpoint resolves a model's base path (no extension).
func (p *Pipeline) checkpoint(name string) (string, error) {
	path, ok := p.cfg.Models[name]
	if !ok {
		return "", fmt.Errorf("trellis2: pipeline has no model %q", name)
	}
	if repo, rest, ok := splitRepo(path); ok {
		return filepath.Join(p.deps[repo], filepath.FromSlash(rest)), nil
	}
	return filepath.Join(p.dir, filepath.FromSlash(path)), nil
}

func (p *Pipeline) logf(format string, args ...any) {
	if p.Log != nil {
		p.Log(format, args...)
	}
}

// Options selects what Run generates.
type Options struct {
	// Type is the pipeline variant: "512" (a 512³ voxel grid, the
	// cheapest), "1024", or the cascades "1024_cascade" and "1536_cascade"
	// (shape sampled at 512 then refined on a finer latent grid). Default
	// "512".
	Type string
	Seed uint64
	// Steps overrides every sampler's step count when positive.
	Steps int
	// MaxTokens caps a cascade's refined latent grid (default 49152
	// cells); the refinement resolution drops until it fits.
	MaxTokens int
	// FovX is the input photo's horizontal field of view in radians, for
	// pipelines with projected image features (default DefaultFovX).
	FovX float64
}

// Projected reports whether the pipeline's models take image features
// projected onto the voxel grid (Pixal3D) rather than plain image tokens.
// Such a pipeline needs Upsampler set, an image from PreprocessMargin(1.1),
// and runs only the cascade types.
func (p *Pipeline) Projected() (bool, error) {
	base, err := p.checkpoint("sparse_structure_flow_model")
	if err != nil {
		return false, err
	}
	cfg, err := LoadFlowConfig(base + ".json")
	return cfg.ProjChannels() > 0, err
}

// Run generates a textured mesh from a preprocessed image (Preprocess).
func (p *Pipeline) Run(img *image.NRGBA, o Options) (*Mesh, error) {
	if o.Type == "" {
		o.Type = "512"
	}
	if o.MaxTokens == 0 {
		o.MaxTokens = 49152
	}
	if proj, err := p.Projected(); err != nil {
		return nil, err
	} else if proj {
		return p.runProjected(img, o)
	}
	type variant struct {
		structure          int    // sparse-structure grid the latent cells come from
		shape, tex         string // flow models
		cond               int    // image size conditioning the final shape and texture
		lowShape           string // cascade: the first shape model, at 512
		target, resolution int
	}
	v, ok := map[string]variant{
		"512":          {structure: 32, shape: "shape_slat_flow_model_512", tex: "tex_slat_flow_model_512", cond: 512, resolution: 512},
		"1024":         {structure: 64, shape: "shape_slat_flow_model_1024", tex: "tex_slat_flow_model_1024", cond: 1024, resolution: 1024},
		"1024_cascade": {structure: 32, lowShape: "shape_slat_flow_model_512", shape: "shape_slat_flow_model_1024", tex: "tex_slat_flow_model_1024", cond: 1024, target: 1024},
		"1536_cascade": {structure: 32, lowShape: "shape_slat_flow_model_512", shape: "shape_slat_flow_model_1024", tex: "tex_slat_flow_model_1024", cond: 1024, target: 1536},
	}[o.Type]
	if !ok {
		return nil, fmt.Errorf("trellis2: unknown pipeline type %q (want 512, 1024, 1024_cascade or 1536_cascade)", o.Type)
	}
	rng := rand.New(rand.NewPCG(o.Seed, 0x7472656c6c697332))
	noise := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(rng.NormFloat64())
		}
		return out
	}
	steps := func(s SamplerParams) SamplerParams {
		if o.Steps > 0 {
			s.Steps = o.Steps
		}
		return s
	}

	cond512, err := p.encode(img, 512)
	if err != nil {
		return nil, err
	}
	cond := cond512
	if v.cond != 512 {
		if cond, err = p.encode(img, v.cond); err != nil {
			return nil, err
		}
	}

	// Stage 1: which cells of the latent grid the object occupies.
	cells, err := p.structure(cond512, nil, v.structure, noise, steps(p.cfg.StructureSampler.params()))
	if err != nil {
		return nil, err
	}
	p.logf("structure: %d occupied cells of %d³", len(cells), v.structure)

	// Stage 2: the shape latent over those cells.
	shapeSampler := steps(p.cfg.ShapeSampler.params())
	var shape []float32 // normalised, [cells, C]
	if v.lowShape == "" {
		if shape, err = p.latent(v.shape, "shape", cells, cond, nil, noise, shapeSampler); err != nil {
			return nil, err
		}
	} else {
		low, err := p.latent(v.lowShape, "shape (512)", cells, cond512, nil, noise, shapeSampler)
		if err != nil {
			return nil, err
		}
		if cells, v.resolution, err = p.refineCells(cells, low, v.target, o.MaxTokens, false); err != nil {
			return nil, err
		}
		p.logf("cascade: %d cells at resolution %d", len(cells), v.resolution)
		if shape, err = p.latent(v.shape, "shape", cells, cond, nil, noise, shapeSampler); err != nil {
			return nil, err
		}
	}

	// Stage 3: the texture latent, conditioned on the shape latent.
	tex, err := p.latent(v.tex, "texture", cells, cond, shape, noise, steps(p.cfg.TexSampler.params()))
	if err != nil {
		return nil, err
	}

	return p.decode(cells, shape, tex, v.resolution)
}

// runProjected is Run for a pipeline with projected image features: the
// cascade, each stage conditioned on the image's global tokens and on
// image features sampled where the stage's cells project into the image.
func (p *Pipeline) runProjected(img *image.NRGBA, o Options) (*Mesh, error) {
	target, ok := map[string]int{"": 1024, "1024_cascade": 1024, "1536_cascade": 1536}[o.Type]
	if !ok {
		return nil, fmt.Errorf("trellis2: a projected-feature pipeline runs only 1024_cascade or 1536_cascade, not %q", o.Type)
	}
	if p.Upsampler == "" {
		return nil, fmt.Errorf("trellis2: this pipeline needs the NAF upsampler checkpoint (Pipeline.Upsampler)")
	}
	fov := o.FovX
	if fov == 0 {
		fov = DefaultFovX
	}
	cam := CameraFromFov(fov, 1)
	p.logf("camera: field of view %.1f°, distance %.3f", fov*180/math.Pi, cam.Distance)
	rng := rand.New(rand.NewPCG(o.Seed, 0x7472656c6c697332))
	noise := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(rng.NormFloat64())
		}
		return out
	}
	steps := func(s SamplerParams) SamplerParams {
		if o.Steps > 0 {
			s.Steps = o.Steps
		}
		return s
	}

	// Stage 1: structure, from features projected onto the whole 16³ grid.
	glob, patches, err := p.encodeSplit(img, 512)
	if err != nil {
		return nil, err
	}
	base, err := p.checkpoint("sparse_structure_flow_model")
	if err != nil {
		return nil, err
	}
	scfg, err := LoadFlowConfig(base + ".json")
	if err != nil {
		return nil, err
	}
	proj, err := ProjectFeatures(cam, GridCoords(scfg.Resolution), scfg.Resolution, 512, patches, nil)
	if err != nil {
		return nil, err
	}
	cells, err := p.structure(glob, proj, 32, noise, steps(p.cfg.StructureSampler.params()))
	if err != nil {
		return nil, err
	}
	p.logf("structure: %d occupied cells of 32³", len(cells))

	// Stage 2: the shape latent at 512, features upsampled to 512².
	shapeSampler := steps(p.cfg.ShapeSampler.params())
	if proj, err = p.projected(img, cam, cells, 32, 512, 512, patches); err != nil {
		return nil, err
	}
	low, _, err := p.sample("shape_slat_flow_model_512", "shape (512)", cells, glob, proj, nil, noise, shapeSampler)
	if err != nil {
		return nil, err
	}

	// Stage 3: refine the cells, then the shape latent on the finer grid.
	cells, res, err := p.refineCells(cells, low, target, o.MaxTokens, true)
	if err != nil {
		return nil, err
	}
	p.logf("cascade: %d cells at resolution %d", len(cells), res)
	if glob, patches, err = p.encodeSplit(img, 1024); err != nil {
		return nil, err
	}
	if proj, err = p.projected(img, cam, cells, res/16, 1024, 512, patches); err != nil {
		return nil, err
	}
	shape, _, err := p.sample("shape_slat_flow_model_1024", "shape", cells, glob, proj, nil, noise, shapeSampler)
	if err != nil {
		return nil, err
	}

	// Stage 4: texture, features upsampled to the full 1024².
	if proj, err = p.projected(img, cam, cells, res/16, 1024, 1024, patches); err != nil {
		return nil, err
	}
	tex, _, err := p.sample("tex_slat_flow_model_1024", "texture", cells, glob, proj, shape, noise, steps(p.cfg.TexSampler.params()))
	if err != nil {
		return nil, err
	}
	return p.decode(cells, shape, tex, res)
}

// encodeSplit runs the image encoder and splits its tokens into the global
// ones (class and registers) and the patch map.
func (p *Pipeline) encodeSplit(img *image.NRGBA, size int) (glob, patches *tensor.Tensor, err error) {
	feats, err := p.encode(img, size)
	if err != nil {
		return nil, nil, err
	}
	n := (size / 16) * (size / 16)
	T, C := feats.Shape()[0], feats.Shape()[1]
	if T <= n {
		return nil, nil, fmt.Errorf("trellis2: %d image tokens leave no global tokens beside %d patches", T, n)
	}
	g := T - n
	glob, patches = tensor.New(tensor.F32, g, C), tensor.New(tensor.F32, n, C)
	copy(glob.F32(), feats.F32()[:g*C])
	copy(patches.F32(), feats.F32()[g*C:])
	return glob, patches, nil
}

// projected builds a stage's per-cell image features: the patch features
// and their NAF-upsampled version (to up×up, guided by the image at
// imageRes), both sampled at the cells' projections from a grid³ grid.
func (p *Pipeline) projected(img *image.NRGBA, cam Camera, cells []sparse.Coord, grid, imageRes, up int, patches *tensor.Tensor) (*tensor.Tensor, error) {
	start := time.Now()
	f, err := safetensors.Open(p.Upsampler)
	if err != nil {
		return nil, fmt.Errorf("trellis2: upsampler: %w", err)
	}
	defer f.Close()
	g, err := BuildNAFEncoder(f, imageRes, up)
	if err != nil {
		return nil, err
	}
	s, err := graph.Compile(g)
	if err != nil {
		return nil, fmt.Errorf("trellis2: upsampler: %w", err)
	}
	res, err := s.Run(map[string]*tensor.Tensor{"image": imageTensor(img, imageRes, false)})
	if err != nil {
		return nil, fmt.Errorf("trellis2: upsampler: %w", err)
	}
	hr, err := Upsample(res["q"], patches, up)
	if err != nil {
		return nil, err
	}
	out, err := ProjectFeatures(cam, cells, grid, imageRes, patches, hr)
	p.logf("projected features for %d cells (upsampled to %d², %.1fs)", len(cells), up, time.Since(start).Seconds())
	return out, err
}

// open maps a checkpoint and returns its base path.
func (p *Pipeline) open(name string) (string, *safetensors.File, error) {
	base, err := p.checkpoint(name)
	if err != nil {
		return "", nil, err
	}
	f, err := safetensors.Open(base + ".safetensors")
	if err != nil {
		return "", nil, fmt.Errorf("trellis2: %s: %w", name, err)
	}
	return base, f, nil
}

// encode runs the image encoder at size×size: [tokens, hidden].
func (p *Pipeline) encode(img *image.NRGBA, size int) (*tensor.Tensor, error) {
	start := time.Now()
	dir := p.deps[p.cfg.ImageCond.Args.ModelName]
	cfg, err := LoadDinoConfig(dir)
	if err != nil {
		return nil, err
	}
	f, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("trellis2: image encoder: %w", err)
	}
	defer f.Close()
	g, err := BuildDino(cfg, f, size)
	if err != nil {
		return nil, err
	}
	r, err := graph.CompileOn(g, p.Device)
	if err != nil {
		return nil, fmt.Errorf("trellis2: image encoder: %w", err)
	}
	res, err := r.Run(map[string]*tensor.Tensor{"image": ImageTensor(img, size)})
	if err != nil {
		return nil, fmt.Errorf("trellis2: image encoder: %w", err)
	}
	out := tensor.New(tensor.F32, res["features"].Shape()...)
	copy(out.F32(), res["features"].F32())
	p.logf("image features at %d px: %d tokens (%.1fs)", size, out.Shape()[0], time.Since(start).Seconds())
	return out, nil
}

// sample runs a flow model's sampler over coords from noise.
func (p *Pipeline) sample(name, what string, coords []sparse.Coord, cond, proj *tensor.Tensor, concat []float32,
	noise func(int) []float32, params SamplerParams) ([]float32, FlowConfig, error) {
	base, f, err := p.open(name)
	if err != nil {
		return nil, FlowConfig{}, err
	}
	defer f.Close()
	cfg, err := LoadFlowConfig(base + ".json")
	if err != nil {
		return nil, cfg, err
	}
	r, err := p.compileFlow(cfg, f, coords, cond.Shape()[0])
	if err != nil {
		return nil, cfg, fmt.Errorf("trellis2: %s: %w", name, err)
	}
	T := len(coords)
	state := cfg.InChannels // channels the sampler integrates
	if concat != nil {
		state = cfg.InChannels - len(concat)/T
	}
	// The condition's keys and values are the same in every evaluation:
	// once for the image features, once for the all-zeros negative.
	kv, err := FlowConditions(cfg, f, cfg.NumBlocks, cond, tensor.New(tensor.F32, cond.Shape()...))
	if err != nil {
		if s, ok := r.(interface{ Close() }); ok {
			s.Close()
		}
		return nil, cfg, fmt.Errorf("trellis2: %s: %w", name, err)
	}
	run := FlowVelocity(r, T, cfg.InChannels, kv[0], kv[1], proj)
	vel := run
	if concat != nil {
		// The condition's channels follow the state's in every row.
		extra := cfg.InChannels - state
		x := make([]float32, T*cfg.InChannels)
		vel = func(s []float32, t float32, positive bool) ([]float32, error) {
			for i := range T {
				copy(x[i*cfg.InChannels:], s[i*state:(i+1)*state])
				copy(x[i*cfg.InChannels+state:], concat[i*extra:(i+1)*extra])
			}
			return run(x, t, positive)
		}
	}
	start := time.Now()
	out, err := Sample(vel, noise(T*state), params, func(i, n int) {
		p.logf("%s: step %d/%d (%.0fs)", what, i, n, time.Since(start).Seconds())
	})
	if s, ok := r.(interface{ Close() }); ok {
		s.Close()
	}
	return out, cfg, err
}

// compileFlow builds and compiles a flow model for the pipeline's device.
// The GPU takes float32 weights; "auto" falls back to the CPU, reading the
// checkpoint's bf16 in place, where there is no GPU.
func (p *Pipeline) compileFlow(cfg FlowConfig, f *safetensors.File, coords []sparse.Coord, condTokens int) (graph.Runner, error) {
	device := p.Device
	if device == "" {
		device = "auto"
	}
	if device != "cpu" {
		g, err := BuildFlow(cfg, f, coords, condTokens, cfg.NumBlocks, true)
		if err != nil {
			return nil, err
		}
		gpu := device
		if gpu == "auto" {
			gpu = "gpu-bf16"
		}
		r, err := graph.CompileOn(g, gpu)
		if err == nil || device != "auto" {
			return r, err
		}
	}
	g, err := BuildFlow(cfg, f, coords, condTokens, cfg.NumBlocks, false)
	if err != nil {
		return nil, err
	}
	return graph.Compile(g)
}

// structure samples the sparse-structure latent and decodes it to the
// occupied cells of a res³ grid.
func (p *Pipeline) structure(cond, proj *tensor.Tensor, res int, noise func(int) []float32, params SamplerParams) ([]sparse.Coord, error) {
	base, err := p.checkpoint("sparse_structure_flow_model")
	if err != nil {
		return nil, err
	}
	fcfg, err := LoadFlowConfig(base + ".json")
	if err != nil {
		return nil, err
	}
	n := fcfg.Resolution
	z, _, err := p.sample("sparse_structure_flow_model", "structure", GridCoords(n), cond, proj, nil, noise, params)
	if err != nil {
		return nil, err
	}
	base, f, err := p.open("sparse_structure_decoder")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	cfg, err := LoadStructureDecoderConfig(base + ".json")
	if err != nil {
		return nil, err
	}
	g, err := BuildStructureDecoder(cfg, f, n)
	if err != nil {
		return nil, err
	}
	r, err := graph.CompileOn(g, "cpu") // 3-D convolution runs on the CPU
	if err != nil {
		return nil, fmt.Errorf("trellis2: structure decoder: %w", err)
	}
	// Tokens are [T, C]; the decoder takes [1, C, n, n, n].
	C := fcfg.InChannels
	zt := tensor.New(tensor.F32, 1, C, n, n, n)
	for tok := range n * n * n {
		for c := range C {
			zt.F32()[c*n*n*n+tok] = z[tok*C+c]
		}
	}
	out, err := r.Run(map[string]*tensor.Tensor{"z": zt})
	if err != nil {
		return nil, fmt.Errorf("trellis2: structure decoder: %w", err)
	}
	logits := out["logits"]
	cells, err := OccupiedCells(logits.F32(), logits.Shape()[2], res)
	if err != nil {
		return nil, err
	}
	if len(cells) == 0 {
		return nil, fmt.Errorf("trellis2: the structure stage produced an empty object")
	}
	return cells, nil
}

// latent samples a structured latent over cells, normalised (the flow
// model's own space). concat, when set, is another normalised latent over
// the same cells that conditions it.
func (p *Pipeline) latent(name, what string, cells []sparse.Coord, cond *tensor.Tensor, concat []float32,
	noise func(int) []float32, params SamplerParams) ([]float32, error) {
	out, _, err := p.sample(name, what, cells, cond, nil, concat, noise, params)
	return out, err
}

// denormalize maps a sampled latent to the decoder's space.
func denormalize(x []float32, n Normalization) (*tensor.Tensor, error) {
	C := len(n.Mean)
	if C == 0 || len(n.Std) != C || len(x)%C != 0 {
		return nil, fmt.Errorf("trellis2: latent of %d values does not match %d-channel normalisation", len(x), C)
	}
	t := tensor.New(tensor.F32, len(x)/C, C)
	for i, v := range x {
		t.F32()[i] = v*n.Std[i%C] + n.Mean[i%C]
	}
	return t, nil
}

// refineCells is the cascade's upsampling: the 512 shape latent is decoded
// far enough to know its surface cells at 512³, and those are regrouped
// onto the latent grid of the target resolution — as fine as fits within
// maxTokens.
//
// spread selects how a 512³ cell maps to the finer latent grid: false
// scales cell centres by grid/512 and truncates (TRELLIS.2); true scales
// them by (grid−1)/512 and rounds half to even (Pixal3D, whose projection
// grid puts cell 0 and cell grid−1 on the cube's faces).
func (p *Pipeline) refineCells(cells []sparse.Coord, low []float32, target, maxTokens int, spread bool) ([]sparse.Coord, int, error) {
	const lowRes = 512
	base, f, err := p.open("shape_slat_decoder")
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	cfg, err := LoadDecoderConfig(base + ".json")
	if err != nil {
		return nil, 0, err
	}
	latent, err := denormalize(low, p.cfg.ShapeNorm)
	if err != nil {
		return nil, 0, err
	}
	d, err := Decode(cfg, f, cells, latent, nil, "cpu", cfg.Levels()-1)
	if err != nil {
		return nil, 0, err
	}
	for res := target; ; res -= 128 {
		grid := res / 16
		seen := map[sparse.Coord]bool{}
		var out []sparse.Coord
		for _, c := range d.Coords {
			var q sparse.Coord
			for a := range 3 {
				if spread {
					q[a] = int32(math.RoundToEven(float64((float32(c[a]) + 0.5) / lowRes * float32(grid-1))))
				} else {
					q[a] = int32((float32(c[a]) + 0.5) / lowRes * float32(grid))
				}
			}
			if !seen[q] {
				seen[q] = true
				out = append(out, q)
			}
		}
		if len(out) < maxTokens || res <= 1024 {
			sort.Slice(out, func(i, j int) bool {
				a, b := out[i], out[j]
				if a[0] != b[0] {
					return a[0] < b[0]
				}
				if a[1] != b[1] {
					return a[1] < b[1]
				}
				return a[2] < b[2]
			})
			return out, res, nil
		}
	}
}

// decode runs both decoders and builds the mesh.
func (p *Pipeline) decode(cells []sparse.Coord, shape, tex []float32, resolution int) (*Mesh, error) {
	start := time.Now()
	base, f, err := p.open("shape_slat_decoder")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scfg, err := LoadDecoderConfig(base + ".json")
	if err != nil {
		return nil, err
	}
	latent, err := denormalize(shape, p.cfg.ShapeNorm)
	if err != nil {
		return nil, err
	}
	sd, err := Decode(scfg, f, cells, latent, nil, "cpu", 0)
	if err != nil {
		return nil, err
	}
	p.logf("shape decoded: %d cells at %d³ (%.1fs)", len(sd.Coords), resolution, time.Since(start).Seconds())
	mesh, err := ExtractMesh(sd.Coords, sd.Out.F32(), resolution, scfg.VoxelMargin)
	if err != nil {
		return nil, err
	}

	start = time.Now()
	base, tf, err := p.open("tex_slat_decoder")
	if err != nil {
		return nil, err
	}
	defer tf.Close()
	tcfg, err := LoadDecoderConfig(base + ".json")
	if err != nil {
		return nil, err
	}
	if latent, err = denormalize(tex, p.cfg.TexNorm); err != nil {
		return nil, err
	}
	td, err := Decode(tcfg, tf, cells, latent, sd.Guide(), "cpu", 0)
	if err != nil {
		return nil, err
	}
	p.logf("texture decoded (%.1fs)", time.Since(start).Seconds())
	if err := mesh.SetMaterial(td.Out.F32()); err != nil {
		return nil, err
	}
	pieces := mesh.OrientFaces()
	mesh.ComputeNormals()
	mesh.Compact()
	p.logf("mesh: %d vertices, %d faces, %d pieces", len(mesh.Vertices), len(mesh.Faces), pieces)
	return mesh, nil
}

// HubDir is the Hugging Face cache directory: $HF_HUB_CACHE, $HF_HOME/hub,
// or ~/.cache/huggingface/hub.
func HubDir() (string, error) {
	if hub := os.Getenv("HF_HUB_CACHE"); hub != "" {
		return hub, nil
	}
	if home := os.Getenv("HF_HOME"); home != "" {
		return filepath.Join(home, "hub"), nil
	}
	u, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("trellis2: %w", err)
	}
	return filepath.Join(u, ".cache", "huggingface", "hub"), nil
}

// FindSnapshot locates repo ("org/name") in the Hugging Face cache.
func FindSnapshot(repo string) (string, error) {
	hub, err := HubDir()
	if err != nil {
		return "", err
	}
	snaps, _ := filepath.Glob(filepath.Join(hub, "models--"+strings.ReplaceAll(repo, "/", "--"), "snapshots", "*"))
	for i := len(snaps) - 1; i >= 0; i-- {
		if entries, err := os.ReadDir(snaps[i]); err == nil && len(entries) > 0 {
			return snaps[i], nil
		}
	}
	return "", fmt.Errorf("trellis2: %s not found under %s (download it, or pass its directory)", repo, hub)
}
