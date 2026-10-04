package qwenimage

import (
	"fmt"
	"path/filepath"
	"reflect"
	"time"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/safetensors"
	"github.com/giraffesyo/ingot/tensor"
)

// batchCache owns one stage's weights. It keeps only the most recent CPU
// sessions / DiT layout, bounding packed weights and GPU scratch as shapes change.
// The single-image pipeline does not reference or link this cache.
type batchCache struct {
	ctx                               interface{ Err() error }
	sets                              map[string]*safetensors.Set
	text                              *MetalTextEncoder
	vae                               *MetalVAE
	dit                               *MetalDiT
	layout                            *DiTLayout
	textSession, pre, target, decoder *graph.Session
	tokens, drop, lh, lw              int
}

func (c *batchCache) open(dir string) (*safetensors.Set, error) {
	if s := c.sets[dir]; s != nil {
		return s, nil
	}
	s, err := safetensors.OpenDir(dir)
	if err != nil {
		return nil, err
	}
	if c.sets == nil {
		c.sets = make(map[string]*safetensors.Set)
	}
	c.sets[dir] = s
	return s, nil
}

func (c *batchCache) close() {
	if c.text != nil {
		c.text.Close()
	}
	if c.vae != nil {
		c.vae.Close()
	}
	if c.dit != nil {
		c.dit.Close()
	}
	for _, s := range c.sets {
		_ = s.Close()
	}
	*c = batchCache{}
	release()
}

func (c *batchCache) encodeText(dir string, ids []int64, drop int, conds []condition, gpu bool) (*tensor.Tensor, []bool, error) {
	tdir := filepath.Join(dir, "text_encoder")
	cfg, err := LoadTextConfig(tdir)
	if err != nil {
		return nil, nil, err
	}
	set, err := c.open(tdir)
	if err != nil {
		return nil, nil, err
	}
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
		if c.text == nil {
			c.text, err = NewMetalTextEncoder(cfg, set)
		}
		if err != nil {
			return nil, nil, err
		}
		m := c.text
		h, err := m.EncodeMM(in, drop, cfg.NumHiddenLayers)
		return h, imgPad, err
	}
	te, err := NewTextEncoder(cfg, set)
	if err != nil {
		return nil, nil, err
	}
	if c.textSession == nil || c.tokens != len(ids) || c.drop != drop {
		c.textSession = nil
		release()
		c.textSession, err = compile(te.Build(len(ids), drop, cfg.NumHiddenLayers))
		c.tokens, c.drop = len(ids), drop
	}
	s := c.textSession
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
	defer s.Release(out)
	return out["hidden"].Clone(), imgPad, nil
}

func (c *batchCache) encodeConditions(dir string, conds []condition) (*tensor.Tensor, error) {
	vdir := filepath.Join(dir, "vae")
	cfg, err := LoadVAEConfig(vdir)
	if err != nil {
		return nil, err
	}
	set, err := c.open(vdir)
	if err != nil {
		return nil, err
	}
	var parts []*tensor.Tensor
	if metalAvailable() { // edit mode runs on the GPU
		if c.vae == nil {
			c.vae, err = NewMetalVAE(cfg, set)
		}
		if err != nil {
			return nil, err
		}
		v := c.vae
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

func (c *batchCache) denoise(dir string, embeds *tensor.Tensor, imgPad []bool, cond *tensor.Tensor, conds []condition, lh, lw int,
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
	set, err := c.open(ddir)
	if err != nil {
		return nil, err
	}
	l, err := ditLayout(cfg, imgPad, conds, lh, lw)
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
		return c.denoiseMetal(cfg, set, l, embeds, cond, x, sched, opt.Steps, opt.Fast, logf)
	}
	if opt.Device != "" && opt.Device != "auto" && opt.Device != "cpu" {
		return nil, fmt.Errorf("qwenimage: unknown device %q (cpu, gpu, auto)", opt.Device)
	}
	if c.pre == nil || !reflect.DeepEqual(c.layout, l) {
		c.pre, c.target = nil, nil
		release()
		c.pre, err = compile(BuildDiTPrefix(cfg, set, l, cfg.NumLayers, false))
		if err != nil {
			return nil, err
		}
		c.target, err = compile(BuildDiTTarget(cfg, set, l, cfg.NumLayers))
		if err != nil {
			return nil, err
		}
		c.layout = l
	}
	pre, tgt := c.pre, c.target

	t0 := time.Now()
	kv, err := pre.Run(l.PrefixFeeds(embeds, cond))
	if err != nil {
		return nil, err
	}
	logf("  prefix      %8.2fs (%d tokens)", time.Since(t0).Seconds(), l.Prefix)
	defer pre.Release(kv)
	for i := range opt.Steps {
		if err := batchContextError(c.ctx); err != nil {
			return nil, err
		}
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

func (c *batchCache) denoiseMetal(cfg DiTConfig, set *safetensors.Set, l *DiTLayout, embeds, cond, x *tensor.Tensor, sched Schedule, steps int,
	fast bool, logf func(string, ...any)) (*tensor.Tensor, error) {
	t0 := time.Now()
	if c.dit != nil && (!reflect.DeepEqual(c.layout, l) || c.dit.Fast != fast) {
		c.dit.Close()
		c.dit = nil
		release()
	}
	if c.dit == nil {
		var err error
		c.dit, err = NewMetalDiT(cfg, set, l, cfg.NumLayers)
		if err != nil {
			return nil, err
		}
		c.layout = l
	}
	m := c.dit
	m.Fast = fast
	if err := m.Prefix(embeds, cond); err != nil {
		return nil, err
	}
	logf("  prefix      %8.2fs (%d tokens, gpu)", time.Since(t0).Seconds(), l.Prefix)
	for i := range steps {
		if err := batchContextError(c.ctx); err != nil {
			return nil, err
		}
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

func (c *batchCache) decode(dir string, x *tensor.Tensor, lh, lw int, gpu bool) (*tensor.Tensor, error) {
	vdir := filepath.Join(dir, "vae")
	cfg, err := LoadVAEConfig(vdir)
	if err != nil {
		return nil, err
	}
	set, err := c.open(vdir)
	if err != nil {
		return nil, err
	}
	if gpu {
		if c.vae == nil {
			c.vae, err = NewMetalVAE(cfg, set)
		}
		if err != nil {
			return nil, err
		}
		v := c.vae
		z := x.Clone() // packed [h·w, C] is NHWC; denormalise per channel
		c := cfg.ZDim
		for i, val := range z.F32() {
			z.F32()[i] = val*cfg.LatentsStd[i%c] + cfg.LatentsMean[i%c]
		}
		return v.Decode(z, lh, lw)
	}
	if c.decoder == nil || c.lh != lh || c.lw != lw {
		c.decoder = nil
		release()
		c.decoder, err = compile(BuildVAEDecoder(cfg, set, lh, lw))
		c.lh, c.lw = lh, lw
	}
	s := c.decoder
	if err != nil {
		return nil, err
	}
	out, err := s.Run(map[string]*tensor.Tensor{"z": UnpackLatents(cfg, x, lh, lw)})
	if err != nil {
		return nil, err
	}
	defer s.Release(out)
	img := out["image"]
	return img.Reshape(img.Shape()[1:]...).Clone(), nil
}
