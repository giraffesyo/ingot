package qwenimage

import (
	"fmt"
	"path/filepath"
	"time"

	"github.com/giraffesyo/ingot/tensor"
	"github.com/giraffesyo/ingot/tokenizer"
)

// BatchRequest is one independently cancellable image in a stage-batched run.
// Cancellation is observed between stages and denoising steps; a running GPU
// command or graph must finish before its resources can be safely released.
type BatchRequest struct {
	// Context accepts a context.Context. Only Err is needed because cancellation
	// is polled between stages/steps. Keeping this interface narrow avoids adding
	// the context package's initialization to executables that never use batching.
	// Nil means the request is not cancellable.
	Context interface{ Err() error }
	Options Options
}

// GenerateBatch processes up to eight requests stage by stage: encode every
// prompt, encode references, denoise every image, then decode every image.
// Each stage opens its checkpoints once; compatible layouts reuse their GPU
// objects or compiled CPU sessions. Only one model stage is resident at a time.
// Requests must resolve to the same compute device. New arrivals belong in the
// next batch, so a continuing queue cannot starve images waiting for decode.
//
// done is called exactly once per request, synchronously, and must not retain
// Result.Float if bounded memory is required. A failed/cancelled request does not
// abort its siblings. Callers that do not use batching do not link this code.
func GenerateBatch(dir string, requests []BatchRequest, done func(int, *Result, error)) {
	jobs := make([]*batchJob, len(requests))
	for i, req := range requests {
		jobs[i] = &batchJob{ctx: req.Context, opt: req.Options, result: &Result{Stages: map[string]time.Duration{}}}
	}
	finish := func(i int, err error) {
		j := jobs[i]
		if j.finished {
			return
		}
		j.finished = true
		var result *Result
		if err == nil {
			result = j.result
		}
		done(i, result, err)
		j.conds, j.embeds, j.cond, j.latents, j.result = nil, nil, nil, nil, nil
		j.opt.Images, j.opt.Latents = nil, nil
	}
	if len(jobs) > 8 {
		for i := range jobs {
			finish(i, fmt.Errorf("qwenimage: batch limit is 8 images"))
		}
		return
	}
	if len(jobs) == 0 {
		return
	}
	tok, err := tokenizer.Load(filepath.Join(dir, "processor", "tokenizer.json"))
	if err != nil {
		for i := range jobs {
			finish(i, err)
		}
		return
	}
	var device *bool
	for i, j := range jobs {
		if err := j.prepare(dir, tok); err != nil {
			finish(i, err)
			continue
		}
		if device != nil && *device != j.gpu {
			finish(i, fmt.Errorf("qwenimage: all batch requests must use the same compute device"))
			continue
		}
		device = &j.gpu
	}
	runBatchStages(jobs, finish, func(stage string, visit func(func(*batchJob) error)) {
		cache := &batchCache{}
		defer cache.close()
		visit(func(j *batchJob) (err error) {
			cache.ctx = j.ctx
			switch stage {
			case "text encoder":
				j.embeds, j.imgPad, err = cache.encodeText(dir, j.ids, j.drop, j.conds, j.gpu)
			case "vae encode":
				if len(j.conds) != 0 {
					j.cond, err = cache.encodeConditions(dir, j.conds)
				}
			case "denoise":
				j.latents, err = cache.denoise(dir, j.embeds, j.imgPad, j.cond, j.conds, j.lh, j.lw, j.opt, j.log)
				j.embeds, j.cond = nil, nil
				if err == nil && j.opt.SaveLatents != "" {
					err = SaveLatents(j.opt.SaveLatents, j.latents, j.lh, j.lw)
					if err == nil {
						j.log("latents saved to %s", j.opt.SaveLatents)
					}
				}
			case "decode":
				j.result.Float, err = cache.decode(dir, j.latents, j.lh, j.lw, j.gpu)
				if err == nil {
					j.result.Image = toImage(j.result.Float)
				}
			}
			return err
		})
	})
}

type batchJob struct {
	ctx                   interface{ Err() error }
	opt                   Options
	gpu, finished         bool
	ids                   []int64
	drop, lh, lw          int
	conds                 []condition
	imgPad                []bool
	embeds, cond, latents *tensor.Tensor
	result                *Result
}

func (j *batchJob) log(format string, args ...any) {
	if j.opt.Log != nil {
		j.opt.Log(format, args...)
	}
}

func batchContextError(ctx interface{ Err() error }) error {
	if ctx != nil {
		return ctx.Err()
	}
	return nil
}

// Separating scheduling from kernels makes stage order, resource lifetime,
// failure isolation, and cancellation testable without the full checkpoints.
func runBatchStages(jobs []*batchJob, finish func(int, error), withStage func(string, func(func(*batchJob) error))) {
	for _, name := range []string{"text encoder", "vae encode", "denoise", "decode"} {
		withStage(name, func(run func(*batchJob) error) {
			for i, j := range jobs {
				if j.finished {
					continue
				}
				if err := batchContextError(j.ctx); err != nil {
					finish(i, err)
					continue
				}
				if name == "vae encode" && len(j.conds) == 0 {
					continue
				}
				j.log("stage: %s", name)
				t0 := time.Now()
				err := run(j)
				if err == nil {
					err = batchContextError(j.ctx)
				}
				j.result.Stages[name] = time.Since(t0)
				j.log("%-13s %8.2fs", name, time.Since(t0).Seconds())
				if err != nil || name == "decode" {
					finish(i, err)
				}
			}
		})
	}
}

func (j *batchJob) prepare(dir string, tok *tokenizer.Tokenizer) error {
	if err := batchContextError(j.ctx); err != nil {
		return err
	}
	opt := &j.opt
	if opt.Resolution == 0 {
		opt.Resolution = 1024
	}
	if opt.Steps == 0 {
		opt.Steps = 40
	}
	if opt.Resolution < 32 || opt.Resolution > 2048 || opt.Width < 0 || opt.Width > 2048 || opt.Height < 0 || opt.Height > 2048 || opt.Steps < 1 || opt.Steps > 100 {
		return fmt.Errorf("qwenimage: invalid batch dimensions or steps")
	}
	if len(opt.Images) > maxImages {
		return fmt.Errorf("qwenimage: at most %d reference images", maxImages)
	}
	j.gpu = opt.Device == "gpu" || ((opt.Device == "" || opt.Device == "auto") && metalAvailable())
	if opt.Device != "" && opt.Device != "auto" && opt.Device != "cpu" && opt.Device != "gpu" {
		return fmt.Errorf("qwenimage: unknown device %q", opt.Device)
	}
	if len(opt.Images) > 0 && !j.gpu {
		return fmt.Errorf("qwenimage: image editing requires Metal")
	}
	for _, img := range opt.Images {
		if img == nil || img.Bounds().Empty() {
			return fmt.Errorf("qwenimage: empty reference image")
		}
		b := img.Bounds()
		cw, ch := ConditionSize(opt.Resolution, b.Dx(), b.Dy())
		j.conds = append(j.conds, condition{img: ResizeLanczos(img, cw, ch), gh: ch / visPatch, gw: cw / visPatch})
	}
	if opt.Width == 0 && opt.Height == 0 && len(j.conds) > 0 {
		b := j.conds[len(j.conds)-1].img.Rect
		opt.Width, opt.Height = ConditionSize(opt.Resolution, b.Dx(), b.Dy())
	}
	if opt.Width == 0 {
		opt.Width = opt.Resolution
	}
	if opt.Height == 0 {
		opt.Height = opt.Resolution
	}
	if opt.Width < 32 || opt.Height < 32 {
		return fmt.Errorf("qwenimage: dimensions must be at least 32 pixels")
	}
	j.lh, j.lw = opt.Height/32*2, opt.Width/32*2
	prompt := opt.Prompt
	if prompt == "" {
		prompt = " "
	}
	var err error
	j.ids, err = tok.Encode(promptText(prompt, j.conds))
	if err != nil {
		return err
	}
	sys, err := tok.Encode(sysMessage)
	if err != nil {
		return err
	}
	j.drop = len(sys)
	if j.gpu {
		return preflight(dir, j.ids, j.drop, j.conds, j.lh, j.lw, j.opt, j.log)
	}
	return nil
}
