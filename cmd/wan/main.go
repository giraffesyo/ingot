// Command wan animates an image (or generates from text alone) with Wan
// 2.2 TI2V-5B, running entirely on the ingot runtime (pure Go; the GPU via
// Metal when available).
//
//	wan -image hero.png -prompt "her hair and cloak blow in a strong wind" -out frames/
//	wan -image hero.png -prompt "..." -frames 49 -size 832x480 -steps 30 -video clip.avi
//	wan -image hero.png -prompt "..." -keep body-mask.png   # masked pixels stay the input's
//
// Frames are written as PNGs (frames/0000.png, ...) as they are decoded;
// -video also writes them as a Motion-JPEG AVI. -keep takes a grey mask
// (white = keep, any size, stretched to the output): those pixels are the
// input image's in every frame, exactly — the model conditions on the
// first frame only, so this is a composite after decoding, and parts of
// the subject that move under a kept region will show a seam.
//
// The weights are the Hugging Face snapshot Wan-AI/Wan2.2-TI2V-5B-Diffusers
// (about 35 GB), found in the HF cache or given with -model. Stages load
// one at a time: text encoder, image encoder, transformer, decoder.
package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/giraffesyo/ingot/models/wan"
)

func main() {
	in := flag.String("image", "", "first frame (PNG/JPEG); omit for text-to-video")
	prompt := flag.String("prompt", "", "what happens in the clip (required)")
	negative := flag.String("negative", "", `negative prompt (default: Wan's; "-" for none)`)
	out := flag.String("out", "frames", "directory for the PNG frames")
	video := flag.String("video", "", "also write the clip as a Motion-JPEG AVI")
	fps := flag.Int("fps", 24, "frame rate of -video")
	keep := flag.String("keep", "", "grey mask image: white pixels keep the input image exactly")
	size := flag.String("size", "", "WxH, multiples of 32 (default: the image's aspect at 1280x704's area)")
	frames := flag.Int("frames", 121, "frames, 4k+1 (121 = 5 s at 24 fps)")
	steps := flag.Int("steps", 50, "denoising steps")
	guidance := flag.Float64("guidance", 5, "classifier-free guidance scale (1 = no negative pass, twice as fast)")
	shift := flag.Float64("shift", 0, "flow shift (default: the scheduler's, 5)")
	seed := flag.Uint64("seed", 42, "noise seed")
	device := flag.String("device", "auto", "transformer and VAE: auto, gpu, gpu-bf16 (fast), cpu")
	textDevice := flag.String("text-device", "cpu", "text encoder: cpu (bf16 in place) or gpu (widens to f32, 22 GB)")
	model := flag.String("model", "", "snapshot directory (default: "+wan.Repo+" in the HF cache)")
	flag.Parse()

	if *prompt == "" {
		fmt.Fprintln(os.Stderr, "wan: -prompt is required")
		flag.Usage()
		os.Exit(2)
	}
	o := wan.Options{Prompt: *prompt, Negative: *negative, Frames: *frames, Steps: *steps, Guidance: *guidance,
		Shift: *shift, Seed: *seed, Device: *device, TextDevice: *textDevice}
	if *size != "" {
		if _, err := fmt.Sscanf(strings.ToLower(*size), "%dx%d", &o.Width, &o.Height); err != nil {
			fmt.Fprintf(os.Stderr, "wan: -size %q: want WxH\n", *size)
			os.Exit(2)
		}
	}
	if err := run(*in, *out, *video, *keep, *model, *fps, o); err != nil {
		msg := err.Error()
		if !strings.HasPrefix(msg, "wan: ") {
			msg = "wan: " + msg
		}
		fmt.Fprintln(os.Stderr, msg)
		os.Exit(1)
	}
}

func readImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return img, nil
}

func run(in, out, video, keep, model string, fps int, o wan.Options) error {
	var err error
	if in != "" {
		if o.Image, err = readImage(in); err != nil {
			return err
		}
	}
	var mask image.Image
	if keep != "" {
		if o.Image == nil {
			return fmt.Errorf("-keep needs -image")
		}
		if mask, err = readImage(keep); err != nil {
			return err
		}
	}
	if model == "" {
		if model, err = wan.FindSnapshot(wan.Repo); err != nil {
			return err
		}
	}
	p, err := wan.Open(model)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	start := time.Now()
	o.Log = func(format string, args ...any) {
		fmt.Fprintf(os.Stderr, "[%6.1fs] %s\n", time.Since(start).Seconds(), fmt.Sprintf(format, args...))
	}
	var avi *aviWriter
	if video != "" {
		if avi, err = newAVI(video, fps); err != nil {
			return err
		}
	}
	// The kept region composites against the input at the output size,
	// which the pipeline produces; Fit is deterministic, so compute it here.
	var src *image.NRGBA
	n := 0
	o.Frame = func(i int, img *image.NRGBA) error {
		if mask != nil {
			if src == nil {
				src = wan.Fit(o.Image, img.Rect.Dx(), img.Rect.Dy())
			}
			wan.KeepRegion(img, src, mask)
		}
		f, err := os.Create(filepath.Join(out, fmt.Sprintf("%04d.png", i)))
		if err != nil {
			return err
		}
		if err := png.Encode(f, img); err != nil {
			f.Close()
			return err
		}
		n++
		if avi != nil {
			if err := avi.add(img); err != nil {
				f.Close()
				return err
			}
		}
		return f.Close()
	}
	res, err := p.Run(o)
	if err != nil {
		return err
	}
	if avi != nil {
		if err := avi.close(); err != nil {
			return err
		}
	}
	var step time.Duration
	for _, d := range res.StepTimes {
		step += d
	}
	if len(res.StepTimes) > 0 {
		step /= time.Duration(len(res.StepTimes))
	}
	fmt.Fprintf(os.Stderr, "wrote %d frames to %s in %.0fs (%.1f s/step)\n", n, out, time.Since(start).Seconds(), step.Seconds())
	return nil
}
