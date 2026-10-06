// Package qwenimagecmd is ingot qwenimage: generate or edit images with
// Qwen-Image-2.1, running entirely on the ingot runtime (pure Go; the GPU via
// Metal when available).
//
//	ingot qwenimage --prompt "a red fox in the snow" --out fox.png
//	ingot qwenimage --prompt "..." --size 512 --steps 20 --seed 7
//	ingot qwenimage --prompt "give the fox a red scarf" --image fox.png --out scarf.png
//	ingot qwenimage --from-latents out.png.latents --out out.png   # retry a failed decode
//
// The denoised latents are written to <out>.latents before decoding and
// removed once the image is written, so a decode that fails (e.g. out of
// memory) can be retried with --from-latents instead of denoising again.
//
// The weights are the Hugging Face snapshot (Qwen/Qwen-Image-2.1), found in
// the HF cache or given with --model. Stages load one at a time; peak memory
// is ~36 GB.
package qwenimagecmd

import (
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"

	"github.com/giraffesyo/ingot/models/qwenimage"
	"github.com/giraffesyo/ingot/tensor"
)

// Command returns the qwenimage subcommand.
func Command() *cobra.Command {
	var (
		o                       qwenimage.Options
		images                  imageList
		out, model, fromLatents string
		keepLatents             bool
	)
	cmd := &cobra.Command{
		Use:   "qwenimage",
		Short: "Generate or edit images (Qwen-Image-2.1)",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			o.Images = images
			return run(o, out, model, fromLatents, keepLatents)
		},
	}
	f := cmd.Flags()
	f.StringVar(&o.Prompt, "prompt", "", "text prompt (required unless --from-latents)")
	f.StringVar(&out, "out", "out.png", "output PNG (RGBA)")
	f.Var(&images, "image", "condition image for editing (PNG/JPEG; repeat for up to 10)")
	f.IntVar(&o.Resolution, "size", 1024, "output resolution: side of the square area outputs and condition images target")
	f.IntVar(&o.Width, "width", 0, "output width in pixels (multiple of 32)")
	f.IntVar(&o.Height, "height", 0, "output height in pixels (multiple of 32)")
	f.IntVar(&o.Steps, "steps", 40, "denoising steps")
	f.Uint64Var(&o.Seed, "seed", 42, "noise seed")
	f.StringVar(&o.Device, "device", "auto", "DiT device: auto (GPU when available), gpu, cpu")
	f.BoolVar(&o.Fast, "fast", true, "GPU: bf16 GEMM inputs (~2x faster, outputs within ~1%); --fast=false for f32")
	f.StringVar(&model, "model", "", "Qwen-Image-2.1 snapshot directory (default: the HF cache)")
	f.StringVar(&fromLatents, "from-latents", "", "decode saved latents (a <out>.latents file) instead of generating")
	f.BoolVar(&keepLatents, "keep-latents", false, "keep <out>.latents after a successful decode")
	cmd.MarkFlagsOneRequired("prompt", "from-latents")
	return cmd
}

func run(o qwenimage.Options, out, dir, fromLatents string, keepLatents bool) error {
	if dir == "" {
		var err error
		if dir, err = findSnapshot(); err != nil {
			return err
		}
	}
	t0 := time.Now()
	latents := out + ".latents"
	var res *qwenimage.Result
	var err error
	if fromLatents != "" {
		var x *tensor.Tensor
		var lh, lw int
		if x, lh, lw, err = qwenimage.LoadLatents(fromLatents); err == nil {
			res, err = qwenimage.DecodeLatents(dir, x, lh, lw, o.Device)
		}
	} else {
		o.SaveLatents = latents
		o.Log = func(format string, args ...any) { fmt.Fprintf(os.Stderr, format+"\n", args...) }
		res, err = qwenimage.Generate(dir, o)
	}
	if err != nil {
		if st, serr := os.Stat(latents); serr == nil && fromLatents == "" && !st.ModTime().Before(t0) {
			fmt.Fprintf(os.Stderr, "qwenimage: the denoised latents are in %s; retry the decode with --from-latents %s\n", latents, latents)
		}
		return err
	}
	f, err := os.Create(out)
	if err == nil {
		err = errors.Join(png.Encode(f, res.Image), f.Close())
	}
	if err != nil {
		return err
	}
	if fromLatents == "" && !keepLatents {
		os.Remove(latents)
	}
	b := res.Image.Bounds()
	fmt.Printf("wrote %s (%dx%d) in %.1fs\n", out, b.Dx(), b.Dy(), time.Since(t0).Seconds())
	return nil
}

// imageList is the repeatable --image flag.
type imageList []image.Image

func (l *imageList) String() string {
	if len(*l) == 0 {
		return ""
	}
	return fmt.Sprintf("%d images", len(*l))
}

func (l *imageList) Type() string { return "path" }

func (l *imageList) Set(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	*l = append(*l, img)
	return nil
}

// findSnapshot locates Qwen/Qwen-Image-2.1 in the Hugging Face cache
// ($HF_HUB_CACHE, $HF_HOME/hub, or ~/.cache/huggingface/hub).
func findSnapshot() (string, error) {
	hub := os.Getenv("HF_HUB_CACHE")
	if hub == "" {
		if home := os.Getenv("HF_HOME"); home != "" {
			hub = filepath.Join(home, "hub")
		} else {
			u, err := os.UserHomeDir()
			if err != nil {
				return "", err
			}
			hub = filepath.Join(u, ".cache", "huggingface", "hub")
		}
	}
	snaps, _ := filepath.Glob(filepath.Join(hub, "models--Qwen--Qwen-Image-2.1", "snapshots", "*"))
	if len(snaps) == 0 {
		return "", fmt.Errorf("Qwen-Image-2.1 not found under %s (download it, or pass --model)", hub)
	}
	return snaps[len(snaps)-1], nil
}
