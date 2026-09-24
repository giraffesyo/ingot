// Command qwenimage generates or edits images with Qwen-Image-2.1, running
// entirely on the ingot runtime (pure Go; the GPU via Metal when available).
//
//	qwenimage -prompt "a red fox in the snow" -out fox.png
//	qwenimage -prompt "..." -size 512 -steps 20 -seed 7
//	qwenimage -prompt "give the fox a red scarf" -image fox.png -out scarf.png
//
// The weights are the Hugging Face snapshot (Qwen/Qwen-Image-2.1), found in
// the HF cache or given with -model. Stages load one at a time; peak memory
// is ~36 GB.
package main

import (
	"errors"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"time"

	"github.com/giraffesyo/ingot/models/qwenimage"
)

func main() {
	prompt := flag.String("prompt", "", "text prompt (required)")
	out := flag.String("out", "out.png", "output PNG (RGBA)")
	var images imageList
	flag.Var(&images, "image", "condition image for editing (PNG/JPEG; repeat for up to 10)")
	size := flag.Int("size", 1024, "output resolution: side of the square area outputs and condition images target")
	width := flag.Int("width", 0, "output width in pixels (multiple of 32)")
	height := flag.Int("height", 0, "output height in pixels (multiple of 32)")
	steps := flag.Int("steps", 40, "denoising steps")
	seed := flag.Uint64("seed", 42, "noise seed")
	device := flag.String("device", "auto", "DiT device: auto (GPU when available), gpu, cpu")
	fast := flag.Bool("fast", true, "GPU: bf16 GEMM inputs (~2x faster, outputs within ~1%); -fast=false for f32")
	model := flag.String("model", "", "Qwen-Image-2.1 snapshot directory (default: the HF cache)")
	flag.Parse()

	if *prompt == "" {
		fmt.Fprintln(os.Stderr, "qwenimage: -prompt is required")
		flag.Usage()
		os.Exit(2)
	}
	dir := *model
	if dir == "" {
		var err error
		if dir, err = findSnapshot(); err != nil {
			fmt.Fprintln(os.Stderr, "qwenimage:", err)
			os.Exit(1)
		}
	}
	t0 := time.Now()
	res, err := qwenimage.Generate(dir, qwenimage.Options{
		Prompt: *prompt, Images: images, Resolution: *size, Width: *width, Height: *height,
		Steps: *steps, Seed: *seed, Device: *device, Fast: *fast,
		Log: func(format string, args ...any) { fmt.Fprintf(os.Stderr, format+"\n", args...) },
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "qwenimage:", err)
		os.Exit(1)
	}
	f, err := os.Create(*out)
	if err == nil {
		err = errors.Join(png.Encode(f, res.Image), f.Close())
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "qwenimage:", err)
		os.Exit(1)
	}
	b := res.Image.Bounds()
	fmt.Printf("wrote %s (%dx%d) in %.1fs\n", *out, b.Dx(), b.Dy(), time.Since(t0).Seconds())
}

// imageList is the repeatable -image flag.
type imageList []image.Image

func (l *imageList) String() string { return fmt.Sprintf("%d images", len(*l)) }

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
		return "", fmt.Errorf("Qwen-Image-2.1 not found under %s (download it, or pass -model)", hub)
	}
	return snaps[len(snaps)-1], nil
}
