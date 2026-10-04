// Command trellis2 generates a textured 3-D mesh from one image with
// TRELLIS.2, running entirely on the ingot runtime (pure Go; the GPU via
// Metal when available).
//
//	trellis2 -image chair.png -out chair.glb
//	trellis2 -image chair.png -type 1024_cascade -seed 7
//
// The image should be a cut-out: a PNG whose background is transparent.
// An image without transparency is used whole, background included.
//
// The weights are Hugging Face snapshots — microsoft/TRELLIS.2-4B, plus the
// repositories its pipeline.json names (the DINOv3 image encoder and the
// sparse-structure decoder) — found in the HF cache or given with -model
// and -dep. Stages load one at a time.
//
// The mesh is written as binary glTF with per-vertex colour: one vertex
// per surface voxel, not simplified or UV-unwrapped.
package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"strings"
	"time"

	"github.com/giraffesyo/ingot/models/trellis2"
)

func main() {
	in := flag.String("image", "", "input image (PNG/JPEG; required)")
	out := flag.String("out", "out.glb", "output mesh (binary glTF)")
	typ := flag.String("type", "512", "pipeline: 512, 1024, 1024_cascade or 1536_cascade")
	seed := flag.Uint64("seed", 42, "noise seed")
	steps := flag.Int("steps", 0, "sampling steps per stage (default: the pipeline's, 12)")
	device := flag.String("device", "auto", "device for the transformers: auto (GPU when available), gpu, gpu-bf16, cpu")
	model := flag.String("model", "", "TRELLIS.2-4B snapshot directory (default: the HF cache)")
	var deps depList
	flag.Var(&deps, "dep", "snapshot of another repository the pipeline needs, as org/name=dir (default: the HF cache; repeatable)")
	flag.Parse()

	if *in == "" {
		fmt.Fprintln(os.Stderr, "trellis2: -image is required")
		flag.Usage()
		os.Exit(2)
	}
	if err := run(*in, *out, *model, deps, trellis2.Options{Type: *typ, Seed: *seed, Steps: *steps}, *device); err != nil {
		msg := err.Error()
		if !strings.HasPrefix(msg, "trellis2: ") {
			msg = "trellis2: " + msg
		}
		fmt.Fprintln(os.Stderr, msg)
		os.Exit(1)
	}
}

func run(in, out, model string, deps depList, opts trellis2.Options, device string) error {
	f, err := os.Open(in)
	if err != nil {
		return err
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		return fmt.Errorf("%s: %w", in, err)
	}
	if model == "" {
		if model, err = trellis2.FindSnapshot("microsoft/TRELLIS.2-4B"); err != nil {
			return err
		}
	}
	repos, err := trellis2.PipelineRepositories(model)
	if err != nil {
		return err
	}
	dirs := map[string]string(deps)
	if dirs == nil {
		dirs = map[string]string{}
	}
	for _, repo := range repos {
		if _, ok := dirs[repo]; !ok {
			if dirs[repo], err = trellis2.FindSnapshot(repo); err != nil {
				return err
			}
		}
	}
	p, err := trellis2.Open(model, dirs)
	if err != nil {
		return err
	}
	start := time.Now()
	p.Device = device
	p.Log = func(format string, args ...any) {
		fmt.Fprintf(os.Stderr, "[%5.0fs] %s\n", time.Since(start).Seconds(), fmt.Sprintf(format, args...))
	}
	subject, err := trellis2.Preprocess(img)
	if err != nil {
		return err
	}
	mesh, err := p.Run(subject, opts)
	if err != nil {
		return err
	}
	w, err := os.Create(out)
	if err != nil {
		return err
	}
	if err := mesh.WriteGLB(w); err != nil {
		w.Close()
		return err
	}
	if err := w.Close(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "wrote %s (%d vertices, %d faces) in %.0fs\n", out, len(mesh.Vertices), len(mesh.Faces), time.Since(start).Seconds())
	return nil
}

// depList collects -dep org/name=dir flags.
type depList map[string]string

func (d *depList) String() string { return fmt.Sprint(map[string]string(*d)) }

func (d *depList) Set(v string) error {
	repo, dir, ok := strings.Cut(v, "=")
	if !ok || repo == "" || dir == "" {
		return fmt.Errorf("want org/name=dir, got %q", v)
	}
	if *d == nil {
		*d = depList{}
	}
	(*d)[repo] = dir
	return nil
}
