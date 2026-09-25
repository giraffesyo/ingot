//go:build darwin && arm64

package qwenimage

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/giraffesyo/ingot/kernels/metal"
)

// preflight checks, before any stage runs, that each GPU stage fits the
// device's working set (recommendedMaxWorkingSetSize): a stage that does
// not fails its command buffer at run time, which for the decoder would be
// after the whole denoise. Each estimate is the stage's mapped weights plus
// its scratch — exact for the DiT and the VAE (the same plans they
// allocate from), weights only for the text encoder, whose activations are
// small next to them.
func preflight(dir string, ids []int64, drop int, conds []condition, lh, lw int, opt Options, logf func(string, ...any)) error {
	dev, err := metal.Open()
	if err != nil {
		return err
	}
	limit := dev.MaxWorkingSet()
	if limit <= 0 {
		return nil
	}
	gb := func(n int) float64 { return float64(n) / (1 << 30) }
	check := func(stage string, need int) error {
		logf("  preflight   %-13s %5.1f GB of %.1f GB", stage, gb(need), gb(limit))
		if need > limit {
			return fmt.Errorf("qwenimage: %dx%d: the %s needs %.1f GB of GPU memory, more than %s's %.1f GB working set; use a smaller size",
				16*lw, 16*lh, stage, gb(need), dev.Name, gb(limit))
		}
		return nil
	}

	w, err := weightBytes(filepath.Join(dir, "text_encoder"))
	if err != nil {
		return err
	}
	if err := check("text encoder", w); err != nil {
		return err
	}

	ddir := filepath.Join(dir, "transformer")
	dcfg, err := LoadDiTConfig(ddir)
	if err != nil {
		return err
	}
	imgPad := make([]bool, len(ids)-drop)
	for i, id := range ids[drop:] {
		imgPad[i] = id == imageTokenID
	}
	l, err := ditLayout(dcfg, imgPad, conds, lh, lw)
	if err != nil {
		return err
	}
	if w, err = weightBytes(ddir); err != nil {
		return err
	}
	if err := check("transformer", w+metalDiTScratch(dcfg, l, dcfg.NumLayers, opt.Fast)); err != nil {
		return err
	}

	vdir := filepath.Join(dir, "vae")
	vcfg, err := LoadVAEConfig(vdir)
	if err != nil {
		return err
	}
	if w, err = weightBytes(vdir); err != nil {
		return err
	}
	v := &MetalVAE{cfg: vcfg}
	for _, c := range conds {
		b := c.img.Rect
		if err := check("vae encoder", w+v.EncodeBytes(b.Dy(), b.Dx())); err != nil {
			return err
		}
	}
	return check("vae decoder", w+v.DecodeBytes(lh, lw))
}

// weightBytes is the size of dir's safetensors files (the GPU stages map
// every shard they read from whole).
func weightBytes(dir string) (int, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		return 0, err
	}
	n := 0
	for _, f := range files {
		st, err := os.Stat(f)
		if err != nil {
			return 0, err
		}
		n += int(st.Size())
	}
	return n, nil
}
