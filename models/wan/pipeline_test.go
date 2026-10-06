package wan

import (
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// TestPipelineTiny replays the reference image-to-video run (tiny
// weights, real tokenizer and scheduler): the prompt and negative prompt
// through the text encoder, the image through the VAE encoder, four
// guided UniPC steps from the reference's noise, the chunked decode.
func TestPipelineTiny(t *testing.T) {
	dir := tinyDir(t)
	c := loadRef(t, "pipeline")
	p, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(filepath.Join(refDir, c.Meta["image"].(string)))
	if err != nil {
		t.Fatal(err)
	}
	img, err := png.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	hw := c.Meta["hw"].([]any)
	H, W := int(hw[0].(float64)), int(hw[1].(float64))

	texts, err := p.EncodePrompts("cpu", c.Meta["prompt"].(string), c.Meta["negative"].(string))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "prompt_embeds", texts[0].F32(), c.tensor(t, "prompt_embeds").F32(), 1e-5)
	compare(t, "negative_embeds", texts[1].F32(), c.tensor(t, "negative_embeds").F32(), 1e-5)

	steps := int(c.metaFloat("steps"))
	for _, dev := range devices() {
		t.Run(dev, func(t *testing.T) {
			res, err := p.Run(Options{
				Image: img, Prompt: c.Meta["prompt"].(string), Negative: c.Meta["negative"].(string),
				Width: W, Height: H, Frames: int(c.metaFloat("frames")), Steps: steps,
				Guidance: c.metaFloat("guidance"), Latents: c.tensor(t, "latents0"), Device: dev, TextDevice: "cpu",
				Log: t.Logf,
				Step: func(i int, l *tensor.Tensor) {
					compare(t, "step latents", l.F32(), c.tensor(t, "latents"+string(rune('1'+i))).F32(), 1)
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			// The reference's last-step latents precede the first-frame
			// replacement: compare the generated frames.
			want := c.tensor(t, "latents"+string(rune('0'+steps)))
			ls := want.Shape()
			per := ls[2] * ls[3]
			var got, ref []float32
			for ch := range ls[0] {
				for fr := 1; fr < ls[1]; fr++ {
					o := (ch*ls[1] + fr) * per
					got = append(got, res.Latents.F32()[o:o+per]...)
					ref = append(ref, want.F32()[o:o+per]...)
				}
			}
			compare(t, "latents", got, ref, 1e-4)
			// Video: ours [F, 3, H, W] in [-1, 1]; the reference [F, H, W, 3] in [0, 1].
			v := c.tensor(t, "video")
			vs := v.Shape()
			out := make([]float32, v.Numel())
			for fr := range vs[0] {
				for y := range vs[1] {
					for x := range vs[2] {
						for ch := range 3 {
							u := res.Video.F32()[((fr*3+ch)*vs[1]+y)*vs[2]+x]/2 + 0.5
							out[((fr*vs[1]+y)*vs[2]+x)*3+ch] = min(max(u, 0), 1)
						}
					}
				}
			}
			compare(t, "video", out, v.F32(), 1e-4)
		})
	}
}
