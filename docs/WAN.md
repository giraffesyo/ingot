# Wan 2.2 TI2V-5B (image-to-video)

`models/wan` and `cmd/wan` run Wan 2.2 TI2V-5B — text- and image-to-video —
in pure Go on the ingot runtime, CPU or Metal GPU.

## Licence

Checked at the source (2026-10-05): the model card and `LICENSE.txt` of
[Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) and the
[Wan2.2 repository](https://github.com/Wan-Video/Wan2.2) are **Apache-2.0**,
with no claim on generated content (the card asks that use not break laws,
harm people, spread misinformation or target vulnerable groups). The text
encoder, [google/umt5-xxl](https://huggingface.co/google/umt5-xxl), is
Apache-2.0. The Diffusers-layout copy this package reads,
[Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers),
carries the same licence. Weights are not part of this repository.

## Use

```
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --include "*.json" --include "tokenizer/*" \
    --include "text_encoder/*" --include "transformer/*" --include "vae/*"
go run ./cmd/wan -image hero.png -prompt "her long hair and cloak blow in a strong wind" \
    -frames 49 -size 832x480 -steps 30 -device gpu-bf16 -out frames -video clip.avi
```

About 35 GB on disk: the transformer is stored in float32 (20 GB), umT5-XXL
in bf16 (11.4 GB), the VAE in float32 (2.8 GB).

Flags: `-image` (omit for text-to-video), `-prompt`, `-negative` (default:
the original repository's Chinese negative prompt; `-` for none), `-frames`
(4k+1; default 121 = 5 s at 24 fps), `-size WxH` (multiples of 32; default
the image's aspect at 1280×704's area, the trained resolution), `-steps`
(50), `-guidance` (5; 1 skips the negative pass and halves the time),
`-shift` (flow shift, 5), `-seed`, `-device` (auto / gpu / gpu-bf16 / cpu),
`-text-device` (cpu: bf16 weights in place; gpu widens them to 22 GB of
f32), `-out` (PNG frames, written as decoded), `-video` (Motion-JPEG AVI),
`-keep mask.png`.

### Keeping a region fixed

TI2V conditions on the input only through the first latent frame (held at
timestep 0); there is no per-pixel conditioning for later frames, and the
first latent frame cannot be repeated into later ones (the causal VAE
encodes the first frame alone and every later latent frame from four
frames). So holding a region pixel-identical — a face, a body, while only
the hair moves — is a composite after decoding: `-keep` takes a grey mask
(white keeps the input's pixels exactly, grey blends, any size, stretched
to the output) and `wan.KeepRegion` applies it to every frame. Prompt for
motion only where the mask is black; anything that moves under a kept
region shows a seam. Feather the mask edge a few pixels.

## Architecture (pinned from diffusers 0.41 / transformers 5 / Wan2.2)

- **Text**: umT5-XXL encoder (24 layers, d_model 4096, 64 heads of 64,
  gated-GELU d_ff 10240, a relative position bias per layer, attention not
  scaled by 1/√d). The prompt is cleaned as the pipeline does (ftfy's
  full-width→ASCII, quotes, ligatures, NFC; HTML unescaped twice;
  whitespace collapsed), tokenised by a Unigram SentencePiece model
  (`tokenizer.LoadUnigram`), cut to 512 tokens; the embeddings are
  zero-padded to 512 rows and the transformer cross-attends to all 512.
- **Transformer**: 30 blocks, 24 heads of 128 (dim 3072), FFN 14336; patch
  1×2×2 over 48 latent channels; interleaved 3-D RoPE (44 / 42 / 42 head
  channels for frame / row / column); RMS norm across heads on q and k;
  adaLN modulation from a shared time projection plus a per-block table.
  TI2V-5B's per-token timesteps (first latent frame at 0) are a few
  distinct values gathered by segment, so the time MLP runs on two rows,
  not on every token. Text keys/values are computed once per prompt.
- **VAE** (Wan 2.2, z_dim 48, 16×16 spatial / 4× temporal): pixels
  patchified 2×2 into 12 channels, residual down/up blocks with
  space-to-depth / duplicate-up shortcuts, causal 3-D convolutions. Ingot
  keeps activations as [frames, C, h, w], so a causal 3-D conv is a sum of
  2-D convs over shifted frames and every op runs on both executors. The
  image encoder is the one-frame first chunk; the decoder streams one latent
  frame per chunk through two graphs (first chunk; steady chunks carrying
  each causal conv's last two input frames), exactly the reference's
  feature-cache decode, so memory is one chunk's regardless of length.
  1 + 4(T−1) frames out of T latent frames.
- **Scheduler**: UniPC (bh2, order 2, x0 prediction, corrector) over flow
  sigmas shifted by 5; integer timesteps to the transformer.
- **Image-to-video**: the image is scaled to cover the output size and
  centre-cropped (Lanczos), encoded, normalised with latents_mean/std,
  written into latent frame 0 before every step and after the last.
  Classifier-free guidance: v = v_neg + g·(v_pos − v_neg).

## Validation

`tools/export/wan22_ref.py` (a venv with torch, transformers, diffusers ≥
0.35 and ftfy) builds seeded **tiny** random-weight instances of the very
diffusers / transformers modules — every structural choice of the real
configs, widths and depths shrunk, head dim 128 kept — writes them as a
Diffusers-layout model directory under `testdata/wan22/tiny`, and saves
reference activations; `WAN22_REAL=1` adds real-weight cases (umT5 on a
prompt, the transformer's first block, the VAE on a small image). All of it
is ignored, regenerated, never committed.

| stage | reference | max relative error |
|---|---|---|
| tokenizer + prompt cleaning | tokenizers / ftfy, 9 prompts incl. the Chinese negative | exact |
| umT5 encoder | UMT5EncoderModel | 2.5e-7 (CPU, GPU) |
| transformer, TI2V and scalar t | WanTransformer3DModel | 1.6e-6 (CPU, GPU f32); 3.1e-3 gpu-bf16 |
| VAE encoder | AutoencoderKLWan.encode | 5.7e-7 |
| VAE decoder, 4 latent → 13 frames | AutoencoderKLWan.decode | 2.5e-6 CPU, 2.7e-6 GPU |
| UniPC, 6 steps | UniPCMultistepScheduler | 2.2e-7 |
| pipeline, 64×96, 9 frames, 4 steps, CFG 5 | WanImageToVideoPipeline(expand_timesteps) | latents ≤ 5e-6 every step, video 1.2e-6 |
| real umT5-XXL, 24-token prompt | UMT5EncoderModel, f32 | 3.1e-6 CPU, 2.4e-6 GPU |
| real transformer, first block | WanTransformer3DModel, f32 | 7.8e-6 CPU, 8.4e-6 GPU, 6.5e-3 gpu-bf16 |
| real VAE encoder / 2-frame decode | AutoencoderKLWan, f32 | 1.5e-6 / 9.4e-6 |

The tokenizer follows tokenizer.json as shipped (what transformers 4.x and
the original Wan code used). transformers 5 rebuilds T5's backend and differs
only on text the prompt cleaning never produces and on spaces beside inline
special tokens.

## Performance

Apple Silicon (unified memory), `-device gpu-bf16`, CFG on (two transformer
evaluations a step), 49 frames, measured with other work on the machine
(niced, CPU workers capped at half the cores):

| size | tokens | text | first step | step (median) | VAE decode | 30 steps end to end |
|---|---|---|---|---|---|---|
| 480×640 | 3,900 | 6.4 s | 74 s | 6.9 s | 41 s | 377 s |
| 704×928 | 8,294 | 6.7 s | 58 s | 15.7 s | 139 s | 719 s |

"Transformer ready" (mapping the 20 GB float32 checkpoint, building the
graphs, the text keys/values) takes ~49 s; the first step adds the one-off
bf16 conversion of the weights and the GPU pipelines. A steady step at
3,900 tokens is ~45 TFLOP of work in 6.9 s, ~13 TFLOPS. The text encoder
runs on the CPU with its bf16 weights in place (on the GPU they must be
widened to 22 GB of f32: 72 s to load). Memory: the float32 transformer
stays a zero-copy mapping of the checkpoint, its bf16 copy (10 GB) lives
for the denoising stage, the text encoder and transformer are released
before the decoder runs, and the decoder works one latent frame (four
video frames) at a time.

Open: the VAE decode is the next lever (10 s per chunk at 704×928 — the
four-frame chunks at full resolution are bandwidth-bound convs plus the
cached-frame concatenations), a cheaper first step (convert the
transformer once into a bf16 file, or convert lazily on load), and a
peak-RSS measurement.
