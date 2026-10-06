"""Reference activations for Wan 2.2 TI2V-5B parity tests.

The real checkpoint is ~35 GB (a float32 5B transformer, the bf16 umT5-XXL
encoder, the VAE). Every stage is therefore checked twice over: against
seeded *tiny* random-weight instances of the very same diffusers /
transformers modules (the architecture, end to end, in minutes and a few
MB), and — when the snapshot's weights are present — against the real
weights on small inputs.

Tiny (always; written to testdata/wan22/tiny/ as a diffusers-layout model
directory plus references next to it):
  - tokenizer: umT5 ids from the real tokenizer.json for a set of prompts,
    and the pipeline's prompt cleaning (ftfy + html unescape + whitespace);
  - umt5: UMT5EncoderModel (2 layers, d_model 32) on real token ids;
  - dit: WanTransformer3DModel (2 blocks, 2 heads of 128) with the TI2V
    per-token timesteps (first latent frame at t=0) and with a scalar one;
  - vae_enc / vae_dec: AutoencoderKLWan in the Wan 2.2 layout
    (is_residual, patch_size 2, z_dim 48) — one image encoded, a 4-frame
    latent decoded chunk by chunk to 13 frames;
  - scheduler: UniPCMultistepScheduler from the real scheduler_config over
    seeded model outputs;
  - pipeline: WanImageToVideoPipeline(expand_timesteps=True) end to end —
    prompt and negative prompt, a 64x96 image, 9 frames, 4 steps, CFG 5 —
    with the initial noise saved so the Go pipeline can replay it.

Real (WAN22_REAL=1 and the weights in the HF cache): the text encoder on a
short prompt, the transformer truncated to its first block, the VAE
encoder on a 64x64 image and decoder on a 2-frame latent.

Needs torch, transformers and a diffusers with the Wan pipelines (>= 0.35),
plus ftfy (the pipeline's prompt cleaning imports it):

    HF_HUB_OFFLINE=1 python wan22_ref.py

Writes testdata/wan22/<name>.{in,out}.<i>.bin + <name>.json in the same
raw little-endian layout as zoo.py. Outputs and the tiny models are
regenerated, never committed.
"""

import inspect
import json
import os
import shutil

import numpy as np
import torch
from huggingface_hub import snapshot_download

import ftfy  # noqa: F401  (diffusers.pipelines.wan imports it lazily)
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline, WanTransformer3DModel
from diffusers.pipelines.wan import pipeline_wan_i2v
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "wan22")
TINY = os.path.join(OUT, "tiny")
SNAP = snapshot_download(REPO, local_files_only=True, allow_patterns=["*.json", "tokenizer/*"])

torch.set_grad_enabled(False)


def save(name, ins, outs, meta):
    os.makedirs(OUT, exist_ok=True)
    man = {"meta": meta, "inputs": [], "outputs": []}
    for kind, arrs in (("in", ins), ("out", outs)):
        for i, (label, x) in enumerate(arrs):
            x = np.ascontiguousarray(x)
            f = f"{name}.{kind}.{i}.bin"
            x.tofile(os.path.join(OUT, f))
            man["inputs" if kind == "in" else "outputs"].append(
                {"name": label, "file": f, "dtype": str(x.dtype), "shape": list(x.shape)})
    json.dump(man, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print(f"{name}: " + ", ".join(f"{l}{list(x.shape)}" for l, x in ins + outs))


def f32(x):
    return x.detach().float().numpy()


def jitter(model, seed):
    """Random norm gains and biases: the default init (ones / zeros) would
    hide a misplaced or missing normalisation weight."""
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if p.ndim == 1 or name.endswith("gamma") or "scale_shift_table" in name:
            p.data += 0.2 * torch.randn(p.shape, generator=g)
    return model


# The tiny instances keep every structural choice of the real configs
# (layouts, patch sizes, z_dim 48, head_dim 128 so the GPU's fused attention
# path is taken) and shrink only widths and depths.
TINY_T5 = dict(vocab_size=256384, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_decoder_layers=2,
               num_heads=4, relative_attention_num_buckets=32, relative_attention_max_distance=128,
               feed_forward_proj="gated-gelu", dense_act_fn="gelu_new", is_gated_act=True,
               layer_norm_epsilon=1e-6, scalable_attention=True, tie_word_embeddings=False,
               pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
TINY_DIT = dict(patch_size=[1, 2, 2], num_attention_heads=2, attention_head_dim=128, in_channels=48,
                out_channels=48, text_dim=32, freq_dim=256, ffn_dim=512, num_layers=2, cross_attn_norm=True,
                qk_norm="rms_norm_across_heads", eps=1e-6, image_dim=None, added_kv_proj_dim=None,
                rope_max_seq_len=1024, pos_embed_seq_len=None)


def tiny_vae_config():
    cfg = json.load(open(os.path.join(SNAP, "vae", "config.json")))
    known = inspect.signature(AutoencoderKLWan.__init__).parameters
    cfg = {k: v for k, v in cfg.items() if k in known}  # e.g. clip_output: newer diffusers only
    cfg.update(base_dim=16, decoder_base_dim=32)
    return cfg


def build_tiny():
    """Writes the tiny model directory (diffusers layout, as the snapshot)."""
    if os.path.exists(TINY):
        shutil.rmtree(TINY)
    torch.manual_seed(0)
    te = jitter(UMT5EncoderModel(UMT5Config(**TINY_T5)).eval(), 1)
    te.save_pretrained(os.path.join(TINY, "text_encoder"))
    torch.manual_seed(2)
    dit = jitter(WanTransformer3DModel(**TINY_DIT).eval(), 3)
    dit.save_pretrained(os.path.join(TINY, "transformer"))
    torch.manual_seed(4)
    vae = jitter(AutoencoderKLWan(**tiny_vae_config()).eval(), 5)
    vae.save_pretrained(os.path.join(TINY, "vae"))
    for sub in ("tokenizer", "scheduler"):
        shutil.copytree(os.path.join(SNAP, sub), os.path.join(TINY, sub))
    shutil.copy(os.path.join(SNAP, "model_index.json"), TINY)
    return te, dit, vae


PROMPTS = [
    "A girl with long silver hair stands on a cliff, her hair and cloak blowing in a strong wind",
    "hello world",
    "  leading and   inner   spaces  ",
    "tabs\tand\nnewlines\n",
    "unicode: café naïve 東京 🦊 ẞ “quoted” ﬁne ＡＢＣ１２３",
    "html &amp; entities &lt;b&gt; &amp;amp;",
    "<extra_id_0> special tokens </s> inline",
    "numbers 1234567 and 3.14159, don't won't",
]

# The original Wan 2.2 repository's default negative prompt (shared/config.py).
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
       "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
       "杂乱的背景，三条腿，背景人很多，倒着走")


def tokenizer_cases():
    tok = AutoTokenizer.from_pretrained(os.path.join(SNAP, "tokenizer"))
    cases = []
    for p in PROMPTS + [NEG]:
        clean = pipeline_wan_i2v.prompt_clean(p)
        cases.append({"text": p, "clean": clean, "ids": tok(clean, add_special_tokens=True).input_ids,
                      "raw_ids": tok(p, add_special_tokens=True).input_ids})
    json.dump({"cases": cases}, open(os.path.join(OUT, "tokenizer.json"), "w"), indent=1, ensure_ascii=False)
    print(f"tokenizer: {len(cases)} cases, negative prompt {len(cases[-1]['ids'])} tokens")


def umt5(te, name="umt5", prompt=PROMPTS[0]):
    tok = AutoTokenizer.from_pretrained(os.path.join(SNAP, "tokenizer"))
    ids = tok(pipeline_wan_i2v.prompt_clean(prompt), return_tensors="pt").input_ids
    out = te(ids).last_hidden_state
    # The pipeline pads to 512 with a mask; real tokens must not change.
    pad = tok(pipeline_wan_i2v.prompt_clean(prompt), padding="max_length", max_length=64, return_tensors="pt")
    outp = te(pad.input_ids, pad.attention_mask).last_hidden_state[:, :ids.shape[1]]
    print(f"{name}: |padded-unpadded| {float((outp - out).abs().max()):.2e}")
    save(name, [("input_ids", ids[0].numpy().astype(np.int64))], [("hidden", f32(out[0]))], {"prompt": prompt})


def dit_case(dit, name, F, H, W, L, text_rows, seed):
    g = torch.Generator().manual_seed(seed)
    C = dit.config.in_channels
    x = torch.randn(1, C, F, H, W, generator=g)
    enc = torch.zeros(1, L, dit.config.text_dim)
    enc[:, :text_rows] = torch.randn(1, text_rows, dit.config.text_dim, generator=g)
    t = 937.0
    mask = torch.ones(1, 1, F, H, W)
    mask[:, :, 0] = 0
    ts = (mask[0][0][:, ::2, ::2] * t).flatten().unsqueeze(0)
    out_tok = dit(hidden_states=x, timestep=ts, encoder_hidden_states=enc, return_dict=False)[0]
    out_one = dit(hidden_states=x, timestep=torch.tensor([t]), encoder_hidden_states=enc, return_dict=False)[0]
    save(name, [("x", f32(x[0])), ("text", f32(enc[0]))],
         [("out_ti2v", f32(out_tok[0])), ("out_scalar", f32(out_one[0]))],
         {"t": t, "frames": F, "hw": [H, W], "text_rows": text_rows, "layers": dit.config.num_layers})


def vae_cases(vae, name, H, W, T, seed):
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(1, 3, 1, H, W, generator=g) * 2 - 1
    enc = vae.encode(img).latent_dist
    z = torch.randn(1, vae.config.z_dim, T, H // 16, W // 16, generator=g)
    video = vae.decode(z, return_dict=False)[0]
    save(name + "_enc", [("image", f32(img[0, :, 0]))], [("mean", f32(enc.mean[0, :, 0]))], {"hw": [H, W]})
    save(name + "_dec", [("z", f32(z[0]))], [("video", f32(video[0]))], {"latent_frames": T, "hw": [H, W]})


def scheduler_case(steps=6):
    sch = UniPCMultistepScheduler.from_pretrained(SNAP, subfolder="scheduler")
    sch.set_timesteps(steps)
    g = torch.Generator().manual_seed(11)
    x0 = torch.randn(1, 48, 2, 2, 2, generator=g)
    x, outs, samples = x0, [], []
    for t in sch.timesteps:
        v = torch.randn(x.shape, generator=g) + 0.5 * x  # a model output with some sample dependence
        outs.append(v)
        x = sch.step(v, t, x, return_dict=False)[0]
        samples.append(x)
    save("scheduler", [("x0", f32(x0)), ("model_outputs", f32(torch.stack(outs)))],
         [("sigmas", f32(sch.sigmas)), ("timesteps", sch.timesteps.numpy().astype(np.int64)),
          ("samples", f32(torch.stack(samples)))],
         {"steps": steps})


def test_image(H, W):
    """A deterministic H x W RGB test card (exact size: no resampling)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H))
    px = img.load()
    for y in range(H):
        for x in range(W):
            px[x, y] = (30 + 3 * y % 200, 90 + 2 * x % 160, 220 - (x + y) % 180)
    d = ImageDraw.Draw(img)
    d.ellipse((W // 5, H // 4, W // 2, 3 * H // 4), fill=(230, 200, 60))
    d.rectangle((3 * W // 5, H // 5, 9 * W // 10, H // 2), fill=(40, 40, 160))
    return img


def pipeline_case(te, dit, vae, H=64, W=96, frames=9, steps=4, cfg=5.0):
    tok = AutoTokenizer.from_pretrained(os.path.join(TINY, "tokenizer"))
    sch = UniPCMultistepScheduler.from_pretrained(TINY, subfolder="scheduler")
    pipe = WanImageToVideoPipeline(tokenizer=tok, text_encoder=te, vae=vae, scheduler=sch, transformer=dit,
                                   expand_timesteps=True)
    img = test_image(H, W)
    img.save(os.path.join(OUT, "pipeline_image.png"))
    g = torch.Generator().manual_seed(21)
    lat_t = (frames - 1) // 4 + 1
    latents = torch.randn(1, 48, lat_t, H // 16, W // 16, generator=g)
    prompt = PROMPTS[0]
    pe, ne = pipe.encode_prompt(prompt=prompt, negative_prompt=NEG, do_classifier_free_guidance=True,
                                max_sequence_length=512)
    steps_lat = []

    def cb(p, i, t, kw):
        steps_lat.append(kw["latents"].clone())
        return {}

    video = pipe(image=img, prompt=prompt, negative_prompt=NEG, height=H, width=W, num_frames=frames,
                 num_inference_steps=steps, guidance_scale=cfg, latents=latents.clone(), output_type="np",
                 callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"], return_dict=False)[0]
    save("pipeline", [("latents0", f32(latents[0]))],
         [("prompt_embeds", f32(pe[0])), ("negative_embeds", f32(ne[0]))]
         + [(f"latents{i + 1}", f32(x[0])) for i, x in enumerate(steps_lat)]
         + [("video", np.asarray(video[0], dtype=np.float32))],
         {"prompt": prompt, "negative": NEG, "hw": [H, W], "frames": frames, "steps": steps, "guidance": cfg,
          "image": "pipeline_image.png"})


def real():
    snap = snapshot_download(REPO, local_files_only=True)
    from safetensors import safe_open

    te = UMT5EncoderModel.from_pretrained(snap, subfolder="text_encoder", torch_dtype=torch.float32).eval()
    umt5(te, "real_umt5", PROMPTS[0])
    del te

    cfg = json.load(open(os.path.join(snap, "transformer", "config.json")))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg["num_layers"] = 1
    dit = WanTransformer3DModel(**cfg).eval()
    sd = {}
    d = os.path.join(snap, "transformer")
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(d, fn), "pt") as fh:
                for k in fh.keys():
                    if not k.startswith("blocks.") or k.startswith("blocks.0."):
                        sd[k] = fh.get_tensor(k).float()
    dit.load_state_dict(sd, strict=True)
    dit_case(dit, "real_dit_l1", 3, 8, 8, 512, 12, 31)
    del dit, sd

    vae = AutoencoderKLWan.from_pretrained(snap, subfolder="vae", torch_dtype=torch.float32).eval()
    vae_cases(vae, "real_vae", 64, 64, 2, 41)


def main():
    os.makedirs(OUT, exist_ok=True)
    tokenizer_cases()
    te, dit, vae = build_tiny()
    umt5(te)
    dit_case(dit, "dit", 3, 8, 12, 512, 7, 13)
    vae_cases(vae, "vae", 64, 96, 4, 17)
    scheduler_case()
    pipeline_case(te, dit, vae)
    if os.environ.get("WAN22_REAL"):
        real()


if __name__ == "__main__":
    main()
