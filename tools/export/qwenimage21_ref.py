"""Reference activations for Qwen-Image-2.1 parity tests.

  pipeline: an end-to-end text-to-image run (256x256, 4 steps, no CFG) with
    every stage's hand-off saved — token ids, prompt embeddings, scheduler
    sigmas, the latents after each step, the decoded image — plus tokenizer
    cases. Stages load one at a time in float32 (text encoder ~35 GB, then
    DiT + VAE ~30 GB), so it fits a 48 GB machine.

Runs, on CPU in float32 with the real checkpoint weights:
  - a 1-layer QwenImage21Transformer2DModel (block 0 + every non-block
    weight: img_in, txt_in, timestep embedder, shared modulation, norm_out,
    proj_out) in three modes: no cache, prefix-KV "extract", and "cached"
    at a second timestep — the three paths the full pipeline takes;
  - the VAE decoder on a small latent.

Needs torch + a diffusers with Qwen-Image-2.1 (not in this venv; use the one
that runs the pipeline):

    HF_HUB_OFFLINE=1 ~/development/qwen/.venv/bin/python qwenimage21_ref.py

Writes testdata/qwenimage21/<name>.{in,out}.<i>.bin + <name>.json in the
same raw little-endian layout as zoo.py. Weights are read from the local HF
cache; outputs are derived from licence-bound weights and are not committed.
"""

import json
import os

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from diffusers import AutoencoderKLQwenImage21
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21KVCache,
    QwenImage21Transformer2DModel,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "qwenimage21")
SNAP = snapshot_download("Qwen/Qwen-Image-2.1", local_files_only=True,
                         allow_patterns=["*.json", "transformer/*", "vae/*", "text_encoder/*", "processor/*", "scheduler/*"])

TXT = 24          # text tokens (last PAD of them masked out)
PAD = 3
LAT = 16          # target latent side → 256 image tokens (a 256² image)


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


def load_subset(dirname, keep):
    d = os.path.join(SNAP, dirname)
    sd = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(d, f), "pt") as fh:
                for k in fh.keys():
                    if keep(k):
                        sd[k] = fh.get_tensor(k).float()
    return sd


def dit():
    cfg = json.load(open(os.path.join(SNAP, "transformer", "config.json")))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg["num_layers"] = 1
    model = QwenImage21Transformer2DModel(**cfg).float().eval()
    sd = load_subset("transformer", lambda k: not k.startswith("transformer_blocks.")
                     or k.startswith("transformer_blocks.0."))
    missing, unexpected = model.load_state_dict(sd, strict=True), None
    del missing, unexpected

    g = torch.Generator().manual_seed(0)
    tgt = LAT * LAT
    hs = torch.randn(1, tgt, cfg["in_channels"], generator=g)
    hs2 = torch.randn(1, tgt, cfg["in_channels"], generator=g)
    enc = torch.randn(1, TXT, cfg["context_in_dim"], generator=g)
    enc_mask = torch.ones(1, TXT, dtype=torch.bool)
    enc_mask[:, TXT - PAD:] = False
    img_mask = torch.zeros(1, TXT + tgt // 4, dtype=torch.bool)
    img_mask[:, TXT:] = True
    shapes = [[(1, LAT, LAT)]]
    t1, t2 = torch.tensor([0.7]), torch.tensor([0.4])

    def run(x, t, **kw):
        with torch.no_grad():
            return model(hidden_states=x, encoder_hidden_states=enc, timestep=t, img_shapes=shapes,
                         img_mask=img_mask, encoder_hidden_states_mask=enc_mask, return_dict=False, **kw)[0]

    full1 = run(hs, t1)
    cache = QwenImage21KVCache(1)
    ext = run(hs, t1, kv_cache=cache, kv_cache_mode="extract")
    cached2 = run(hs2, t2, kv_cache=cache, kv_cache_mode="cached")
    full2 = run(hs2, t2)
    print(f"dit: |extract-full| {float((ext - full1).abs().max()):.2e}  "
          f"|cached-full| {float((cached2 - full2[:, -cached2.shape[1]:]).abs().max()):.2e}")

    meta = {"layers": 1, "txt": TXT, "txt_pad": PAD, "latent_hw": [LAT, LAT], "t": [0.7, 0.4],
            "config": cfg}
    f = lambda x: x.float().numpy()
    save("dit_l1",
         [("hidden_states", f(hs)), ("hidden_states_step2", f(hs2)), ("encoder_hidden_states", f(enc)),
          ("encoder_hidden_states_mask", enc_mask.numpy().astype(np.uint8)),
          ("img_mask", img_mask.numpy().astype(np.uint8))],
         [("out_step1", f(full1)), ("out_step2_cached", f(cached2))], meta)


def vae():
    model = AutoencoderKLQwenImage21.from_pretrained(SNAP, subfolder="vae", torch_dtype=torch.float32).eval()
    g = torch.Generator().manual_seed(1)
    z = torch.randn(1, model.config.z_dim, 1, LAT, LAT, generator=g)
    with torch.no_grad():
        img = model.decode(z, return_dict=False)[0]
    save("vae_dec", [("z", z.numpy())], [("image", img.numpy())], {"latent_hw": [LAT, LAT]})


PROMPT = 'A red fox standing in fresh snow, holding a wooden sign that reads "INGOT"'
TOKENIZER_CASES = [
    "hello world", "Hello, World!", "  leading and   inner   spaces  ", "tabs\tand\nnewlines\n\n",
    "numbers 1234567 and 3.14159", "unicode: café naïve 東京 🦊 ẞ", "don't won't I'm we'll they've",
    "<|im_start|>user\nhi<|im_end|>", "trailing space ", "MiXeD CaSe_snake-kebab.dot/slash",
    PROMPT,
]


def pipeline():
    import gc
    from diffusers import QwenImage21Pipeline

    # Stage 1: text encoder (float32), then free it.
    pipe = QwenImage21Pipeline.from_pretrained(SNAP, transformer=None, vae=None, torch_dtype=torch.float32)
    tok = pipe.processor.tokenizer
    templ = pipe.prompt_template_t2i.format(PROMPT)
    ids = pipe.processor(text=[templ], return_tensors="pt").input_ids
    with torch.no_grad():
        embeds, emb_mask, img_pad = pipe.encode_prompt(PROMPT)
    cases = [{"text": t, "ids": tok(t, add_special_tokens=False).input_ids} for t in TOKENIZER_CASES]
    json.dump({"cases": cases, "template": templ, "drop_idx": pipe._drop_idx}, open(os.path.join(OUT, "tokenizer.json"), "w"), indent=1)
    print(f"text: {ids.shape[1]} tokens, embeds {list(embeds.shape)}, mask {emb_mask}, drop_idx {pipe._drop_idx}")
    del pipe
    gc.collect()

    # Stage 2: DiT + VAE (float32) over the saved embeddings and fixed noise.
    pipe = QwenImage21Pipeline.from_pretrained(SNAP, text_encoder=None, torch_dtype=torch.float32)
    lat = LAT * LAT
    g = torch.Generator().manual_seed(7)
    lat0 = torch.randn(1, lat, pipe.transformer.config.in_channels, generator=g)
    steps = []
    def cb(p, i, t, kw):
        steps.append(kw["latents"].clone())
        return {}
    with torch.no_grad():
        final = pipe(prompt_embeds=embeds, width=LAT * 16, height=LAT * 16, num_inference_steps=4, latents=lat0,
                     output_type="latent", callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"],
                     return_dict=False)[0]
        z = pipe._unpack_latents(final, LAT * 16, LAT * 16, pipe.vae_scale_factor)
        mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1)
        image = pipe.vae.decode(z * std + mean, return_dict=False)[0][:, :, 0]
    f = lambda x: x.float().numpy()
    save("pipeline",
         [("input_ids", ids.numpy().astype(np.int64)), ("prompt_embeds", f(embeds[0])), ("latents0", f(lat0[0]))],
         [("sigmas", f(pipe.scheduler.sigmas)), ("timesteps", f(pipe.scheduler.timesteps))]
         + [(f"latents{i + 1}", f(x[0])) for i, x in enumerate(steps)] + [("image", f(image[0]))],
         {"prompt": PROMPT, "steps": 4, "hw": [LAT * 16, LAT * 16]})


if __name__ == "__main__":
    import sys
    torch.set_num_threads(os.cpu_count())
    which = sys.argv[1:] or ["dit", "vae", "pipeline"]
    for name in which:
        {"dit": dit, "vae": vae, "pipeline": pipeline}[name]()
