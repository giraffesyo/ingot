"""Reference activations for Qwen-Image-2.1 parity tests.

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
SNAP = snapshot_download("Qwen/Qwen-Image-2.1", local_files_only=True, allow_patterns=["transformer/*", "vae/*"])

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


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())
    dit()
    vae()
