"""Reference activations for TRELLIS.2 parity tests.

Runs, on CPU in float32 with the real checkpoint weights:
  - dino: the DINOv3 ViT-L/16 conditioning encoder exactly as the pipeline's
    DinoV3FeatureExtractor does (embeddings, rotary layers, a final
    unparameterised layer norm), on a small image;
  - ss_flow: the sparse-structure flow transformer truncated to its first
    blocks, one velocity evaluation;
  - ss_dec: the sparse-structure decoder (dense 3-D convs) on a latent;
  - ss_sample: the whole sparse-structure stage (sampler, decoder, cells);
  - slat_flow: a structured-latent flow transformer over those cells;
  - slat_dec: the shape and texture decoders (sparse convs) over them;
  - preprocess: the subject crop and Lanczos resize of an example image.

Needs torch, transformers, safetensors and a checkout of microsoft/TRELLIS.2
(only its model definitions are imported; none of its CUDA extensions):

    TRELLIS2_REPO=/path/to/TRELLIS.2 HF_HUB_OFFLINE=1 python trellis2_ref.py

Writes testdata/trellis2/<name>.{in,out}.<i>.bin + <name>.json in the same
raw little-endian layout as zoo.py. Weights are read from the local HF
cache; outputs are derived from licence-bound weights and are not committed.
"""

import glob
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

os.environ.setdefault("ATTN_BACKEND", "sdpa")
REPO = os.environ.get("TRELLIS2_REPO")
if not REPO:
    sys.exit("set TRELLIS2_REPO to a checkout of microsoft/TRELLIS.2")
# Import the model definitions without the package __init__, which pulls in
# the renderers and their CUDA extensions.
pkg = types.ModuleType("trellis2")
pkg.__path__ = [os.path.join(REPO, "trellis2")]
sys.modules["trellis2"] = pkg



def _install_cpu_shims():
    """The sparse modules call two CUDA-only libraries. These stand-ins give
    them the same functions in plain torch, so the reference's own module
    code runs unmodified on CPU (batch size 1 only)."""
    import torch.nn.functional as F

    def neighbor_map(coords, shape, ksize, dilation):
        _, _, W, H, D = shape
        M = torch.tensor([W * H * D, H * D, D, 1]).long()
        keys = (coords.long() * M[None]).sum(dim=-1)
        sorted_keys, indices = torch.sort(keys)
        offset = torch.stack(torch.meshgrid(*[
            torch.arange(-(k // 2) * d, k // 2 * d + 1, d) for k, d in zip(ksize, dilation)], indexing="ij"), dim=-1).reshape(-1, 3)
        V = offset.shape[0]
        nc = coords.long().unsqueeze(1).repeat(1, V, 1)
        nc[:, :, 1:] += offset.unsqueeze(0)
        nc = nc.reshape(-1, 4)
        valid = ((nc[:, 1] >= 0) & (nc[:, 1] < W) & (nc[:, 2] >= 0) & (nc[:, 2] < H) & (nc[:, 3] >= 0) & (nc[:, 3] < D))
        nkeys = (nc * M[None]).sum(dim=-1)
        pos = torch.searchsorted(sorted_keys, nkeys).clamp(0, sorted_keys.shape[0] - 1)
        valid &= sorted_keys[pos] == nkeys
        out = torch.full((nc.shape[0],), -1, dtype=torch.long)
        out[valid] = indices[pos[valid]]
        return out.reshape(coords.shape[0], V)

    def sparse_submanifold_conv3d(feats, coords, shape, weight, bias, neighbor_cache, dilation):
        Co, Kw, Kh, Kd, Ci = weight.shape
        if neighbor_cache is None:
            neighbor_cache = neighbor_map(coords, shape, (Kw, Kh, Kd), dilation)
        N, V = neighbor_cache.shape
        col = torch.zeros((N * V, Ci), dtype=feats.dtype)
        mask = neighbor_cache.view(-1) >= 0
        col[mask] = feats[neighbor_cache.view(-1)[mask]]
        out = col.view(N, V * Ci) @ weight.reshape(Co, V * Ci).T
        return (out + bias if bias is not None else out), neighbor_cache

    fg, fops, fsp = types.ModuleType("flex_gemm"), types.ModuleType("flex_gemm.ops"), types.ModuleType("flex_gemm.ops.spconv")
    fsp.sparse_submanifold_conv3d = sparse_submanifold_conv3d
    fsp.set_algorithm = fsp.set_hashmap_ratio = lambda *a: None
    fg.ops, fops.spconv = fops, fsp
    sys.modules.update({"flex_gemm": fg, "flex_gemm.ops": fops, "flex_gemm.ops.spconv": fsp})

    def attend(q, k, v):
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])
        return o[0].transpose(0, 1)

    fa = types.ModuleType("flash_attn")
    fa.flash_attn_varlen_qkvpacked_func = lambda qkv, cu, mx: attend(*qkv.unbind(dim=1))
    fa.flash_attn_varlen_kvpacked_func = lambda q, kv, cq, ck, mq, mk: attend(q, *kv.unbind(dim=1))
    fa.flash_attn_varlen_func = lambda q, k, v, cq, ck, mq, mk: attend(q, k, v)
    sys.modules["flash_attn"] = fa


_install_cpu_shims()

HUB = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "trellis2")


def snapshot(repo):
    snaps = glob.glob(os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots", "*"))
    if not snaps:
        sys.exit(f"{repo} is not in the HF cache")
    return snaps[0]


T2 = snapshot("microsoft/TRELLIS.2-4B")
T1 = snapshot("microsoft/TRELLIS-image-large")
DINO = snapshot("facebook/dinov3-vitl16-pretrain-lvd1689m")


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


def load_model(cls, path, **override):
    """Builds cls from path.json in float32 and loads path.safetensors."""
    args = json.load(open(path + ".json"))["args"]
    args.update(override)
    model = cls(**args)
    sd = {k: v.float() for k, v in load_file(path + ".safetensors").items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "rope_phases" not in k]
    if missing:
        sys.exit(f"{path}: missing {missing[:5]}")
    return model.float().eval(), unexpected


@torch.no_grad()
def dino(size=224):
    from transformers import DINOv3ViTModel
    model = DINOv3ViTModel.from_pretrained(DINO).eval()
    g = torch.Generator().manual_seed(1)
    image = torch.rand(1, 3, size, size, generator=g)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (image - mean) / std
    h = model.embeddings(x, bool_masked_pos=None)
    pos = model.rope_embeddings(x)
    # transformers moved the layer list under .model in later releases.
    for layer in getattr(model, "layer", None) or model.model.layer:
        h = layer(h, position_embeddings=pos)
    out = F.layer_norm(h, h.shape[-1:])
    save("dino", [("image", x.numpy())], [("features", out[0].numpy())], {"size": size})


@torch.no_grad()
def ss_flow(blocks=2, cond_tokens=37):
    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    model, _ = load_model(SparseStructureFlowModel, os.path.join(T2, "ckpts", "ss_flow_img_dit_1_3B_64_bf16"),
                          num_blocks=blocks, dtype="float32")
    g = torch.Generator().manual_seed(2)
    r = model.resolution
    x = torch.randn(1, model.in_channels, r, r, r, generator=g)
    cond = torch.randn(1, cond_tokens, model.cond_channels, generator=g)
    t = 0.7
    v = model(x, torch.tensor([1000 * t]), cond)
    tok = lambda a: a.reshape(a.shape[1], -1).T
    save("ss_flow", [("x", tok(x).numpy()), ("cond", cond[0].numpy())], [("v", tok(v).numpy())],
         {"blocks": blocks, "t": t})


@torch.no_grad()
def ss_dec():
    from trellis2.models.sparse_structure_vae import SparseStructureDecoder
    model, _ = load_model(SparseStructureDecoder, os.path.join(T1, "ckpts", "ss_dec_conv3d_16l8_fp16"), use_fp16=False)
    g = torch.Generator().manual_seed(3)
    z = torch.randn(1, model.latent_channels, 16, 16, 16, generator=g)
    save("ss_dec", [("z", z.numpy())], [("logits", model(z).numpy())], {})


@torch.no_grad()
def ss_sample(cond_tokens=201):
    """The full sparse-structure stage: the pipeline's sampler over the
    30-block flow model from fixed noise, then the decoder and the
    occupancy coordinates at 32³ (the '512' pipeline's structure)."""
    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    from trellis2.models.sparse_structure_vae import SparseStructureDecoder
    from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
    args = json.load(open(os.path.join(T2, "pipeline.json")))["args"]["sparse_structure_sampler"]
    model, _ = load_model(SparseStructureFlowModel, os.path.join(T2, "ckpts", "ss_flow_img_dit_1_3B_64_bf16"),
                          dtype="float32")
    dec, _ = load_model(SparseStructureDecoder, os.path.join(T1, "ckpts", "ss_dec_conv3d_16l8_fp16"), use_fp16=False)
    g = torch.Generator().manual_seed(4)
    r = model.resolution
    noise = torch.randn(1, model.in_channels, r, r, r, generator=g)
    # The image features of the dino case, so the sample is a real shape.
    cond = torch.from_numpy(np.fromfile(os.path.join(OUT, "dino.out.0.bin"), dtype=np.float32).reshape(1, cond_tokens, -1))
    sampler = FlowEulerGuidanceIntervalSampler(**args["args"])
    z = sampler.sample(model, noise, cond=cond, neg_cond=torch.zeros_like(cond), verbose=True, **args["params"]).samples
    occ = dec(z) > 0
    occ = F.max_pool3d(occ.float(), 2, 2, 0) > 0.5
    coords = torch.argwhere(occ)[:, [2, 3, 4]].int()
    tok = lambda a: a.reshape(a.shape[1], -1).T
    save("ss_sample", [("noise", tok(noise).numpy()), ("cond", cond[0].numpy())],
         [("z", tok(z).numpy()), ("coords", coords.numpy())], {"params": args["params"], "sigma_min": args["args"]["sigma_min"]})


def _structure():
    """The cells of the ss_sample case, as the reference's [N, 4] coords."""
    c = np.fromfile(os.path.join(OUT, "ss_sample.out.1.bin"), dtype=np.int32).reshape(-1, 3)
    return torch.cat([torch.zeros(len(c), 1, dtype=torch.int32), torch.from_numpy(c)], dim=1)


@torch.no_grad()
def slat_flow(blocks=2, cond_tokens=201):
    """The shape structured-latent flow transformer, truncated, for one
    velocity evaluation over the ss_sample cells."""
    from trellis2.models.structured_latent_flow import SLatFlowModel
    from trellis2.modules import sparse as sp
    model, _ = load_model(SLatFlowModel, os.path.join(T2, "ckpts", "slat_flow_img2shape_dit_1_3B_512_bf16"),
                          num_blocks=blocks, dtype="float32")
    coords = _structure()
    g = torch.Generator().manual_seed(5)
    x = torch.randn(coords.shape[0], model.in_channels, generator=g)
    cond = torch.from_numpy(np.fromfile(os.path.join(OUT, "dino.out.0.bin"), dtype=np.float32).reshape(1, cond_tokens, -1))
    t = 0.4
    v = model(sp.SparseTensor(feats=x, coords=coords), torch.tensor([1000 * t]), cond)
    save("slat_flow", [("x", x.numpy()), ("coords", coords[:, 1:].numpy()), ("cond", cond[0].numpy())],
         [("v", v.feats.numpy())], {"blocks": blocks, "t": t})


@torch.no_grad()
def slat_dec():
    """Both structured-latent decoders over the ss_sample cells: the shape
    decoder predicting its subdivisions, the texture decoder guided by
    them. Saves every level's cells and subdivision logits."""
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    from trellis2.modules import sparse as sp
    norm = json.load(open(os.path.join(T2, "pipeline.json")))["args"]
    coords = _structure()
    g = torch.Generator().manual_seed(6)
    for name, ckpt, key in (("shape_dec", "shape_dec_next_dc_f16c32_fp16", "shape_slat_normalization"),
                            ("tex_dec", "tex_dec_next_dc_f16c32_fp16", "tex_slat_normalization")):
        args = json.load(open(os.path.join(T2, "ckpts", ckpt + ".json")))["args"]
        args.pop("resolution", None)
        args.pop("voxel_margin", None)
        args.setdefault("out_channels", 7)
        args["use_fp16"] = False
        model = SparseUnetVaeDecoder(**args)
        sd = {k: v.float() for k, v in load_file(os.path.join(T2, "ckpts", ckpt + ".safetensors")).items()}
        model.load_state_dict(sd)
        model = model.float().eval()
        std, mean = torch.tensor(norm[key]["std"]), torch.tensor(norm[key]["mean"])
        latent = torch.randn(coords.shape[0], model.from_latent.in_features, generator=g) * std + mean
        x = sp.SparseTensor(feats=latent, coords=coords)
        outs = [("latent", latent.numpy())]
        if model.pred_subdiv:
            h, subs = model(x, return_subs=True)
            for i, s in enumerate(subs):
                outs += [(f"coords{i}", s.coords[:, 1:].int().numpy()), (f"subdiv{i}", s.feats.numpy())]
        else:
            h = model(x, guide_subs=subs)
        outs += [("coords", h.coords[:, 1:].int().numpy()), ("out", h.feats.numpy())]
        save(name, outs[:1], outs[1:], {})


def preprocess(size=512):
    """Image preparation on one of the repository's example cut-outs: the
    pipeline's subject crop (preprocess_image, alpha path) and the
    encoder's Lanczos resize."""
    from PIL import Image
    path = sorted(glob.glob(os.path.join(REPO, "assets", "example_image", "*.webp")))[0]
    image = Image.open(path)
    assert image.mode == "RGBA"
    rgba = np.array(image)
    alpha = rgba[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    half = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) // 2
    out = image.crop((center[0] - half, center[1] - half, center[0] + half, center[1] + half))
    out = np.array(out).astype(np.float32) / 255
    out = Image.fromarray((out[:, :, :3] * out[:, :, 3:4] * 255).astype(np.uint8))
    resized = out.resize((size, size), Image.LANCZOS)
    save("preprocess", [("rgba", rgba)], [("subject", np.array(out)), ("resized", np.array(resized))], {"size": size})


if __name__ == "__main__":
    want = sys.argv[1:] or ["dino", "ss_flow", "ss_dec", "ss_sample", "slat_flow", "slat_dec", "preprocess"]
    for name in want:
        globals()[name]()
