"""Reference activations for the Pixal3D parity tests (models/trellis2's
projected-feature mode).

Runs, on CPU in float32:
  - proj: the reference's ProjGrid — a 3-D grid projected through the
    camera and image features sampled there;
  - naf: the NAF feature upsampler with its released weights;
  - flows: Pixal3D's flow transformers, truncated, with projected
    features (needs the TencentARC/Pixal3D checkpoints);
  - preprocess: Pixal3D's wider subject crop of an example image.

Needs torch, safetensors, einops and checkouts of TencentARC/Pixal3D and
valeoai/NAF. NAF's neighbourhood attention comes from NATTEN, a compiled
extension; this script substitutes a plain-torch version of its window
rule (natten_cpu_commons.h, get_window_start), so nothing is compiled:

    PIXAL3D_REPO=/path/to/Pixal3D NAF_REPO=/path/to/NAF python pixal3d_ref.py

Writes testdata/trellis2/pixal_<name>.*; see trellis2_ref.py for the
layout. Run trellis2_ref.py first: the flow cases reuse its cells.
"""

import glob
import json
import os
import sys
import types

import numpy as np
import torch

from trellis2_ref import OUT, load_model, save, snapshot, stub_package

PIXAL = os.environ.get("PIXAL3D_REPO")
NAF = os.environ.get("NAF_REPO")


def _natten_shim():
    """natten.functional.na2d_qk / na2d_av in plain torch: each position
    attends a kernel×kernel window of positions `dilation` apart, shifted
    inwards at the borders."""

    def starts(length, kernel, dilation):
        idx = torch.arange(length)
        phase, ip = idx % dilation, idx // dilation
        lp = (length + dilation - 1) // dilation
        padded = lp * dilation - length
        lp = lp - (phase >= dilation - padded).long()
        ns = kernel // 2
        start = (ip - ns).clamp(min=0) + (ip + ns >= lp).long() * (lp - ip - ns - 1)
        return start * dilation + phase

    def window(h, w, kernel, dilation):
        ys = starts(h, kernel[0], dilation[0])[:, None] + torch.arange(kernel[0])[None] * dilation[0]  # [h, K]
        xs = starts(w, kernel[1], dilation[1])[:, None] + torch.arange(kernel[1])[None] * dilation[1]  # [w, K]
        return (ys[:, None, :, None] * w + xs[None, :, None, :]).reshape(h, w, -1)  # [h, w, K²] flat index

    def na2d_qk(q, k, kernel_size, dilation):
        b, n, h, w, d = q.shape
        win = window(h, w, kernel_size, dilation)
        kf = k.reshape(b, n, h * w, d)
        out = torch.empty(b, n, h, w, win.shape[-1], dtype=q.dtype)
        for y in range(h):  # row by row: the gathered keys are large
            out[:, :, y] = torch.einsum("bnwd,bnwkd->bnwk", q[:, :, y], kf[:, :, win[y]])
        return out

    def na2d_av(attn, v, kernel_size, dilation):
        b, n, h, w, d = v.shape
        win = window(h, w, kernel_size, dilation)
        vf = v.reshape(b, n, h * w, d)
        out = torch.empty(b, n, h, w, d, dtype=v.dtype)
        for y in range(h):
            out[:, :, y] = torch.einsum("bnwk,bnwkd->bnwd", attn[:, :, y], vf[:, :, win[y]])
        return out

    nat, fn = types.ModuleType("natten"), types.ModuleType("natten.functional")
    fn.na2d_qk, fn.na2d_av = na2d_qk, na2d_av
    nat.functional = fn
    sys.modules.update({"natten": nat, "natten.functional": fn})


def _naf_model():
    _natten_shim()
    sys.path.insert(0, NAF)
    from src.model.naf import NAF as Model
    path = os.path.expanduser("~/.cache/torch/hub/checkpoints/naf_release.pth")
    if not os.path.exists(path):
        sys.exit(f"{path} missing: run naf_convert.py first")
    model = Model()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


@torch.no_grad()
def naf():
    """The upsampler at 1× (image = output) and 2× (image pooled down)."""
    model = _naf_model()
    g = torch.Generator().manual_seed(11)
    for name, size, out, n in (("pixal_naf", 160, 160, 20), ("pixal_naf_pool", 192, 96, 12)):
        image = torch.rand(1, 3, size, size, generator=g)
        feats = torch.randn(1, 64, n, n, generator=g)
        hr = model(image, feats, (out, out))
        save(name, [("image", image.numpy()), ("features", feats[0].reshape(64, -1).T.numpy())],
             [("upsampled", hr[0].reshape(64, -1).T.numpy())], {"size": size, "out": out})


def _proj_module():
    # The trainer mixin's module imports the whole training stack; take
    # only its projection classes, which need torch alone.
    path = os.path.join(PIXAL, "pixal3d", "trainers", "flow_matching", "mixins", "image_conditioned_proj.py")
    src = open(path).read()
    start, end = src.index("def project_points_to_image_batch"), src.index("class ProjGridMV")
    mod = types.ModuleType("pixal_proj")
    prelude = "import numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom PIL import Image\nfrom typing import *\n"
    exec(prelude + src[start:end], mod.__dict__)
    return mod


@torch.no_grad()
def proj():
    """ProjGrid at two grid and image resolutions, off-centre camera values."""
    mod = _proj_module()
    g = torch.Generator().manual_seed(12)
    for name, R, res, n, fov, dist, scale in (("pixal_proj", 16, 512, 32, 0.8575560450553894, 1.0943, 1.0),
                                              ("pixal_proj_wide", 24, 1024, 64, 1.3, 0.9, 0.8)):
        fmap = torch.randn(1, n, n, 24, generator=g)
        out = mod.ProjGrid(grid_resolution=R, image_resolution=res)(
            fmap, torch.tensor([fov]), torch.tensor([dist]), torch.tensor([scale]))
        save(name, [("features", fmap[0].reshape(n * n, -1).numpy())], [("projected", out[0].numpy())],
             {"grid": R, "image": res, "fov": fov, "distance": dist, "mesh_scale": scale})


@torch.no_grad()
def flows(blocks=2):
    """Pixal3D's dense and sparse flow transformers, truncated, for one
    velocity evaluation with global tokens and projected features."""
    import trellis2_ref
    stub_package("pixal3d", PIXAL)
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel
    from pixal3d.models.structured_latent_flow import SLatFlowModel
    from pixal3d.modules import sparse as sp
    ckpts = os.path.join(snapshot("TencentARC/Pixal3D"), "ckpts")
    g = torch.Generator().manual_seed(13)

    model, _ = load_model(SparseStructureFlowModel, os.path.join(ckpts, "ss_flow_img_dit_1_3B_64_bf16"),
                          num_blocks=blocks, dtype="float32")
    r = model.resolution
    x = torch.randn(1, model.in_channels, r, r, r, generator=g)
    glob_, pr = torch.randn(1, 5, model.cond_channels, generator=g), torch.randn(1, r ** 3, 1024, generator=g)
    t = 0.7
    v = model(x, torch.tensor([1000 * t]), {"global": glob_, "proj": pr})
    tok = lambda a: a.reshape(a.shape[1], -1).T
    save("pixal_ss_flow", [("x", tok(x).numpy()), ("cond", glob_[0].numpy()), ("proj", pr[0].numpy())],
         [("v", tok(v).numpy())], {"blocks": blocks, "t": t})

    model, _ = load_model(SLatFlowModel, os.path.join(ckpts, "slat_flow_img2shape_dit_1_3B_512_bf16"),
                          num_blocks=blocks, dtype="float32")
    coords = trellis2_ref._structure()
    x = torch.randn(coords.shape[0], model.in_channels, generator=g)
    glob_ = torch.randn(1, 5, model.cond_channels, generator=g)
    pr = torch.randn(coords.shape[0], model.proj_in_channels, generator=g)
    t = 0.4
    v = model(sp.SparseTensor(feats=x, coords=coords), torch.tensor([1000 * t]),
              {"global": glob_, "proj": sp.SparseTensor(feats=pr, coords=coords)})
    save("pixal_slat_flow", [("x", x.numpy()), ("coords", coords[:, 1:].numpy()), ("cond", glob_[0].numpy()), ("proj", pr.numpy())],
         [("v", v.feats.numpy())], {"blocks": blocks, "t": t})


def preprocess():
    """Pixal3D's subject crop (preprocess_image, alpha path, black
    background): TRELLIS.2's with the square widened by 1.1."""
    from PIL import Image
    # Any cut-out with transparency; the reference repository ships none.
    path = os.environ.get("PIXAL3D_IMAGE")
    if not path:
        sys.exit("set PIXAL3D_IMAGE to an RGBA cut-out for the preprocess case")
    image = Image.open(path).convert("RGBA")
    rgba = np.array(image)
    bbox = np.argwhere(rgba[:, :, 3] > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.1)
    out = image.crop((center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2))
    out = np.array(out).astype(np.float32) / 255
    out = (np.clip(out[:, :, :3] * out[:, :, 3:4], 0, 1) * 255).astype(np.uint8)
    save("pixal_preprocess", [("rgba", rgba)], [("subject", out)], {"file": os.path.basename(path)})


if __name__ == "__main__":
    if not PIXAL or not NAF:
        sys.exit("set PIXAL3D_REPO and NAF_REPO to checkouts of TencentARC/Pixal3D and valeoai/NAF")
    for name in sys.argv[1:] or ["proj", "naf", "flows", "preprocess"]:
        globals()[name]()
