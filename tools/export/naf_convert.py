"""Convert the NAF feature upsampler's released checkpoint to safetensors.

NAF (valeoai/NAF, Apache-2.0) ships as a PyTorch pickle; the runtime reads
safetensors. Downloads naf_release.pth unless a path is given.

    python naf_convert.py [naf_release.pth] [out.safetensors]

Default output: $HF_HUB_CACHE (or ~/.cache/huggingface/hub)/naf/naf.safetensors
"""

import os
import sys

import torch
from safetensors.torch import save_file

URL = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"

if __name__ == "__main__":
    hub = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(hub, "naf", "naf.safetensors")
    if len(sys.argv) > 1:
        sd = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
    else:
        sd = torch.hub.load_state_dict_from_url(URL, map_location="cpu", weights_only=True)
    sd = {k: v.contiguous() for k, v in sd.items() if torch.is_tensor(v)}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_file(sd, out)
    for k, v in sd.items():
        print(f"{k:60s} {str(v.dtype):14s} {list(v.shape)}")
    print("wrote", out)
