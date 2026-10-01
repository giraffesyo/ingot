"""wav2vec2 CTC aligner: ONNX export + reference alignment (models/align).

Exports facebook/wav2vec2-base-960h (English, character CTC, 16 kHz, one
frame per 20 ms) to testdata/align/wav2vec2-base-960h.onnx with a dynamic
time axis, and for a known clip saves the reference the Go aligner is
tested against: the normalised input, the CTC log-probabilities, and
torchaudio.functional.forced_align's frame path and per-word spans.

    HF_HUB_OFFLINE=1 tools/export/.venv-tts/bin/python tools/export/align_ref.py [clip.wav "TRANSCRIPT"]

Without arguments the clip is testdata/qwen3tts's design waveform
(tools/export/qwen3tts_ref.py) with its text.
"""

import json
import os
import sys

import numpy as np
import torch
import torchaudio
from huggingface_hub import snapshot_download
from torchaudio.functional import forced_align
from transformers import Wav2Vec2ForCTC

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "align")
REPO = "facebook/wav2vec2-base-960h"


def save(name, arrays, meta):
    os.makedirs(OUT, exist_ok=True)
    man = {"meta": meta, "tensors": []}
    for label, x in arrays:
        x = np.ascontiguousarray(x)
        f = f"{name}.{label}.bin"
        x.tofile(os.path.join(OUT, f))
        man["tensors"].append({"name": label, "file": f, "dtype": str(x.dtype), "shape": list(x.shape)})
    json.dump(man, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print(f"{name}: " + ", ".join(f"{l}{list(x.shape)}" for l, x in arrays))


def load_clip():
    if len(sys.argv) >= 3:
        wav, sr = torchaudio.load(sys.argv[1])
        return wav.mean(0).numpy(), sr, sys.argv[2]
    d = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "qwen3tts")
    man = json.load(open(os.path.join(d, "design.json")))
    wav = next(o for o in man["outputs"] if o["name"] == "wav")
    x = np.fromfile(os.path.join(d, wav["file"]), dtype=np.float32)
    return x, 24000, man["meta"]["text"]


def main():
    snap = snapshot_download(REPO, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(snap).eval()
    vocab = json.load(open(os.path.join(snap, "vocab.json")))

    os.makedirs(OUT, exist_ok=True)
    dummy = torch.zeros(1, 16000)
    torch.onnx.export(model, (dummy,), os.path.join(OUT, "wav2vec2-base-960h.onnx"),
                      input_names=["input_values"], output_names=["logits"],
                      dynamic_axes={"input_values": {1: "samples"}, "logits": {1: "frames"}},
                      opset_version=17, dynamo=False)
    print("exported", os.path.join(OUT, "wav2vec2-base-960h.onnx"))

    x, sr, text = load_clip()
    if sr != 16000:
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
    x = x.astype(np.float32)
    xn = (x - x.mean()) / np.sqrt(x.var() + 1e-7)  # Wav2Vec2FeatureExtractor do_normalize
    with torch.no_grad():
        logits = model(torch.from_numpy(xn)[None]).logits[0]
    logp = logits.log_softmax(-1)

    # Transcript → CTC targets: upper case, spaces as "|", others dropped.
    chars = [c if c != " " else "|" for c in text.upper() if c == " " or c in vocab]
    words = "".join(chars).split("|")
    targets = torch.tensor([[vocab[c] for c in "|".join(w for w in words if w)]], dtype=torch.int32)
    path, scores = forced_align(logp[None], targets, blank=vocab["<pad>"])
    path = path[0].numpy()

    # Word spans [start, end) in frames, from torchaudio's token spans.
    spans = []
    # torchaudio's merge_tokens gives token spans exactly.
    tspans = torchaudio.functional.merge_tokens(torch.from_numpy(path), scores[0].exp())
    cur = None
    seq = targets[0].tolist()
    for i, ts in enumerate(tspans):
        if seq[i] == vocab["|"]:
            if cur is not None:
                spans.append(cur)
                cur = None
            continue
        cur = [ts.start, ts.end] if cur is None else [cur[0], ts.end]
    if cur is not None:
        spans.append(cur)
    words = [w for w in words if w]
    meta = {"text": text, "words": words, "spans": spans, "frame_seconds": 0.02,
            "vocab": vocab}
    save("clip", [("x", xn.astype(np.float32)), ("logp", logp.numpy().astype(np.float32)),
                  ("path", path.astype(np.int64))], meta)
    # YIN pitch on the same 16 kHz clip (librosa, as audio.Yin mirrors).
    import librosa
    f0 = librosa.yin(x.astype(np.float32), fmin=60, fmax=500, sr=16000, frame_length=1024, hop_length=160)
    save("yin", [("x", x.astype(np.float32)), ("f0", f0.astype(np.float64))],
         {"fmin": 60, "fmax": 500, "sr": 16000, "frame_length": 1024, "hop_length": 160})
    for wd, (a, b) in zip(words, spans):
        print(f"  {wd:12s} {a * 0.02:6.2f}-{b * 0.02:6.2f} s")


if __name__ == "__main__":
    main()
