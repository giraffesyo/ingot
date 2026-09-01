"""Export PARSeq (baudm/parseq, pretrained) to ONNX + ORT reference outputs.

The AR decode loop + refinement is Python control flow; tracing unrolls it at
max_length. If the AR export fails, fall back to NAR (decode_ar=False,
refine_iters=2), which is a pure feed-forward pass with near-AR accuracy.
Run: .venv/bin/python parseq.py
"""
import json, os
import numpy as np
import torch, onnx, onnxruntime as ort

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "models")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

def try_export(name, model, x):
    path = os.path.join(OUT, name + ".onnx")
    torch.onnx.export(model, (x,), path, input_names=["x"], output_names=["logits"],
                      opset_version=18, do_constant_folding=True)
    m = onnx.load(path); onnx.checker.check_model(m)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"x": x.numpy()})
    man = {"model": name + ".onnx", "opset": 18, "inputs": [], "outputs": []}
    xa = x.numpy(); f = f"{name}.in.0.bin"; xa.tofile(os.path.join(OUT, f))
    man["inputs"].append({"name": "x", "dtype": str(xa.dtype), "shape": list(xa.shape), "file": f})
    for i, (o, y) in enumerate(zip(sess.get_outputs(), outs)):
        f = f"{name}.out.{i}.bin"; np.ascontiguousarray(y).tofile(os.path.join(OUT, f))
        man["outputs"].append({"name": o.name, "dtype": str(y.dtype), "shape": list(y.shape), "file": f})
    json.dump(man, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    with torch.no_grad():
        ref = model(x).numpy()
    err = np.abs(outs[0] - ref).max()
    print(f"{name}: {len(m.graph.node)} nodes, ORT-vs-torch max err {err:.2e}")
    ops = sorted({n.op_type for n in m.graph.node})
    print("ops:", ops)
    return name

x = torch.randn(1, 3, 32, 128)
try:
    m = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True)
    m.eval()
    try_export("parseq", m, x)
except Exception as e:
    print("AR export failed:", type(e).__name__, str(e)[:200])
    m = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True,
                       decode_ar=False, refine_iters=2)
    m.eval()
    try_export("parseq_nar", m, x)

chars = m.hparams.charset_train if hasattr(m.hparams, "charset_train") else None
if chars:
    open(os.path.join(OUT, "charset.txt"), "w").write(chars)
    print("charset:", len(chars), "chars")
