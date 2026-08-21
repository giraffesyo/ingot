"""Export reference ONNX models + ONNX Runtime outputs into testdata/models.

Usage: .venv/bin/python export.py [names...]

Each model produces:
  <name>.onnx            the model
  <name>.json            manifest: inputs/outputs with dtype, shape, file
  <name>.<io>.<i>.bin    raw little-endian tensor data
Go conformance tests load the manifest, run the model, and diff outputs.
"""
import json, os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import onnx, onnxruntime as ort

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "models")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)
OPSET = 17

def export(name, model, inputs, input_names, dynamic=None, opset=OPSET):
    model.eval()
    path = os.path.join(OUT, name + ".onnx")
    kwargs = dict(input_names=input_names, output_names=["out"], opset_version=opset,
                  dynamic_axes=dynamic or {}, do_constant_folding=True)
    try:
        torch.onnx.export(model, tuple(inputs), path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, tuple(inputs), path, **kwargs)
    m = onnx.load(path)
    onnx.checker.check_model(m)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {n: x.numpy() for n, x in zip(input_names, inputs)}
    outs = sess.run(None, feeds)
    man = {"model": name + ".onnx", "opset": opset, "inputs": [], "outputs": []}
    for i, (n, x) in enumerate(feeds.items()):
        f = f"{name}.in.{i}.bin"; x.astype(x.dtype).tofile(os.path.join(OUT, f))
        man["inputs"].append({"name": n, "dtype": str(x.dtype), "shape": list(x.shape), "file": f})
    for i, (o, y) in enumerate(zip(sess.get_outputs(), outs)):
        f = f"{name}.out.{i}.bin"; np.ascontiguousarray(y).tofile(os.path.join(OUT, f))
        man["outputs"].append({"name": o.name, "dtype": str(y.dtype), "shape": list(y.shape), "file": f})
    json.dump(man, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"{name}: {len(m.graph.node)} nodes, ops={ops}, size={os.path.getsize(path)//1024}KB")

class TinyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1, groups=8)   # depthwise
        self.c3 = nn.Conv2d(8, 16, 1)                        # pointwise
        self.bn3 = nn.BatchNorm2d(16)
        self.c4 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.fc = nn.Linear(16, 10)
        # non-trivial BN stats
        with torch.no_grad():
            for bn in (self.bn1, self.bn3):
                bn.running_mean.normal_(); bn.running_var.uniform_(0.5, 2)
                bn.weight.normal_(); bn.bias.normal_()
    def forward(self, x):
        x = F.relu(self.bn1(self.c1(x)))
        x = F.max_pool2d(x, 2)
        x = F.hardswish(self.c2(x))
        x = self.bn3(self.c3(x))
        x = F.hardsigmoid(x) * x
        x = F.avg_pool2d(self.c4(x), 3, stride=1, padding=1)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return F.softmax(self.fc(x), dim=1)

class TinyTransformer(nn.Module):
    def __init__(self, d=32, h=4, L=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d); self.proj = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, 4*d); self.fc2 = nn.Linear(4*d, d)
        self.h = h; self.d = d
        self.pos = nn.Parameter(torch.randn(1, 16, d))
    def forward(self, x):  # x: [B, T, d]
        B, T, d = x.shape
        x = x + self.pos[:, :T]
        y = self.ln1(x)
        qkv = self.qkv(y).reshape(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (1.0 / (d // self.h) ** 0.5)
        att = att.softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x

def main(names):
    todo = {
        "tiny_conv": lambda: export("tiny_conv", TinyConv(), [torch.randn(2, 3, 16, 16)], ["x"]),
        "tiny_transformer": lambda: export("tiny_transformer", TinyTransformer(), [torch.randn(2, 8, 32)], ["x"]),
        "mobilenet_v3_small": lambda: _mnv3(),
    }
    for n in (names or todo):
        todo[n]()

def _mnv3():
    import torchvision
    try:
        m = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception as e:
        print("pretrained weights unavailable, using random init:", e)
        m = torchvision.models.mobilenet_v3_small()
    export("mobilenet_v3_small", m, [torch.randn(1, 3, 224, 224)], ["x"])

if __name__ == "__main__":
    main(sys.argv[1:])
