"""Export a diverse model zoo to probe runtime op coverage.

Random-init small models (conformance checks parity with ONNX Runtime, not
accuracy). Each exercises a different op cluster. Run: .venv/bin/python zoo.py
"""
import json, os, sys, traceback
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import onnx, onnxruntime as ort

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "models")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

def export(name, model, inputs, input_names, opset=17, dynamic_shapes=None):
    model.eval()
    path = os.path.join(OUT, name + ".onnx")
    try:
        torch.onnx.export(model, tuple(inputs), path, input_names=input_names,
                          output_names=["out"], opset_version=opset, do_constant_folding=True,
                          dynamic_shapes=dynamic_shapes)
    except Exception as e:
        print(f"{name}: EXPORT FAILED: {e}")
        return
    m = onnx.load(path); onnx.checker.check_model(m)
    # Annotate value shapes: the runtime's layout passes gate on static spatial
    # sizes and some exporter paths omit value_info entirely (mv3 did).
    m = onnx.shape_inference.infer_shapes(m); onnx.save(m, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {n: x.numpy() for n, x in zip(input_names, inputs)}
    outs = sess.run(None, feeds)
    man = {"model": name + ".onnx", "opset": opset, "inputs": [], "outputs": []}
    for i, (n, x) in enumerate(feeds.items()):
        f = f"{name}.in.{i}.bin"; x.tofile(os.path.join(OUT, f))
        man["inputs"].append({"name": n, "dtype": str(x.dtype), "shape": list(x.shape), "file": f})
    for i, (o, y) in enumerate(zip(sess.get_outputs(), outs)):
        f = f"{name}.out.{i}.bin"; np.ascontiguousarray(y).tofile(os.path.join(OUT, f))
        man["outputs"].append({"name": o.name, "dtype": str(y.dtype), "shape": list(y.shape), "file": f})
    json.dump(man, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"{name}: {len(m.graph.node)} nodes, ops={ops}")

# --- ResNet-style: Conv/BN/Relu/Add residual, MaxPool, GAP, Linear ---
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d): m.running_var.uniform_(0.5, 1.5); m.running_mean.normal_()
    def forward(self, x):
        idn = x if self.down is None else self.down(x)
        x = F.relu(self.b1(self.c1(x))); x = self.b2(self.c2(x))
        return F.relu(x + idn)
class ResNetish(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 7, 2, 3, bias=False), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        self.l1 = BasicBlock(16, 16); self.l2 = BasicBlock(16, 32, 2); self.l3 = BasicBlock(32, 64, 2)
        self.fc = nn.Linear(64, 10)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d): m.running_var.uniform_(0.5, 1.5)
    def forward(self, x):
        x = self.l3(self.l2(self.l1(self.stem(x))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

# --- ViT: patch embed conv, cls token, pos emb, transformer, head ---
class ViT(nn.Module):
    def __init__(self, dim=48, heads=4, depth=2, patch=8, img=32):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch, patch)
        n = (img // patch) ** 2
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim))
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, dim*2, batch_first=True, activation="gelu", norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, 10)
    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)          # [B, N, D]
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], 1) + self.pos
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x)[:, 0])

# --- BERT-ish encoder: embedding gather, LN, attention, GELU, tanh pooler ---
class Bertish(nn.Module):
    def __init__(self, vocab=100, dim=48, heads=4, depth=2, maxlen=16):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim); self.pos = nn.Embedding(maxlen, dim)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, dim*2, batch_first=True, activation="gelu"), depth)
        self.pool = nn.Linear(dim, dim)
    def forward(self, ids):
        pos = torch.arange(ids.shape[1]).unsqueeze(0)
        x = self.tok(ids) + self.pos(pos)
        x = self.enc(x)
        return torch.tanh(self.pool(x[:, 0]))

# --- Segmentation head: encoder + bilinear upsample (Resize) + transpose conv ---
class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU())
        self.e2 = nn.Sequential(nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU())
        self.up = nn.ConvTranspose2d(32, 16, 2, 2)
        self.head = nn.Conv2d(16, 2, 1)
    def forward(self, x):
        x = self.e2(self.e1(x))
        x = F.relu(self.up(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)

# --- LLM block: RMSNorm + RoPE + SwiGLU + causal attention (decomposed) ---
class RMSNorm(nn.Module):
    def __init__(self, d): super().__init__(); self.w = nn.Parameter(torch.ones(d))
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w
class LLMBlock(nn.Module):
    def __init__(self, dim=48, heads=4, hidden=128):
        super().__init__()
        self.n1 = RMSNorm(dim); self.n2 = RMSNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False); self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False); self.o = nn.Linear(dim, dim, bias=False)
        self.w1 = nn.Linear(dim, hidden, bias=False); self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False); self.h = heads; self.dim = dim
    def forward(self, x):
        B, T, D = x.shape; hd = D // self.h
        y = self.n1(x)
        q = self.q(y).view(B, T, self.h, hd).transpose(1, 2)
        k = self.k(y).view(B, T, self.h, hd).transpose(1, 2)
        v = self.v(y).view(B, T, self.h, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
        mask = torch.triu(torch.full((T, T), float("-inf")), 1)
        att = (att + mask).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o(y)
        g = self.n2(x)
        x = x + self.w3(F.silu(self.w1(g)) * self.w2(g))
        return x

# --- GPT-ish: a realistic-scale decoder stack (the SDPA-at-scale and
# bf16-storage benchmark target). dim 512, 8 heads, T=256: each block's
# [1,8,256,256] score tensor is 2 MB — fuse-sdpa keeps it in cache.
class GPTish(nn.Module):
    def __init__(self, dim=512, heads=8, depth=4, hidden=1376):
        super().__init__()
        self.blocks = nn.ModuleList([LLMBlock(dim, heads, hidden) for _ in range(depth)])
        self.norm = RMSNorm(dim)
    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

MODELS = {
    "resnetish": lambda: export("resnetish", ResNetish(), [torch.randn(1, 3, 64, 64)], ["x"]),
    "vit": lambda: export("vit", ViT(), [torch.randn(1, 3, 32, 32)], ["x"]),
    "bertish": lambda: export("bertish", Bertish(), [torch.randint(0, 100, (1, 16))], ["ids"]),
    "segnet": lambda: export("segnet", SegNet(), [torch.randn(1, 3, 32, 32)], ["x"]),
    "llmblock": lambda: export("llmblock", LLMBlock(), [torch.randn(1, 12, 48)], ["x"]),
    "gptish": lambda: export("gptish", GPTish(), [torch.randn(1, 256, 512)], ["x"]),
    "gptish_1k": lambda: export("gptish_1k", GPTish(), [torch.randn(1, 1024, 512)], ["x"]),
    "gptish_dyn": lambda: export("gptish_dyn", GPTish(), [torch.randn(1, 256, 512)], ["x"],
                                 dynamic_shapes={"x": {1: torch.export.Dim("T", min=2, max=2048)}}),
    "mobilenet_v2": lambda: _mv2(),
    "efficientnet_b0": lambda: _effnet(),
}
def _mv2():
    import torchvision
    export("mobilenet_v2", torchvision.models.mobilenet_v2(), [torch.randn(1, 3, 224, 224)], ["x"])
def _effnet():
    import torchvision
    export("efficientnet_b0", torchvision.models.efficientnet_b0(), [torch.randn(1, 3, 224, 224)], ["x"])

MODELS["opprobe"] = lambda: export("opprobe", OpProbe(), [torch.randn(1, 3, 8, 8)], ["x"])
MODELS["deconvprobe"] = lambda: export("deconvprobe", DeconvProbe(), [torch.randn(1, 4, 8, 8)], ["x"])


# ---- statically quantized (int8) variants: QLinearConv islands ----
# Convs only (op_types_to_quantize) keeps the model to standard ONNX ops
# (QuantizeLinear/DequantizeLinear/QLinearConv); full QOperator quantization
# emits com.microsoft contrib ops (QLinearAdd, ...) we don't implement.
def export_int8(name, src_name):
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)
    src = os.path.join(OUT, f"{src_name}.onnx")
    dst = os.path.join(OUT, f"{name}.onnx")
    man = json.load(open(os.path.join(OUT, f"{src_name}.json")))

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.n = 0
        def get_next(self):
            if self.n >= 8:
                return None
            self.n += 1
            rng = np.random.default_rng(self.n)
            return {i["name"]: rng.standard_normal(i["shape"]).astype(np.float32)
                    for i in man["inputs"]}

    quantize_static(src, dst, Reader(), quant_format=QuantFormat.QOperator,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, op_types_to_quantize=["Conv"])
    sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    feeds = {}
    inputs = []
    for i, im in enumerate(man["inputs"]):
        rng = np.random.default_rng(100 + i)
        arr = rng.standard_normal(im["shape"]).astype(np.float32)
        fn = f"{name}.in.{i}.bin"
        arr.tofile(os.path.join(OUT, fn))
        feeds[im["name"]] = arr
        inputs.append({"name": im["name"], "dtype": "float32", "shape": im["shape"], "file": fn})
    outs = sess.run(None, feeds)
    outputs = []
    for i, (o, arr) in enumerate(zip(sess.get_outputs(), outs)):
        fn = f"{name}.out.{i}.bin"
        np.asarray(arr).tofile(os.path.join(OUT, fn))
        outputs.append({"name": o.name, "dtype": str(arr.dtype), "shape": list(arr.shape), "file": fn})
    m = onnx.load(dst)
    ops = sorted({n.op_type for n in m.graph.node})
    json.dump({"model": f"{name}.onnx", "opset": man.get("opset", 17),
               "inputs": inputs, "outputs": outputs}, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print(f"{name}: {len(m.graph.node)} nodes, ops: {' '.join(ops)}")


# --- Post-processing / sampling op probes, built directly with onnx.helper ---
def export_graph(name, model, feeds):
    """Export a hand-built onnx ModelProto with ORT reference outputs."""
    path = os.path.join(OUT, name + ".onnx")
    onnx.checker.check_model(model)
    onnx.save(model, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    outs = sess.run(None, feeds)
    man = {"model": name + ".onnx", "opset": model.opset_import[0].version, "inputs": [], "outputs": []}
    for i, (n, x) in enumerate(feeds.items()):
        f = f"{name}.in.{i}.bin"; x.tofile(os.path.join(OUT, f))
        man["inputs"].append({"name": n, "dtype": str(x.dtype), "shape": list(x.shape), "file": f})
    for i, (o, y) in enumerate(zip(sess.get_outputs(), outs)):
        f = f"{name}.out.{i}.bin"; np.ascontiguousarray(y).tofile(os.path.join(OUT, f))
        man["outputs"].append({"name": o.name, "dtype": str(y.dtype), "shape": list(y.shape), "file": f})
    json.dump(man, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    ops = sorted({n.op_type for n in model.graph.node})
    print(f"{name}: {len(model.graph.node)} nodes, ops={ops}")

def export_postprobe():
    from onnx import helper, TensorProto
    B, S, C, K = 1, 40, 3, 5
    rng = np.random.RandomState(7)
    boxes = rng.rand(B, S, 4).astype(np.float32) * 10
    boxes[..., 2:] += boxes[..., :2]  # y2>y1, x2>x1
    scores = rng.rand(B, C, S).astype(np.float32)
    nodes = [
        helper.make_node("TopK", ["scores", "k"], ["tv", "ti"], axis=-1),
        helper.make_node("NonMaxSuppression",
                         ["boxes", "scores", "maxout", "iou", "scoreth"], ["sel"]),
    ]
    g = helper.make_graph(nodes, "postprobe",
        [helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [B, S, 4]),
         helper.make_tensor_value_info("scores", TensorProto.FLOAT, [B, C, S])],
        [helper.make_tensor_value_info("tv", TensorProto.FLOAT, [B, C, K]),
         helper.make_tensor_value_info("ti", TensorProto.INT64, [B, C, K]),
         helper.make_tensor_value_info("sel", TensorProto.INT64, [None, 3])],
        [helper.make_tensor("k", TensorProto.INT64, [1], [K]),
         helper.make_tensor("maxout", TensorProto.INT64, [1], [10]),
         helper.make_tensor("iou", TensorProto.FLOAT, [1], [0.5]),
         helper.make_tensor("scoreth", TensorProto.FLOAT, [1], [0.1])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    export_graph("postprobe", m, {"boxes": boxes, "scores": scores})

def export_gridprobe():
    from onnx import helper, TensorProto
    N, Ch, H, W, Ho, Wo = 1, 3, 9, 11, 6, 7
    rng = np.random.RandomState(11)
    x = rng.randn(N, Ch, H, W).astype(np.float32)
    grid = (rng.rand(N, Ho, Wo, 2).astype(np.float32) * 2.6 - 1.3)  # includes out-of-range
    combos = [("bilinear", "zeros", 0), ("bilinear", "border", 1),
              ("bilinear", "reflection", 0), ("nearest", "zeros", 0),
              ("nearest", "border", 1), ("bilinear", "zeros", 1)]
    nodes, outs = [], []
    for i, (mode, pad, ac) in enumerate(combos):
        nodes.append(helper.make_node("GridSample", ["x", "grid"], [f"y{i}"],
                                      mode=mode, padding_mode=pad, align_corners=ac))
        outs.append(helper.make_tensor_value_info(f"y{i}", TensorProto.FLOAT, [N, Ch, Ho, Wo]))
    g = helper.make_graph(nodes, "gridprobe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [N, Ch, H, W]),
         helper.make_tensor_value_info("grid", TensorProto.FLOAT, [N, Ho, Wo, 2])],
        outs, [])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    export_graph("gridprobe", m, {"x": x, "grid": grid})

def export_ctrlprobe():
    from onnx import helper, TensorProto
    rng = np.random.RandomState(3)
    x = rng.randn(2, 3).astype(np.float32)
    cscale = helper.make_tensor("cscale", TensorProto.FLOAT, [1], [2.5])
    # If: both branches capture x from the outer scope.
    then_g = helper.make_graph(
        [helper.make_node("Mul", ["x", "cs"], ["ty"])], "then", [],
        [helper.make_tensor_value_info("ty", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor("cs", TensorProto.FLOAT, [1], [3.0])])
    else_g = helper.make_graph(
        [helper.make_node("Sub", ["x", "cs2"], ["ey"])], "else", [],
        [helper.make_tensor_value_info("ey", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor("cs2", TensorProto.FLOAT, [1], [1.0])])
    # Loop: M=4 trips, carried v starts at x, scans each intermediate.
    body = helper.make_graph(
        [helper.make_node("Mul", ["v_in", "cscale"], ["v_out"]),  # captures cscale
         helper.make_node("Identity", ["cond_in"], ["cond_out"]),
         helper.make_node("Identity", ["v_out"], ["scan0"])],
        "body",
        [helper.make_tensor_value_info("iter", TensorProto.INT64, []),
         helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
         helper.make_tensor_value_info("v_in", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
         helper.make_tensor_value_info("v_out", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("scan0", TensorProto.FLOAT, [2, 3])])
    nodes = [
        helper.make_node("If", ["condT"], ["yt"], then_branch=then_g, else_branch=else_g),
        helper.make_node("If", ["condF"], ["yf"], then_branch=then_g, else_branch=else_g),
        helper.make_node("Loop", ["M", "lcond", "x"], ["vfinal", "vscan"], body=body),
    ]
    g = helper.make_graph(nodes, "ctrlprobe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("yt", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("yf", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("vfinal", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("vscan", TensorProto.FLOAT, [4, 2, 3])],
        [cscale,
         helper.make_tensor("condT", TensorProto.BOOL, [], [True]),
         helper.make_tensor("condF", TensorProto.BOOL, [], [False]),
         helper.make_tensor("M", TensorProto.INT64, [], [4]),
         helper.make_tensor("lcond", TensorProto.BOOL, [], [True])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    export_graph("ctrlprobe", m, {"x": x})

def export_whileprobe():
    from onnx import helper, TensorProto
    rng = np.random.RandomState(5)
    x = np.abs(rng.randn(2, 3)).astype(np.float32) + 0.1
    # while sum(v) < 200: v *= 1.7   (data-dependent trip count)
    body = helper.make_graph(
        [helper.make_node("Mul", ["v_in", "growth"], ["v_out"]),
         helper.make_node("ReduceSum", ["v_out"], ["vsum"], keepdims=0),
         helper.make_node("Less", ["vsum", "limit"], ["cond_out"])],
        "wbody",
        [helper.make_tensor_value_info("iter", TensorProto.INT64, []),
         helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
         helper.make_tensor_value_info("v_in", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
         helper.make_tensor_value_info("v_out", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor("growth", TensorProto.FLOAT, [1], [1.7]),
         helper.make_tensor("limit", TensorProto.FLOAT, [], [200.0])])
    # zero-trip: M=0 passes the carried value through untouched.
    zbody = helper.make_graph(
        [helper.make_node("Identity", ["cond_in"], ["cond_out"]),
         helper.make_node("Mul", ["v_in", "v_in"], ["v_out"])],
        "zbody",
        [helper.make_tensor_value_info("iter", TensorProto.INT64, []),
         helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
         helper.make_tensor_value_info("v_in", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
         helper.make_tensor_value_info("v_out", TensorProto.FLOAT, [2, 3])])
    nodes = [
        helper.make_node("Loop", ["", "wcond", "x"], ["wfinal"], body=body),
        helper.make_node("Loop", ["zero", "wcond", "x"], ["zfinal"], body=zbody),
    ]
    g = helper.make_graph(nodes, "whileprobe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("wfinal", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("zfinal", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor("wcond", TensorProto.BOOL, [], [True]),
         helper.make_tensor("zero", TensorProto.INT64, [], [0])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    export_graph("whileprobe", m, {"x": x})

MODELS["ctrlprobe"] = export_ctrlprobe
MODELS["whileprobe"] = export_whileprobe

MODELS["postprobe"] = export_postprobe
MODELS["gridprobe"] = export_gridprobe

MODELS["tiny_conv_int8"] = lambda: export_int8("tiny_conv_int8", "tiny_conv")
MODELS["mobilenet_v3_small_int8"] = lambda: export_int8("mobilenet_v3_small_int8", "mobilenet_v3_small")


if __name__ == "__main__":
    names = sys.argv[1:] or MODELS
    for n in names:
        try: MODELS[n]()
        except Exception:
            print(f"{n}: ERROR\n{traceback.format_exc()}")

# --- Op-probe modules: exercise Pad/Resize/ConvTranspose variants directly ---
class OpProbe(nn.Module):
    def forward(self, x):
        a = F.pad(x, (1, 2, 2, 1), mode="reflect")
        b = F.pad(x, (2, 2, 1, 1), mode="replicate")
        c = F.pad(x, (1, 1, 1, 1), mode="constant", value=0.5)
        a = F.interpolate(a, scale_factor=2, mode="nearest")
        b = F.interpolate(b, size=(20, 20), mode="bilinear", align_corners=True)
        c = F.interpolate(c, scale_factor=1.5, mode="bilinear", align_corners=False)
        return a.mean() + b.mean() + c.mean()
class DeconvProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.ConvTranspose2d(4, 6, 3, stride=2, padding=1, output_padding=1)
        self.d2 = nn.ConvTranspose2d(6, 6, 2, stride=2, groups=2)
    def forward(self, x):
        return self.d2(F.relu(self.d1(x)))
