"""Quantize the PP-OCR det/rec models (convs only, QOperator) with calibration
from the synthetic corpus, and dump ORT reference outputs for parity tests.

Usage: .venv/bin/python ocr_int8.py
Writes testdata/ocr/{det_int8,rec_int8}.onnx, plus .in.bin/.out.bin pairs.
"""
import glob
import os

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
import onnx
from onnx import numpy_helper


def constants_to_initializers(src, dst):
    """The PP-OCR exports keep every weight as a Constant node; the quantizer
    requires initializers."""
    m = onnx.load(src)
    keep = []
    for n in m.graph.node:
        if n.op_type == "Constant" and len(n.attribute) == 1 and n.attribute[0].name == "value":
            t = n.attribute[0].t
            t.name = n.output[0]
            m.graph.initializer.append(t)
        else:
            keep.append(n)
    del m.graph.node[:]
    m.graph.node.extend(keep)
    onnx.save(m, dst)
from PIL import Image

D = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "ocr")
IMGS = sorted(glob.glob(os.path.join(D, "corpus", "*.png")))[:12]

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def det_pre(path, limit=960):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(limit / max(w, h), 1.0)
    nw = max(32, round(w * scale / 32) * 32)
    nh = max(32, round(h * scale / 32) * 32)
    im = im.resize((nw, nh), Image.BILINEAR)
    x = np.asarray(im, np.float32)
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]


def rec_crops():
    """Real text-line crops from the corpus ground truth — rec calibration
    must see tight line crops, not downscaled pages (a whole-page 'crop'
    mis-scales every activation and wrecks accuracy)."""
    import json
    man = json.load(open(os.path.join(D, "corpus", "manifest.json")))
    crops = []
    for gi in man:
        im = Image.open(os.path.join(D, "corpus", gi["image"])).convert("RGB")
        for ln in gi["lines"]:
            xs = [p[0] for p in ln["quad"]]
            ys = [p[1] for p in ln["quad"]]
            box = (max(0, min(xs)), max(0, min(ys)), min(im.width, max(xs)), min(im.height, max(ys)))
            if box[2]-box[0] < 4 or box[3]-box[1] < 4:
                continue
            c = im.crop(box)
            w = max(4, min(320, round(48 * c.width / c.height)))
            c = c.resize((w, 48), Image.BILINEAR)
            x = np.zeros((48, 320, 3), np.float32)
            x[:, :w] = np.asarray(c, np.float32)
            x = (x / 255 - 0.5) / 0.5
            crops.append(x.transpose(2, 0, 1)[None])
            if len(crops) >= 64:
                return crops
    return crops


class Reader(CalibrationDataReader):
    def __init__(self, name, pre):
        if pre is None:
            self.data = [{name: c} for c in rec_crops()]
        else:
            self.data = [{name: pre(p)} for p in IMGS]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        self.i += 1
        return self.data[self.i - 1]


def quantize(model, pre, ref_shape):
    src = os.path.join(D, f"{model}.onnx")
    pre_src = os.path.join(D, f"{model}_pre.onnx")
    dst = os.path.join(D, f"{model}_int8.onnx")
    # fold Constant-node weights into initializers (quantizer requirement)
    constants_to_initializers(src, pre_src)
    quant_pre_process(pre_src, pre_src, skip_symbolic_shape=True)
    s0 = ort.InferenceSession(pre_src, providers=["CPUExecutionProvider"])
    iname = s0.get_inputs()[0].name
    quantize_static(pre_src, dst, Reader(iname, pre), quant_format=QuantFormat.QOperator,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, op_types_to_quantize=["Conv"])
    sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(7)
    x = rng.standard_normal(ref_shape).astype(np.float32)
    y = sess.run(None, {iname: x})[0]
    x.tofile(os.path.join(D, f"{model}_int8.in.bin"))
    np.asarray(y).tofile(os.path.join(D, f"{model}_int8.out.bin"))
    os.remove(pre_src)
    m = onnx.load(dst)
    print(f"{model}_int8: {len(m.graph.node)} nodes, in {ref_shape} out {list(np.asarray(y).shape)},",
          " ".join(sorted({n.op_type for n in m.graph.node})))


quantize("det", det_pre, (1, 3, 640, 640))
quantize("rec", None, (1, 3, 48, 320))
