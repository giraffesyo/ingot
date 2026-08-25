"""Time the PP-OCR det/rec ONNX models in ONNX Runtime at fixed shapes.

Usage: .venv/bin/python orttime.py [threads]
Shapes match models/ocr/bench_test.go (BenchmarkOCRModels) for like-for-like.
"""
import sys, time, os
import numpy as np
import onnxruntime as ort

D = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "ocr")
SHAPES = [
    ("det_640", "det.onnx", (1, 3, 640, 640)),
    ("det_960", "det.onnx", (1, 3, 960, 960)),
    ("rec_320", "rec.onnx", (1, 3, 48, 320)),
    ("rec_b8_320", "rec.onnx", (8, 3, 48, 320)),
    ("det_int8_640", "det_int8.onnx", (1, 3, 640, 640)),
    ("rec_int8_320", "rec_int8.onnx", (1, 3, 48, 320)),
]

def main():
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    print(f"onnxruntime {ort.__version__} threads={threads or 'default'}")
    for name, model, shape in SHAPES:
        s = ort.InferenceSession(os.path.join(D, model), so, providers=["CPUExecutionProvider"])
        inp = s.get_inputs()[0].name
        x = ((np.arange(int(np.prod(shape))) % 97) / 97 - 0.5).astype(np.float32).reshape(shape)
        for _ in range(5):
            s.run(None, {inp: x})
        n = 30 if name.startswith("det") else 100
        t0 = time.perf_counter()
        for _ in range(n):
            s.run(None, {inp: x})
        dt = (time.perf_counter() - t0) / n
        print(f"{name:12s} {dt*1e3:8.2f} ms")

if __name__ == "__main__":
    main()
