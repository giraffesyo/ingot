"""Per-op profile of the PP-OCR det/rec models in ONNX Runtime.

Usage: .venv/bin/python ortprof.py <det_640|det_960|rec_320|rec_b8_320> [threads]
Aggregates ORT's JSON profile by op type and prints the slowest nodes, so
"where does ORT spend its time" can be compared with models/ocr TestOCRProfile.
"""
import sys, os, json, collections
import numpy as np
import onnxruntime as ort

D = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "ocr")
SHAPES = {
    "det_640": ("det.onnx", (1, 3, 640, 640)),
    "det_960": ("det.onnx", (1, 3, 960, 960)),
    "rec_320": ("rec.onnx", (1, 3, 48, 320)),
    "rec_b8_320": ("rec.onnx", (8, 3, 48, 320)),
}

def main():
    name = sys.argv[1]
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    model, shape = SHAPES[name]
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    so.enable_profiling = True
    so.profile_file_prefix = "/tmp/ortprof"
    s = ort.InferenceSession(os.path.join(D, model), so, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0].name
    x = ((np.arange(int(np.prod(shape))) % 97) / 97 - 0.5).astype(np.float32).reshape(shape)
    runs = 30
    for _ in range(5 + runs):
        s.run(None, {inp: x})
    prof = s.end_profiling()
    ev = json.load(open(prof))
    os.remove(prof)
    # Only "Node" category events with dur; skip the first 5 warm-up runs by
    # keeping the last `runs` occurrences per node name.
    per_node = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
            per_node[(e["name"], e["args"].get("op_name"))].append(e["dur"])
    by_op = collections.defaultdict(lambda: [0, 0.0])
    nodes = []
    total = 0.0
    for (nm, op), durs in per_node.items():
        durs = durs[-runs:]
        us = sum(durs) / len(durs)
        by_op[op][0] += 1
        by_op[op][1] += us
        total += us
        nodes.append((us, op, nm))
    print(f"onnxruntime {ort.__version__} {name} threads={threads or 'default'}: {total/1e3:.2f} ms/run (sum of node kernel times)")
    for op, (n, us) in sorted(by_op.items(), key=lambda kv: -kv[1][1]):
        print(f"  {op:22s} n={n:3d} {us:9.1f} µs/run {100*us/total:5.1f}%")
    nodes.sort(reverse=True)
    for us, op, nm in nodes[:25]:
        print(f"    {op:10s} {nm:50s} {us:8.1f} µs")

if __name__ == "__main__":
    main()
