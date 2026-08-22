"""Time every zoo model (testdata/models/*.json manifests) in ONNX Runtime.

Usage: .venv/bin/python zootime.py [threads]
Same inputs as the Go conformance/benchmark harness (graph.BenchmarkModels).
"""
import sys, os, json, glob, time
import numpy as np
import onnxruntime as ort

D = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "models")
DT = {"float32": np.float32, "int64": np.int64, "int32": np.int32, "bool": np.bool_}

def main():
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    print(f"onnxruntime {ort.__version__} threads={threads or 'default'}")
    for mf in sorted(glob.glob(os.path.join(D, "*.json"))):
        man = json.load(open(mf))
        name = os.path.basename(mf)[:-5]
        try:
            s = ort.InferenceSession(os.path.join(D, man["model"]), so, providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"{name:20s} load failed: {e}")
            continue
        feeds = {}
        for inp in man["inputs"]:
            arr = np.fromfile(os.path.join(D, inp["file"]), dtype=DT[inp["dtype"]]).reshape(inp["shape"])
            feeds[inp["name"]] = arr
        for _ in range(10):
            s.run(None, feeds)
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 0.5:
            s.run(None, feeds); n += 1
        dt = (time.perf_counter() - t0) / n
        print(f"{name:20s} {dt*1e6:10.1f} µs")

if __name__ == "__main__":
    main()
