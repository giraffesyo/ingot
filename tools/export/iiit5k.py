"""Prepare the IIIT5K-Word test split for models/ocr's TestIIIT5K.

Fetches IIIT5K-Word_V3.0.tar.gz (CVIT, ~100 MB) unless SRC points at an
extracted copy, then writes testdata/iiit5k/test/*.png and manifest.json
([{"image": "test/1_1.png", "text": "..."}, ...], 3000 words). The data is
gitignored (dataset license) — this script is the reproducible fetch.
Run: .venv/bin/python iiit5k.py [SRC_DIR]
"""
import io, json, os, shutil, sys, tarfile, urllib.request
import scipy.io

URL = "http://cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", "iiit5k")

def extracted_root(src):
    if src and os.path.isdir(src):
        return src
    cache = os.path.join(OUT, "_src")
    root = os.path.join(cache, "IIIT5K")
    if os.path.isdir(root):
        return root
    os.makedirs(cache, exist_ok=True)
    print("downloading", URL)
    data = urllib.request.urlopen(URL, timeout=600).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(cache)
    return root

def main():
    root = extracted_root(sys.argv[1] if len(sys.argv) > 1 else None)
    mat = scipy.io.loadmat(os.path.join(root, "testdata.mat"))["testdata"][0]
    os.makedirs(os.path.join(OUT, "test"), exist_ok=True)
    man = []
    for rec in mat:
        name = str(rec["ImgName"][0])        # "test/1002_1.png"
        text = str(rec["GroundTruth"][0])
        shutil.copyfile(os.path.join(root, name), os.path.join(OUT, name))
        man.append({"image": name, "text": text})
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=0)
    print(f"{len(man)} test words -> {OUT}")

if __name__ == "__main__":
    main()
