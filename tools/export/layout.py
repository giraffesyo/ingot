"""Synthetic document pages + PP-DocLayoutV3 references for ingot's layout
tests.

Renders a few pages (title, section headings, one- and two-column body text,
a ruled table, a figure, header/footer) with exact ground-truth regions, then
runs the RapidLayout reference (PP-DocLayoutV3 on ONNX Runtime) over them:

  testdata/layout/pages/page_N.png   the pages (committed)
  testdata/layout/pages/truth.json   ground-truth regions per page, in
                                     reading order (committed)
  testdata/layout/pages/ref.json     RapidLayout's regions per page
                                     (label, score, box), its reading order
  testdata/layout/v3_in_*.bin,       page 0's exact model inputs and raw
  testdata/layout/v3_out_*.bin       outputs (f32/i32 little-endian) + json

Usage: .venv/bin/python layout.py   (needs testdata/layout/pp_doc_layoutv3.onnx)
"""
import os, json, random, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "layout")
PAGES = os.path.join(ROOT, "pages")
os.makedirs(PAGES, exist_ok=True)
random.seed(11)

FONT_DIR = "/System/Library/Fonts/Supplemental"


def font(name, size):
    for cand in (os.path.join(FONT_DIR, name), *glob.glob(f"/usr/share/fonts/**/{name}", recursive=True)):
        if os.path.exists(cand):
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


WORDS = ("the model runs every layer of the network on the processor with packed weights and "
         "measured kernels so that each step of the pipeline reads its inputs once and writes "
         "results in place while the scheduler keeps workers busy across the whole graph").split()


def sentence(n):
    w = [random.choice(WORDS) for _ in range(n)]
    w[0] = w[0].capitalize()
    return " ".join(w) + "."


def wrap(draw, text, f, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


class Page:
    def __init__(self, w=1000, h=1300):
        self.img = Image.new("RGB", (w, h), "white")
        self.d = ImageDraw.Draw(self.img)
        self.w, self.h = w, h
        self.regions = []  # {label, box [x1,y1,x2,y2], text}

    def add(self, label, box, text=""):
        self.regions.append({"label": label, "box": [int(round(v)) for v in box], "text": text})

    def block(self, label, x, y, width, text, f, gap=6):
        lines = wrap(self.d, text, f, width)
        y0, lh = y, f.size + gap
        for i, ln in enumerate(lines):
            self.d.text((x, y + i * lh), ln, font=f, fill="black")
        bottom = y + len(lines) * lh - gap
        right = x + max(self.d.textlength(ln, font=f) for ln in lines)
        self.add(label, (x, y0, right, bottom + 4), " ".join(lines))
        return bottom + 4


def header_footer(p, n):
    f = font("Arial.ttf", 14)
    p.d.text((60, 30), "ingot technical report", font=f, fill="gray")
    p.add("header", (60, 30, 60 + p.d.textlength("ingot technical report", font=f), 48), "ingot technical report")
    s = f"page {n}"
    p.d.text((p.w / 2 - 20, p.h - 50), s, font=f, fill="gray")
    p.add("number", (p.w / 2 - 20, p.h - 50, p.w / 2 - 20 + p.d.textlength(s, font=f), p.h - 32), s)


def page_single(n):
    p = Page()
    header_footer(p, n)
    title, body, head = font("Arial Bold.ttf", 34), font("Times New Roman.ttf", 20), font("Arial Bold.ttf", 24)
    y = p.block("doc_title", 60, 90, 880, "A Pure Go Runtime for Document Models", title) + 30
    for sec in ("Introduction", "Method"):
        y = p.block("paragraph_title", 60, y, 880, sec, head) + 14
        for _ in range(2):
            y = p.block("text", 60, y, 880, " ".join(sentence(14) for _ in range(4)), body) + 22
    return p


def page_two_col(n):
    p = Page()
    header_footer(p, n)
    title, body, head = font("Arial Bold.ttf", 30), font("Times New Roman.ttf", 18), font("Arial Bold.ttf", 22)
    y = p.block("doc_title", 60, 90, 880, "Two Column Layout Study", title) + 30
    colw, xs = 420, (60, 520)
    for x in xs:
        yy = y
        yy = p.block("paragraph_title", x, yy, colw, "Results" if x == xs[0] else "Discussion", head) + 12
        for _ in range(3):
            yy = p.block("text", x, yy, colw, " ".join(sentence(12) for _ in range(3)), body) + 20
    return p


def page_table(n):
    p = Page()
    header_footer(p, n)
    body, head, cell = font("Times New Roman.ttf", 20), font("Arial Bold.ttf", 24), font("Arial.ttf", 18)
    y = p.block("paragraph_title", 60, 90, 880, "Benchmark Results", head) + 14
    y = p.block("text", 60, y, 880, " ".join(sentence(14) for _ in range(3)), body) + 30
    rows = [["model", "cpu ms", "gpu ms", "ratio"], ["detector", "17.4", "12.0", "1.45"], ["recognizer", "72.0", "59.0", "1.22"],
            ["parseq", "9.5", "6.0", "1.58"], ["mobilenet", "1.7", "1.3", "1.31"]]
    cw, rh, x0, y0 = 200, 40, 100, y
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            p.d.rectangle((x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh), outline="black", width=2)
            p.d.text((x0 + c * cw + 12, y0 + r * rh + 10), v, font=cell, fill="black")
    tbl = (x0, y0, x0 + len(rows[0]) * cw, y0 + len(rows) * rh)
    p.add("table", tbl, json.dumps(rows))
    y = tbl[3] + 12
    y = p.block("figure_title", 100, y, 800, "Table 1: Forward latency per model.", font("Arial Italic.ttf", 18)) + 30
    y = p.block("text", 60, y, 880, " ".join(sentence(14) for _ in range(3)), body)
    return p


def page_figure(n):
    p = Page()
    header_footer(p, n)
    body, head = font("Times New Roman.ttf", 20), font("Arial Bold.ttf", 24)
    y = p.block("paragraph_title", 60, 90, 880, "Architecture", head) + 14
    y = p.block("text", 60, y, 880, " ".join(sentence(14) for _ in range(3)), body) + 30
    # A "figure": boxes and arrows.
    fx0, fy0, fx1, fy1 = 200, y, 800, y + 300
    rng = random.Random(3)
    for i in range(4):
        bx = fx0 + 20 + i * 145
        p.d.rectangle((bx, fy0 + 110, bx + 110, fy0 + 190), fill=(200 + rng.randint(0, 40), 220, 255), outline="navy", width=3)
        if i:
            p.d.line((bx - 35, fy0 + 150, bx, fy0 + 150), fill="navy", width=4)
    p.d.ellipse((fx0 + 250, fy0 + 10, fx0 + 350, fy0 + 90), outline="darkred", width=4)
    p.add("image", (fx0, fy0, fx1, fy1))
    y = fy1 + 12
    y = p.block("figure_title", 220, y, 560, "Figure 1: The runtime layers.", font("Arial Italic.ttf", 18)) + 30
    y = p.block("text", 60, y, 880, " ".join(sentence(14) for _ in range(4)), body)
    return p


def main():
    pages = [page_single(1), page_two_col(2), page_table(3), page_figure(4)]
    truth = []
    for i, p in enumerate(pages):
        name = f"page_{i}.png"
        p.img.save(os.path.join(PAGES, name))
        # Reading order: the page number closes the page (header_footer
        # drew it first).
        regs = [r for r in p.regions if r["label"] != "number"] + [r for r in p.regions if r["label"] == "number"]
        truth.append({"image": name, "regions": regs})
    json.dump(truth, open(os.path.join(PAGES, "truth.json"), "w"), indent=1)

    model = os.path.join(ROOT, "pp_doc_layoutv3.onnx")
    if not os.path.exists(model):
        print("no model; wrote pages only")
        return
    import cv2
    from rapid_layout import RapidLayout
    from rapid_layout.model_handler.pp_doc_layout.pre_process import PPDocLayoutPreProcess
    import onnxruntime as ort

    eng = RapidLayout(model_type="pp_doc_layoutv3", model_dir_or_path=model)
    ref = []
    for t in truth:
        img = cv2.imread(os.path.join(PAGES, t["image"]))
        out = eng(img)
        ref.append({"image": t["image"], "regions": [
            {"label": l, "score": float(s), "box": [float(v) for v in b]}
            for b, s, l in zip(out.boxes, out.scores, out.class_names)]})
        print(t["image"], [(r["label"], round(r["score"], 2)) for r in ref[-1]["regions"]])
    json.dump(ref, open(os.path.join(PAGES, "ref.json"), "w"), indent=1)

    # Raw model I/O for page 0 (parity of the graph itself).
    img = cv2.imread(os.path.join(PAGES, truth[0]["image"]))
    _, inputs = PPDocLayoutPreProcess(img_size=(800, 800))(img)
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    feeds = dict(zip(names, inputs))
    outs = sess.run(None, feeds)
    meta = {"inputs": [], "outputs": []}
    for k, v in feeds.items():
        fn = f"v3_in_{k}.bin"
        v.astype(np.float32).tofile(os.path.join(ROOT, fn))
        meta["inputs"].append({"name": k, "file": fn, "shape": list(v.shape), "dtype": "float32"})
    for o, v in zip(sess.get_outputs(), outs):
        fn = f"v3_out_{o.name}.bin"
        v.tofile(os.path.join(ROOT, fn))
        meta["outputs"].append({"name": o.name, "file": fn, "shape": list(v.shape), "dtype": str(v.dtype)})
    json.dump(meta, open(os.path.join(ROOT, "v3_io.json"), "w"), indent=1)
    print("raw io:", [(m["name"], m["shape"], m["dtype"]) for m in meta["outputs"]])


if __name__ == "__main__":
    main()
