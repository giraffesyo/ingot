"""Real-document evaluation: a small DocLayNet + PubTabNet sample, reference
predictions from the Python tools, and scoring for both them and ingot.

  realdocs.py fetch   ~60 DocLayNet v1.2 test pages (10 per document
                      category) and ~60 PubTabNet val tables into
                      testdata/realdocs/ (gitignored; ~10-15 MB kept) via
                      the Hugging Face dataset-server rows API — no shard
                      downloads.
  realdocs.py ref     RapidLayout (PP-DocLayoutV3) on the pages and
                      RapidTable (SLANet-plus + RapidOCR) on the tables.
  realdocs.py score   layout F1@0.5 (class-agnostic and DocLayNet-mapped),
                      page text CER, table TEDS — for ref.json and, when
                      present, ingot.json (written by the Go test
                      TestRealDocs with INGOT_REALDOCS=1).

Datasets: DocLayNet (CDLA-Permissive-1.0), PubTabNet (CDLA-Permissive-1.0).
"""
import os, sys, json, io, time, urllib.request, urllib.parse

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "realdocs")
PAGES = os.path.join(ROOT, "doclaynet")
TABLES = os.path.join(ROOT, "pubtabnet")
SERVER = "https://datasets-server.huggingface.co/rows"
DLN_CLASSES = ["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header",
               "Picture", "Section-header", "Table", "Text", "Title"]


def rows(dataset, split, offset, length):
    q = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": split,
                                "offset": offset, "length": length})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{SERVER}?{q}", timeout=120) as r:
                return json.load(r)["rows"]
        except Exception as e:  # the server is occasionally slow
            print("retry", attempt, e)
            time.sleep(3)
    raise SystemExit("dataset-server unavailable")


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def fetch_pages(per_cat=10):
    os.makedirs(PAGES, exist_ok=True)
    counts, gt = {}, []
    for off in range(0, 5000, 500):  # spread over the split; 20 rows per probe
        for row in rows("docling-project/DocLayNet-v1.2", "test", off, 20):
            r = row["row"]
            cat = r["metadata"]["doc_category"]
            if counts.get(cat, 0) >= per_cat:
                continue
            counts[cat] = counts.get(cat, 0) + 1
            name = f"page_{len(gt):03d}.png"
            with open(os.path.join(PAGES, name), "wb") as f:
                f.write(get(r["image"]["src"]))
            regions = []
            for i, (b, c) in enumerate(zip(r["bboxes"], r["category_id"])):
                cells = r["pdf_cells"][i] if i < len(r["pdf_cells"]) else []
                regions.append({"label": DLN_CLASSES[c - 1], "box": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
                                "text": " ".join(c["text"] for c in cells)})
            gt.append({"image": name, "category": cat, "regions": regions})
        if all(counts.get(c, 0) >= per_cat for c in counts) and len(counts) >= 6:
            break
    json.dump(gt, open(os.path.join(PAGES, "gt.json"), "w"), indent=1)
    print("pages:", counts)


def pubtabnet_html(ann):
    """PubTabNet annotation (structure tokens + cell tokens) → HTML table."""
    struct = ann["structure"]["tokens"]
    cells = ann["cells"]
    out, ci = [], 0
    for tok in struct:
        out.append(tok)
        if tok in ("<td>", ">"):  # a cell's content follows its opening
            if ci < len(cells):
                out.append("".join(cells[ci]["tokens"]))
            ci += 1
    return "<html><body><table>" + "".join(out) + "</table></body></html>"


def fetch_tables(n=60):
    import ast
    os.makedirs(TABLES, exist_ok=True)
    gt = []
    for row in rows("apoidea/pubtabnet-html", "validation", 0, n):
        r = row["row"]
        ann = ast.literal_eval(r["html"])
        name = f"table_{len(gt):03d}.jpg"
        with open(os.path.join(TABLES, name), "wb") as f:
            f.write(get(r["image"]["src"]))
        gt.append({"image": name, "html": pubtabnet_html(ann)})
    json.dump(gt, open(os.path.join(TABLES, "gt.json"), "w"), indent=1)
    print("tables:", len(gt))


def ref():
    import cv2
    from rapid_layout import RapidLayout
    layout = RapidLayout(model_type="pp_doc_layoutv3",
                         model_dir_or_path=os.path.join(ROOT, "..", "layout", "pp_doc_layoutv3.onnx"))
    out = {"pages": [], "tables": []}
    for p in json.load(open(os.path.join(PAGES, "gt.json"))):
        o = layout(cv2.imread(os.path.join(PAGES, p["image"])))
        out["pages"].append({"image": p["image"], "regions": [
            {"label": l, "score": float(s), "box": [float(v) for v in b]}
            for b, s, l in zip(o.boxes, o.scores, o.class_names)]})
    from rapid_table import RapidTable, RapidTableInput
    tab = RapidTable(RapidTableInput(model_type="slanet_plus",
                                     model_dir_or_path=os.path.join(ROOT, "..", "layout", "slanet-plus.onnx")))
    for t in json.load(open(os.path.join(TABLES, "gt.json"))):
        res = tab(os.path.join(TABLES, t["image"]))
        out["tables"].append({"image": t["image"], "html": res.pred_htmls[0] if res.pred_htmls else ""})
    out["text_cer"] = ref_text_cer()
    json.dump(out, open(os.path.join(ROOT, "ref.json"), "w"), indent=1)
    print("ref: %d pages, %d tables" % (len(out["pages"]), len(out["tables"])))


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def ref_text_cer():
    """RapidOCR (PP-OCRv4 det+rec, its defaults) text per ground-truth
    region, scored exactly as the Go test scores ingot's."""
    from rapidocr import RapidOCR
    eng = RapidOCR()
    edits = chars = 0
    for p in json.load(open(os.path.join(PAGES, "gt.json"))):
        res = eng(os.path.join(PAGES, p["image"]))
        lines = []
        for box, txt in zip(res.boxes if res.boxes is not None else [], res.txts or []):
            xs, ys = [q[0] for q in box], [q[1] for q in box]
            lines.append(((min(ys) + max(ys)) / 2, min(xs), (min(xs) + max(xs)) / 2, txt))
        for g in p["regions"]:
            if not g["text"] or g["label"] in ("Picture", "Table", "Formula"):
                continue
            x0, y0, x1, y1 = g["box"]
            inr = sorted([l for l in lines if x0 <= l[2] <= x1 and y0 <= l[0] <= y1], key=lambda l: (round(l[0] / 10), l[1]))
            want = " ".join(g["text"].split())
            got = " ".join(" ".join(l[3] for l in inr).split())
            edits += lev(want, got)
            chars += len(want)
    print("ref text CER %.4f (%d / %d)" % (edits / max(chars, 1), edits, chars))
    return edits / max(chars, 1)


# PP-DocLayoutV3 label → DocLayNet class (None: not scored). DocLayNet's
# List-item and Text are both "Text" here: the model has no list class.
MAP = {"text": "Text", "abstract": "Text", "content": "Text", "reference": "Text", "reference_content": "Text",
       "algorithm": "Text", "aside_text": "Text", "vertical_text": "Text", "paragraph_title": "Section-header",
       "doc_title": "Title", "header": "Page-header", "header_image": "Page-header", "footer": "Page-footer",
       "footer_image": "Page-footer", "number": "Page-footer", "footnote": "Footnote", "vision_footnote": "Footnote",
       "table": "Table", "image": "Picture", "chart": "Picture", "seal": "Picture", "figure_title": "Caption",
       "display_formula": "Formula", "formula_number": "Formula", "inline_formula": None}


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def layout_f1(gt_pages, pred_pages, classes):
    tp = fp = fn = 0
    for g, p in zip(gt_pages, pred_pages):
        G = [(("Text" if r["label"] == "List-item" else r["label"]) if classes else "*", r["box"]) for r in g["regions"]]
        P = []
        for r in p["regions"]:
            m = MAP.get(r["label"], "Text")
            if m is None:
                continue
            P.append((m if classes else "*", r["box"]))
        used = set()
        for gl, gb in G:
            best, bi = 0.5, -1
            for i, (pl, pb) in enumerate(P):
                if i in used or pl != gl:
                    continue
                v = iou(gb, pb)
                if v >= best:
                    best, bi = v, i
            if bi >= 0:
                used.add(bi)
                tp += 1
            else:
                fn += 1
        fp += len(P) - len(used)
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return 2 * prec * rec / max(1e-9, prec + rec), prec, rec


def teds(pred, gt, structure_only=False):
    from apted import APTED, Config
    from apted.helpers import Tree
    from lxml import html as lh
    import distance

    class N:
        def __init__(self, tag, colspan=1, rowspan=1, content=""):
            self.tag, self.colspan, self.rowspan, self.content, self.children = tag, colspan, rowspan, content, []

    def tree(e):
        n = N(e.tag, int(e.attrib.get("colspan", 1)), int(e.attrib.get("rowspan", 1)),
              "".join(e.itertext()) if e.tag == "td" else "")
        if e.tag != "td":
            n.children = [tree(c) for c in e]
        return n

    class C(Config):
        def rename(self, a, b):
            if a.tag != b.tag or a.colspan != b.colspan or a.rowspan != b.rowspan:
                return 1.0
            if a.tag == "td":
                if structure_only or not a.content and not b.content:
                    return 0.0
                return distance.nlevenshtein(a.content, b.content)
            return 0.0

        def children(self, n):
            return n.children

    def count(n):
        return 1 + sum(count(c) for c in n.children)

    def norm(h):  # thead/tbody are presentation: compare rows and cells
        for t in ("<thead>", "</thead>", "<tbody>", "</tbody>"):
            h = h.replace(t, "")
        return h

    try:
        tp = lh.fromstring(norm(pred)).xpath("//table")[0]
    except Exception:
        return 0.0
    tg = lh.fromstring(norm(gt)).xpath("//table")[0]
    a, b = tree(tp), tree(tg)
    d = APTED(a, b, C()).compute_edit_distance()
    return 1 - d / max(count(a), count(b))


def score():
    gtp = json.load(open(os.path.join(PAGES, "gt.json")))
    gtt = json.load(open(os.path.join(TABLES, "gt.json")))
    for name in ("ref.json", "ingot.json", "ingot_upscale.json"):
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            continue
        pr = json.load(open(path))
        f1a = layout_f1(gtp, pr["pages"], False)
        f1c = layout_f1(gtp, pr["pages"], True)
        t = [teds(p["html"], g["html"]) for p, g in zip(pr["tables"], gtt)]
        ts = [teds(p["html"], g["html"], True) for p, g in zip(pr["tables"], gtt)]
        print(f"{name}: layout F1@0.5 class-agnostic {f1a[0]:.3f} (P {f1a[1]:.3f} R {f1a[2]:.3f}), "
              f"class-mapped {f1c[0]:.3f}; TEDS {sum(t)/max(1,len(t)):.3f}, TEDS-S {sum(ts)/max(1,len(ts)):.3f} "
              f"over {len(t)} tables")
        if "text_cer" in pr:
            print(f"  page text CER {pr['text_cer']:.4f}")
        cats = {}
        for g, p in zip(gtp, pr["pages"]):
            cats.setdefault(g["category"], ([], []))
            cats[g["category"]][0].append(g)
            cats[g["category"]][1].append(p)
        for c, (gs, ps) in sorted(cats.items()):
            print(f"    {c:22s} class-agnostic F1 {layout_f1(gs, ps, False)[0]:.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fetch":
        fetch_pages()
        fetch_tables()
    elif cmd == "ref":
        ref()
    elif cmd == "score":
        score()
    else:
        print(__doc__)
