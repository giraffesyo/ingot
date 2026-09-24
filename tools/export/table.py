"""Synthetic table images + SLANet-plus structure references for ingot's
table tests.

  testdata/layout/tables/table_N.png  ruled tables (committed)
  testdata/layout/tables/truth.json   cell texts per row with colspans
  testdata/layout/tables/ref.json     RapidTable's SLANet-plus structure
                                      tokens, cell boxes (crop pixels, 8
                                      coords) and score per table

Usage: .venv/bin/python table.py   (needs testdata/layout/slanet-plus.onnx)
"""
import os, json
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "layout")
OUT = os.path.join(ROOT, "tables")
os.makedirs(OUT, exist_ok=True)
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def render(rows, colw=170, rowh=44, pad=24, size=19):
    """rows: list of rows; a cell is text or (text, colspan)."""
    f = ImageFont.truetype(FONT, size)
    ncol = sum(c[1] if isinstance(c, tuple) else 1 for c in rows[0])
    W, H = ncol * colw + 2 * pad, len(rows) * rowh + 2 * pad
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    truth = []
    for r, row in enumerate(rows):
        x, tr = 0, []
        for cell in row:
            text, span = cell if isinstance(cell, tuple) else (cell, 1)
            x0, y0 = pad + x * colw, pad + r * rowh
            x1, y1 = x0 + span * colw, y0 + rowh
            d.rectangle((x0, y0, x1, y1), outline="black", width=2)
            d.text((x0 + 12, y0 + 11), text, font=f, fill="black")
            tr.append({"text": text, "colspan": span})
            x += span
        truth.append(tr)
    return img, truth


TABLES = [
    [["model", "cpu ms", "gpu ms", "ratio"], ["detector", "17.4", "12.0", "1.45"],
     ["recognizer", "72.0", "59.0", "1.22"], ["parseq", "9.5", "6.0", "1.58"]],
    [["name", "value", "unit"], ["width", "1920", "px"], ["height", "1080", "px"],
     ["dpi", "300", "in"], ["pages", "12", "count"], ["size", "4.2", "MB"]],
    [[("latency by device", 3)], ["model", "cpu", "gpu"], ["vit", "92", "44"], ["bert", "61", "51"]],
]


def main():
    truth = []
    for i, rows in enumerate(TABLES):
        img, t = render(rows)
        name = f"table_{i}.png"
        img.save(os.path.join(OUT, name))
        truth.append({"image": name, "rows": t})
    json.dump(truth, open(os.path.join(OUT, "truth.json"), "w"), indent=1)

    model = os.path.join(ROOT, "slanet-plus.onnx")
    if not os.path.exists(model):
        print("no model; wrote tables only")
        return
    import cv2
    from rapid_table.table_structure.pp_structure import PPTableStructurer
    from rapid_table.utils.typings import ModelType, EngineType

    st = PPTableStructurer({"model_type": ModelType.SLANETPLUS, "model_dir_or_path": model,
                            "engine_type": EngineType.ONNXRUNTIME, "engine_cfg": {}})
    ref = []
    for t in truth:
        img = cv2.imread(os.path.join(OUT, t["image"]))
        structs, cells = st([img])
        tokens, score = structs[0]
        ref.append({"image": t["image"], "tokens": tokens, "score": score,
                    "cells": [[float(v) for v in c] for c in cells[0]]})
        print(t["image"], "".join(tokens)[:160], len(cells[0]), "cells", round(score, 3))
    json.dump(ref, open(os.path.join(OUT, "ref.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
