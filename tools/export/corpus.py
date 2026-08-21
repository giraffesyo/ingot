"""Generate a synthetic OCR test corpus with exact ground truth.

Each image contains several text lines placed at known positions with varied
fonts, sizes, colours, backgrounds, and a few rotations. Ground truth (text +
quadrilateral, in image pixel coords) is written to corpus/manifest.json for the
Go harness to score detection and recognition against.
"""
import os, json, random, math
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "ocr", "corpus")
os.makedirs(OUT, exist_ok=True)
random.seed(7)

FONTS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
]
FONTS = [f for f in FONTS if os.path.exists(f)]
CJK = [f for f in ["/System/Library/Fonts/Hiragino Sans GB.ttc",
                   "/System/Library/Fonts/STHeiti Medium.ttc"] if os.path.exists(f)]

WORDS = ("the quick brown fox jumps over lazy dog Invoice Total Amount Date Name "
         "Address Order Number Customer Product Quantity Price Subtotal Shipping "
         "Receipt Payment Balance Account Reference Description Model Serial").split()
def sentence(n): return " ".join(random.choice(WORDS) for _ in range(n))
def number():
    kinds = [lambda: f"{random.randint(1,9999)}",
             lambda: f"${random.randint(1,999)}.{random.randint(0,99):02d}",
             lambda: f"{random.randint(1,12):02d}/{random.randint(1,28):02d}/20{random.randint(20,26)}",
             lambda: f"REF-{random.randint(1000,9999)}",
             lambda: f"+1 {random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"]
    return random.choice(kinds)()
CJK_LINES = ["发票金额", "订单编号", "客户名称", "支付宝", "北京市海淀区", "总计一百二十元"]

def line_text():
    r = random.random()
    if r < 0.45: return sentence(random.randint(1, 4)), False
    if r < 0.75: return number(), False
    if r < 0.9 and CJK: return random.choice(CJK_LINES), True
    return random.choice(WORDS).capitalize(), False

def quad_of(x, y, w, h, angle):
    cx, cy = x + w/2, y + h/2
    c, s = math.cos(angle), math.sin(angle)
    pts = []
    for dx, dy in [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]:
        pts.append([cx + dx*c - dy*s, cy + dx*s + dy*c])
    return pts

def gen_image(idx):
    W, H = random.choice([(480, 360), (640, 480), (400, 500), (720, 300)])
    bg = random.choice([(255,255,255),(245,245,240),(250,250,255),(255,252,245),(230,235,240)])
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    gt = []
    y = random.randint(15, 40)
    nlines = random.randint(3, 7)
    for _ in range(nlines):
        if y > H - 40: break
        text, is_cjk = line_text()
        size = random.randint(18, 34)
        fpath = random.choice(CJK) if is_cjk else random.choice(FONTS)
        try: font = ImageFont.truetype(fpath, size)
        except Exception: font = ImageFont.load_default()
        x = random.randint(15, 60)
        bbox = d.textbbox((x, y), text, font=font)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        if x + w > W - 10:
            # shrink text to fit
            text = text[: max(1, int(len(text) * (W - 20 - x) / max(w,1)))]
            bbox = d.textbbox((x, y), text, font=font); w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        col = random.choice([(0,0,0),(20,20,60),(60,20,20),(40,40,40)])
        angle = 0.0
        if random.random() < 0.15:  # occasional slight rotation
            angle = random.uniform(-0.12, 0.12)
        if angle == 0.0:
            d.text((x, y), text, fill=col, font=font)
            quad = [[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]]]
        else:
            # render on a transparent layer, rotate, paste
            pad = 8
            tl = Image.new("RGBA", (w+2*pad, h+2*pad), (0,0,0,0))
            ImageDraw.Draw(tl).text((pad-bbox[0]+x-x, pad-(bbox[1]-y)), text, fill=col+(255,), font=font)
            tl = tl.rotate(math.degrees(-angle), expand=True, resample=Image.BICUBIC)
            px, py = x-pad, y-pad
            img.paste(tl, (px, py), tl)
            quad = quad_of(x-bbox[0]+bbox[0], y, w, h, angle)
            # recompute quad center around placed text
            quad = quad_of(x, bbox[1], w, h, angle)
        gt.append({"text": text, "quad": [[float(a),float(b)] for a,b in quad]})
        y = int(bbox[3]) + random.randint(10, 30)
    name = f"img_{idx:03d}.png"
    img.save(os.path.join(OUT, name))
    return {"image": name, "w": W, "h": H, "lines": gt}

def main(n=24):
    manifest = [gen_image(i) for i in range(n)]
    manifest = [m for m in manifest if m["lines"]]
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    total = sum(len(m["lines"]) for m in manifest)
    print(f"generated {len(manifest)} images, {total} text lines -> {OUT}")

if __name__ == "__main__":
    main()
