"""Rendered text lines in several scripts for ingot's PP-OCRv5 recognizer
tests: testdata/ocr/multilingual/<lang>_<i>.png + truth.json (committed).

Arabic is shaped with arabic-reshaper + python-bidi (PIL here has no raqm).
Usage: .venv/bin/python multilingual.py
"""
import os, json
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "ocr", "multilingual")
os.makedirs(OUT, exist_ok=True)
F = "/System/Library/Fonts"
S = F + "/Supplemental"

LINES = {
    # lang: (model, font, lines)
    "zh": ("ch", F + "/Hiragino Sans GB.ttc", ["文档布局分析与表格识别", "纯Go推理运行时", "今天的天气很好"]),
    "ja": ("ch", F + "/Hiragino Sans GB.ttc", ["日本語の文字認識", "東京駅から出発します"]),
    "ko": ("korean", F + "/AppleSDGothicNeo.ttc", ["한국어 문자 인식", "오늘은 날씨가 좋습니다"]),
    "ru": ("cyrillic", S + "/Arial.ttf", ["Распознавание текста", "Москва и Санкт-Петербург"]),
    "fr": ("latin", S + "/Arial.ttf", ["Reconnaissance des caractères", "Élève à côté du château"]),
    "de": ("latin", S + "/Arial.ttf", ["Größe und Übersetzung", "Straße in München"]),
    "es": ("latin", S + "/Arial.ttf", ["Reconocimiento de señales", "Mañana en Córdoba"]),
    "ar": ("arabic", S + "/Arial Unicode.ttf", ["التعرف على النصوص", "مرحبا بالعالم"]),
}


def shape(lang, s):
    if lang != "ar":
        return s
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(s))


def main():
    truth = []
    for lang, (model, font, lines) in LINES.items():
        f = ImageFont.truetype(font, 34)
        for i, text in enumerate(lines):
            shown = shape(lang, text)
            d0 = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            x0, y0, x1, y1 = d0.textbbox((0, 0), shown, font=f)
            img = Image.new("RGB", (x1 - x0 + 40, y1 - y0 + 30), "white")
            ImageDraw.Draw(img).text((20 - x0, 15 - y0), shown, font=f, fill="black")
            name = f"{lang}_{i}.png"
            img.save(os.path.join(OUT, name))
            truth.append({"image": name, "lang": lang, "model": model, "text": text})
    json.dump(truth, open(os.path.join(OUT, "truth.json"), "w"), ensure_ascii=False, indent=1)
    print(len(truth), "lines")


if __name__ == "__main__":
    main()
