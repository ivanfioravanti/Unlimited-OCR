"""List, per page, the words (>= 4 letters, prose blocks only) that a raw Unlimited-OCR-2 output contains but the PDF page does not.
usage: python bench/unknown_words.py RAW.txt [RAW2.txt ...] [--pdf Unlimited-OCR-2.pdf]   (needs pymupdf)

This is a sharper quality check than score_vs_pdf.py: a misread word ("recessions" for "has", "alvettri") shows up here, while the
similarity score barely moves. Equations, tables, images, headers and page numbers are skipped. A few benign hits repeat in every
output (PDF line-break artefacts such as "hundred", "cross", "experts", "popo"), so compare outputs against each other.
"""
import argparse, os, re
import fitz

ap = argparse.ArgumentParser()
ap.add_argument("raw", nargs="+"); ap.add_argument("--pdf", default=os.path.join(os.path.dirname(__file__), "..", "Unlimited-OCR-2.pdf"))
a = ap.parse_args()
pdf = fitz.open(a.pdf)
tag = re.compile(r"<BLOCK>(\w+)(?:<LEVEL\d>)?\[[^\]]*\]<FORK>")
words = lambda s: re.findall(r"[a-z]{4,}", s.lower())
skip = {"equation", "image", "table", "figure", "image_caption", "table_caption", "formula", "header", "footer", "page_number", "other"}
for path in a.raw:
    pages = [p for p in open(path, encoding="utf-8").read().split("<PAGE>") if p]
    print(f"== {os.path.basename(path)}")
    for n, p in enumerate(pages):
        truth = set(words(pdf[n].get_text().replace("-\n", "")))
        parts = tag.split(p)  # ['', label, text, label, text, ...]
        txt = " ".join(parts[i + 1] for i in range(1, len(parts) - 1, 2) if parts[i] not in skip)
        txt = re.sub(r"\$[^$]*\$", " ", txt)  # inline math
        bad = [w for w in dict.fromkeys(words(txt)) if w not in truth]
        print(f"  page {n + 1}: {len(bad):2d} unknown words: {' '.join(bad[:14])}")
