"""Score a raw Unlimited-OCR-2 output against the PDF's own text layer (ground truth), per page.
usage: python bench/score_vs_pdf.py RAW.txt [RAW2.txt ...] [--pdf Unlimited-OCR-2.pdf]   (needs pymupdf)

Both sides are reduced to lowercase alphanumerics, so layout tags, bboxes, whitespace, hyphenation and punctuation do not count.
Equation pages (4 and 6 of the tech report) score low for every engine because the PDF text layer of formulas is not LaTeX; use
them only to compare engines with each other. A runaway shows up as fewer pages than expected and a very low score on its last page.
"""
import argparse, difflib, os, re
import fitz

ap = argparse.ArgumentParser()
ap.add_argument("raw", nargs="+"); ap.add_argument("--pdf", default=os.path.join(os.path.dirname(__file__), "..", "Unlimited-OCR-2.pdf"))
a = ap.parse_args()
norm = lambda s: re.sub(r"[^0-9a-z]", "", s.lower())
tags = re.compile(r"<BLOCK>\w+(?:<LEVEL\d>)?\[[^\]]*\]<FORK>|<image>|<PAGE>")
doc = fitz.open(a.pdf)

for path in a.raw:
    pages = [p for p in open(path, encoding="utf-8").read().split("<PAGE>") if p]
    row = []
    for n, p in enumerate(pages):
        truth, got = norm(doc[n].get_text()), norm(tags.sub("", p))
        row.append(f"{difflib.SequenceMatcher(None, truth, got, autojunk=False).ratio():.3f}")
    print(f"{os.path.basename(path):40s} pages_out={len(pages)}  per-page similarity vs PDF text: {' '.join(row)}")
