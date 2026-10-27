"""Render the first N pages of the tech report to bench/doc/page_<n>.jpg (identical input on every machine).
usage: python bench/prepare_doc.py [--pages 6] [--dpi 150]   (needs pymupdf)
"""
import argparse, hashlib, os, urllib.request
import fitz

ap = argparse.ArgumentParser()
ap.add_argument("--pages", type=int, default=6); ap.add_argument("--dpi", type=int, default=150)
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
pdf = os.path.join(here, "..", "Unlimited-OCR-2.pdf")
SHA = "72e95cde363b991cdde1c8d8b0f50dccd40e2e0521ceaf0ca490995cfdb91c49"
if not open(pdf, "rb").read(5) == b"%PDF-":  # git-LFS pointer, fetch the real file
    pdf = os.path.join(here, "report.pdf")
    if not os.path.exists(pdf):
        urllib.request.urlretrieve("https://github.com/baidu/Unlimited-OCR/raw/main/Unlimited-OCR-2.pdf", pdf)
assert hashlib.sha256(open(pdf, "rb").read()).hexdigest() == SHA, "unexpected PDF"
out = os.path.join(here, "doc"); os.makedirs(out, exist_ok=True)
for i, p in enumerate(fitz.open(pdf)):
    if i >= a.pages: break
    p.get_pixmap(matrix=fitz.Matrix(a.dpi / 72, a.dpi / 72)).save(os.path.join(out, f"page_{i + 1}.jpg"))
print(f"wrote {min(a.pages, i + 1)} pages to {out}")
