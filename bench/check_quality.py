"""PASS/FAIL quality checks for a raw Unlimited-OCR-2 output of the first N pages of Unlimited-OCR-2.pdf.
usage: python bench/check_quality.py RAW.txt [RAW2.txt ...] [--pages N] [--pdf Unlimited-OCR-2.pdf]   (needs pymupdf)
Exit code 1 if any file has a FAIL. WARN does not fail.

Checks (the PDF's own text layer is the ground truth):
  pages      the output has exactly N <PAGE> markers                      (a runaway/early stop fails here)
  no-loop    no sentence repeated 3+ times in a row                       ("The document is a key to the development of the document.")
  prose      pages 2 and 3 have similarity >= 0.99 to the PDF text        (LaTeX markup ignored)
  canary     page 3 starts "Document parsing has undergone a major paradigm shift"  (N >= 3; base mode misreads it from 4 pages)
  words      WARN for every prose word absent from the PDF page, except the few that every engine produces on these 6 pages
"""
import argparse, difflib, os, re
import fitz

ap = argparse.ArgumentParser()
ap.add_argument("raw", nargs="+"); ap.add_argument("--pages", type=int, default=6)
ap.add_argument("--pdf", default=os.path.join(os.path.dirname(__file__), "..", "Unlimited-OCR-2.pdf"))
a = ap.parse_args()

CANARY = "document parsing has undergone a major paradigm shift"
BENIGN = {"hundred", "cross", "experts", "snarcv", "ubbw"}  # PDF line-break artefacts / a tiny italic line misread by every engine
SKIP = {"equation", "image", "table", "figure", "image_caption", "table_caption", "formula", "header", "footer", "page_number", "other"}
norm = lambda s: re.sub(r"[^0-9a-z]", "", re.sub(r"\\[a-zA-Z]+", "", s).lower())
words = lambda s: re.findall(r"[a-z]{4,}", s.lower())
tags = re.compile(r"<BLOCK>\w+(?:<LEVEL\d>)?\[[^\]]*\]<FORK>|<image>|<PAGE>")
block = re.compile(r"<BLOCK>(\w+)(?:<LEVEL\d>)?\[[^\]]*\]<FORK>")
pdf = fitz.open(a.pdf)


def prose(page_raw):
    parts = block.split(page_raw)  # ['', label, text, label, text, ...]
    txt = " ".join(parts[i + 1] for i in range(1, len(parts) - 1, 2) if parts[i] not in SKIP)
    return re.sub(r"\$[^$]*\$", " ", txt)


def looped(text):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", tags.sub(" ", text)) if len(s.strip()) >= 25]
    run = 1
    for prev, cur in zip(sents, sents[1:]):
        run = run + 1 if cur == prev else 1
        if run >= 3:
            return cur[:70]
    return None


failed = False
for path in a.raw:
    raw = open(path, encoding="utf-8").read()
    pages = [p for p in raw.split("<PAGE>") if p]
    res = []  # (name, status, detail)

    n_markers = raw.count("<PAGE>")
    res.append(("pages", "PASS" if n_markers == a.pages else "FAIL", f"{n_markers} <PAGE> markers, expected {a.pages}"))

    loop = looped(raw)
    res.append(("no-loop", "FAIL" if loop else "PASS", f"repeated: {loop!r}" if loop else "no repeated sentence"))

    sims = {}
    for n in (1, 2):  # pages 2 and 3 (0-based 1, 2)
        if n < len(pages):
            sims[n + 1] = difflib.SequenceMatcher(None, norm(pdf[n].get_text()), norm(tags.sub("", pages[n])), autojunk=False).ratio()
    ok = len(sims) == min(2, max(0, a.pages - 1)) and all(v >= 0.99 for v in sims.values())
    res.append(("prose", "PASS" if ok else "FAIL", " ".join(f"p{k}={v:.3f}" for k, v in sims.items()) or "pages missing"))

    if a.pages >= 3:
        if len(pages) >= 3:
            p3 = prose(pages[2])
            i = p3.lower().find("document parsing")
            sentence = p3[i:i + 80].replace("\n", " ") if i >= 0 else "(sentence not found)"
            res.append(("canary", "PASS" if CANARY in p3.lower() else "FAIL", sentence))
        else:
            res.append(("canary", "FAIL", "page 3 missing"))

    extra = []
    for n, p in enumerate(pages[: a.pages]):
        truth = set(words(pdf[n].get_text().replace("-\n", "")))
        extra += [f"p{n + 1}:{w}" for w in dict.fromkeys(words(prose(p))) if w not in truth and w not in BENIGN]
    res.append(("words", "WARN" if extra else "PASS", " ".join(extra[:12]) or "no unexpected words"))

    print(f"== {os.path.basename(path)}")
    for name, status, detail in res:
        print(f"  {status:4}  {name:8} {detail}")
    failed |= any(s == "FAIL" for _, s, _ in res)
raise SystemExit(1 if failed else 0)
