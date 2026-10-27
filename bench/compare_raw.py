"""Compare two raw Unlimited-OCR-2 outputs (e.g. CUDA reference vs MLX): first divergence, per-page blocks, text/bbox drift.
usage: python bench/compare_raw.py REF.txt OTHER.txt [--tokenizer MODEL_DIR]   (--tokenizer adds token-level numbers; needs `tokenizers`)
"""
import argparse, difflib, re, statistics

BLOCK = re.compile(r"(\w+)(?:<LEVEL\d>)?\[(\d+), (\d+), (\d+), (\d+)\]<FORK>(.*)", re.S)

ap = argparse.ArgumentParser()
ap.add_argument("ref"); ap.add_argument("other"); ap.add_argument("--tokenizer")
a = ap.parse_args()
ref, oth = (open(f, encoding="utf-8").read() for f in (a.ref, a.other))

i = next((k for k, (x, y) in enumerate(zip(ref, oth)) if x != y), min(len(ref), len(oth)))
print(f"chars {len(ref)} vs {len(oth)}; first differing char {i} (page {ref[:i].count('<PAGE>')}, block {ref[:i].count('<BLOCK>')})")
print(f"  ref  ...{ref[max(0, i - 40):i + 60]!r}\n  other...{oth[max(0, i - 40):i + 60]!r}")

if a.tokenizer:
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(a.tokenizer.rstrip("/") + "/tokenizer.json")
    r, o = (tk.encode(t, add_special_tokens=False).ids for t in (ref, oth))
    lcp = next((k for k, (x, y) in enumerate(zip(r, o)) if x != y), min(len(r), len(o)))
    matched = sum(b.size for b in difflib.SequenceMatcher(None, r, o, autojunk=False).get_matching_blocks())
    print(f"tokens {len(r)} vs {len(o)}; identical prefix {lcp}; matched anywhere {matched} "
          f"({matched / len(r):.1%} of ref, {matched / len(o):.1%} of other)")

def parse(raw):
    pages = []
    for p in [p for p in raw.split("<PAGE>") if p]:
        pages.append([m.groups() for b in p.split("<BLOCK>")[1:] if (m := BLOCK.match(b))])
    return pages

rp, op = parse(ref), parse(oth)
print(f"\n{'page':>4} {'blocks ref/other':>17} {'text similarity':>16} {'identical blocks':>17} {'bbox |d| median/max px':>23}")
for n, (pr, po) in enumerate(zip(rp, op), 1):
    ta, tb = ("\n".join(b[5].strip() for b in p) for p in (pr, po))
    sim = difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()
    pool = {}
    for b in po: pool.setdefault((b[0], b[5].strip()), []).append(b)
    same, deltas = 0, []
    for b in pr:
        c = pool.get((b[0], b[5].strip()))
        if c:
            o = c.pop(0); same += 1
            deltas += [abs(int(x) - int(y)) for x, y in zip(b[1:5], o[1:5])]
    d = f"{statistics.median(deltas):.0f}/{max(deltas)}" if deltas else "-"
    print(f"{n:>4} {len(pr):>8}/{len(po):<8} {sim:>16.3f} {same:>9}/{len(pr):<7} {d:>23}")
