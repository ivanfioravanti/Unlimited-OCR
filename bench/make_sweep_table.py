"""Build bench/reference/gb10_page_sweep.md from bench/out_gb10/p{N}_nostream.{json,raw.txt}."""
import json, os, re, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__)); out = os.path.join(here, "out_gb10")
rows = []
for n in (3, 4, 5, 6):
    j = os.path.join(out, f"p{n}_nostream.json")
    if not os.path.exists(j): continue
    d = json.load(open(j)); raw = open(os.path.join(out, f"p{n}_nostream.raw.txt")).read()
    sc = subprocess.run([sys.executable, os.path.join(here, "score_vs_pdf.py"), "--pdf", os.path.join(here, "report.pdf"), os.path.join(out, f"p{n}_nostream.raw.txt")],
                        capture_output=True, text=True).stdout
    scores = re.search(r"vs PDF text:\s*(.*)", sc); scores = scores.group(1).strip() if scores else "?"
    pages = raw.split("<PAGE>")[1:]
    p3 = pages[2] if len(pages) > 2 else ""
    m = re.search(r"<FORK>(Document parsing[^\n]{0,80}|[^\n<]*paradigm shift)", p3) or re.search(r"<BLOCK>text\[[^\]]*\]<FORK>([^\n]{0,70})", p3)
    rows.append((n, d["output_tokens"], d["finish"], d["num_pages_out"], scores, (m.group(1) if m else "")[:70].replace("|", "\\|"), d["tok_per_s_total"]))
with open(os.path.join(here, "reference", "gb10_page_sweep.md"), "w") as f:
    f.write("| pages | output tokens | ended on | `<PAGE>` out (must = N) | per-page similarity vs PDF | page-3 first sentence (verbatim, truncated) | tok/s |\n|---|---|---|---|---|---|---|\n")
    for n, t, fin, pg, sc, s3, ts in rows:
        f.write(f"| {n} | {t} | {'EOS' if fin == 'stop' else 'CAP (' + fin + ')'} | {pg} | {sc} | {s3} | {ts} |\n")
print(open(os.path.join(here, "reference", "gb10_page_sweep.md")).read())
