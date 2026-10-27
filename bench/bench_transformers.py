"""Transformers benchmark for Unlimited-OCR-2. Device-agnostic (cuda / mps / cpu).
usage: python bench.py DOC_DIR [--pages N] [--runs R] [--out DIR]
"""
import argparse, glob, json, os, platform, re, time
import torch
from transformers import AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("doc"); ap.add_argument("--pages", type=int, default=0)
ap.add_argument("--runs", type=int, default=2); ap.add_argument("--out", default="bench_out")
ap.add_argument("--dtype", default="bfloat16")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
sync = {"cuda": torch.cuda.synchronize, "mps": getattr(getattr(torch, "mps", None), "synchronize", lambda: None)}.get(dev, lambda: None)
files = sorted(glob.glob(os.path.join(a.doc, "page_*.*")), key=lambda f: int(re.search(r"page_(\d+)", f).group(1)))
if a.pages: files = files[:a.pages]

name = "baidu/Unlimited-OCR-2"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModel.from_pretrained(name, trust_remote_code=True, use_safetensors=True,
                                  torch_dtype=getattr(torch, a.dtype)).eval().to(dev)
sync(); load_s = time.time() - t0

res = {"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor() or platform.machine(),
       "torch": torch.__version__, "dtype": a.dtype, "pages": len(files), "load_s": round(load_s, 1), "runs": []}
for r in range(a.runs):
    if dev == "cuda": torch.cuda.reset_peak_memory_stats()
    sync(); t = time.time()
    text, n_tok = model.infer_multi(tok, image_files=files, output_path=os.path.join(a.out, f"run{r}"), save_results=True)
    sync(); dt = time.time() - t
    run = {"run": r, "seconds": round(dt, 2), "output_tokens": n_tok, "tok_per_s": round(n_tok / dt, 2) if n_tok else None,
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else None}
    if dev == "mps": run["mps_mem_gb"] = round(torch.mps.current_allocated_memory() / 2**30, 2)
    res["runs"].append(run); print(run, flush=True)
    open(os.path.join(a.out, f"raw_run{r}.txt"), "w").write(text if isinstance(text, str) else str(text))
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
