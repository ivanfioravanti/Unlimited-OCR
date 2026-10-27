"""mlx-vlm benchmark for Unlimited-OCR-2 (Apple Silicon), same protocol and result schema as bench_transformers.py.
usage: python bench/bench_mlx.py DOC_DIR [--model PATH_OR_HF_ID] [--pages N] [--runs R] [--out DIR]

Needs an mlx-vlm build with Unlimited-OCR-2 support (Blaizzy/mlx-vlm PR #2601, not yet in a release).

Timing scope (matches model.infer_multi in bench_transformers.py): `seconds` is the wall time of the whole request,
image decode + preprocessing + vision encoder + prefill + decode; model load is excluded. `tok_per_s` is
output_tokens / seconds, so it includes prefill. `ttft_s` is the time to the first generated token (everything
before decode, image preprocessing included) and `decode_tps` is the decode-only rate after that.
"""
import argparse, glob, json, os, platform, re, subprocess, time
import mlx.core as mx
import mlx_vlm
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template

ap = argparse.ArgumentParser()
ap.add_argument("doc"); ap.add_argument("--model", default="baidu/Unlimited-OCR-2")
ap.add_argument("--pages", type=int, default=0)
ap.add_argument("--runs", type=int, default=2); ap.add_argument("--out", default="bench_out_mlx")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.doc, "page_*.*")), key=lambda f: int(re.search(r"page_(\d+)", f).group(1)))
if a.pages: files = files[:a.pages]
os.makedirs(a.out, exist_ok=True)

t0 = time.perf_counter()
model, processor = load(os.path.expanduser(a.model))  # weights stay in the checkpoint dtype (bf16)
mx.eval(model.parameters()); load_s = time.perf_counter() - t0

# Task text only: the processor expands it to "<PAGE><image>" * n + "Multi page parsing." (base mode for >1 page).
prompt = apply_chat_template(processor, model.config, "Multi page parsing.", num_images=len(files))
sysctl = lambda k: subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
res = {"device": "mps/metal (mlx)", "gpu": sysctl("machdep.cpu.brand_string"), "mlx": mx.__version__,
       "mlx_vlm": mlx_vlm.__version__, "dtype": "bfloat16", "pages": len(files), "load_s": round(load_s, 1),
       "memory_gb": int(sysctl("hw.memsize")) // 2**30, "platform": platform.platform(), "runs": []}

for r in range(a.runs):
    mx.reset_peak_memory()
    stamps, chunks, last = [], [], None
    t = time.perf_counter()
    for g in stream_generate(model, processor, prompt, image=files, max_tokens=4096 * len(files), temperature=0.0):
        stamps.append(time.perf_counter()); chunks.append(g.text); last = g
    dt = time.perf_counter() - t
    n = last.generation_tokens
    run = {"run": r, "seconds": round(dt, 2), "output_tokens": n, "tok_per_s": round(n / dt, 2),
           "peak_mem_gb": round(mx.get_peak_memory() / 2**30, 2),
           "prompt_tokens": last.prompt_tokens, "prefill_s": round(stamps[0] - t, 3),  # incl. image preprocessing
           "decode_tps": round((n - 1) / (stamps[-1] - stamps[0]), 2), "finish_reason": last.finish_reason}
    res["runs"].append(run); print(run, flush=True)
    open(os.path.join(a.out, f"raw_run{r}.txt"), "w").write("".join(chunks))
json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
