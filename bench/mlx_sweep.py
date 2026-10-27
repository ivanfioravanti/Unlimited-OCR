"""Page-count x mode sweep for the MLX port: one request per cell, raw output + timing, for check_quality.py.
usage: python bench/mlx_sweep.py DOC_DIR MODEL_DIR OUT_DIR [--pages 3 4 5 6]

Modes: base (one 1024 view per page) and gundam (cropping=True, every page tiled). Writes OUT_DIR/mlx_<mode>_p<N>_raw.txt and
OUT_DIR/sweep.json (prompt/output tokens, finish reason, time to first token, total time, decode tokens/s, and the mean decode
ms/token over each quarter of the output, which shows whether per-token cost stays flat as the output grows).
"""
import argparse, json, os, platform, statistics, subprocess, time
import mlx.core as mx
import mlx_vlm
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template

ap = argparse.ArgumentParser()
ap.add_argument("doc"); ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--pages", type=int, nargs="+", default=[3, 4, 5, 6])
ap.add_argument("--max-tokens-per-page", type=int, default=4096, help="output cap per page (a runaway stops here)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
model, processor = load(os.path.expanduser(a.model))
files = lambda n: [os.path.join(a.doc, f"page_{i}.jpg") for i in range(1, n + 1)]
prompt = lambda n: apply_chat_template(processor, model.config, "Multi page parsing.", num_images=n)

for crop in (False, True):  # warm both input paths so kernel compilation is not measured
    for _ in stream_generate(model, processor, prompt(2), image=files(2), max_tokens=24, temperature=0.0, cropping=crop):
        pass

chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
rows = []
for mode in ("base", "gundam"):
    for n in a.pages:
        stamps, chunks, t = [], [], time.perf_counter()
        for g in stream_generate(model, processor, prompt(n), image=files(n), max_tokens=a.max_tokens_per_page * n, temperature=0.0,
                                 cropping=(mode == "gundam")):  # explicit in both modes: main tiles by default
            stamps.append(time.perf_counter()); chunks.append(g.text)
        total = stamps[-1] - t
        gaps = [y - x for x, y in zip(stamps, stamps[1:])]
        q = len(gaps) // 4
        quart = [round(1000 * statistics.fmean(gaps[i * q:(i + 1) * q if i < 3 else None]), 2) for i in range(4)]
        row = {"chip": chip, "mode": mode, "pages": n, "prompt_tokens": g.prompt_tokens, "output_tokens": g.generation_tokens,
               "finish": g.finish_reason, "ttft_s": round(stamps[0] - t, 2), "total_s": round(total, 2),
               "tok_per_s": round(g.generation_tokens / total, 1),
               "decode_tps": round((g.generation_tokens - 1) / (stamps[-1] - stamps[0]), 1), "quartile_ms_per_token": quart}
        rows.append(row)
        open(os.path.join(a.out, f"mlx_{mode}_p{n}_raw.txt"), "w").write("".join(chunks))
        print(json.dumps(row), flush=True)

json.dump({"mlx": mx.__version__, "mlx_vlm": mlx_vlm.__version__, "platform": platform.platform(), "rows": rows},
          open(os.path.join(a.out, "sweep.json"), "w"), indent=1)
