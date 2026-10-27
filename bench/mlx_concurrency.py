"""Concurrency sweep for the MLX port: C identical 6-page requests started together, one process each.
usage: python bench/mlx_concurrency.py DOC_DIR MODEL_DIR OUT.json [--concurrency 1 2 4 8 16] [--pages 6] [--cropping]

The Unlimited-OCR-2 port decodes one sequence per model instance (its DASH attention rejects batch size > 1), so concurrency here means C
independent processes sharing one GPU. Same measurement as the GB10 SGLang sweep: aggregate = total output tokens / wall time from the
first request's start to the last request's end, after every worker has loaded the model and warmed up. n-gram rule off (the SGLang
runs had it off); the page-count guard stays on.
"""
import argparse, json, os, subprocess, sys, tempfile, time

ap = argparse.ArgumentParser()
ap.add_argument("doc"); ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
ap.add_argument("--pages", type=int, default=6); ap.add_argument("--cropping", action="store_true")
ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
a = ap.parse_args()


def worker(go_path):
    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor = load(os.path.expanduser(a.model))
    files = [os.path.join(a.doc, f"page_{i}.jpg") for i in range(1, a.pages + 1)]
    prompt = apply_chat_template(processor, model.config, "Multi page parsing.", num_images=a.pages)
    # builds with model-owned generation guards (the old PR head) get the n-gram rule switched off; main has no guards
    kw = dict(temperature=0.0, cropping=a.cropping, **({"no_repeat_ngram_size": 0} if hasattr(model, "make_generation_guards") else {}))
    for _ in stream_generate(model, processor, prompt, image=files, max_tokens=24, **kw):  # warm up
        pass
    print("READY", flush=True)
    while not os.path.exists(go_path):
        time.sleep(0.02)
    start_at = float(open(go_path).read())
    while time.time() < start_at:
        time.sleep(0.001)
    t0, first, n, fin = time.time(), None, 0, None
    for g in stream_generate(model, processor, prompt, image=files, max_tokens=4096 * a.pages, **kw):
        if first is None:
            first = time.time()
        n, fin = g.generation_tokens, g.finish_reason
    print("RESULT " + json.dumps({"start": t0, "first": first, "end": time.time(), "tokens": n, "finish": fin}), flush=True)


if a.worker:
    worker(a.worker); sys.exit(0)

rows = []
for c in a.concurrency:
    go = os.path.join(tempfile.mkdtemp(), "go")
    cmd = [sys.executable, os.path.abspath(__file__), a.doc, a.model, a.out, "--pages", str(a.pages), "--worker", go] + (["--cropping"] if a.cropping else [])
    errs = [tempfile.NamedTemporaryFile("w+", suffix=".err", delete=False) for _ in range(c)]
    procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=e, text=True) for e in errs]
    for p in procs:  # wait until every worker is loaded and warm
        while True:
            line = p.stdout.readline()
            if line.startswith("READY") or not line:
                break
    open(go, "w").write(str(time.time() + 1.5))
    res, failures = [], []
    for i, p in enumerate(procs):
        got = False
        for line in p.stdout:
            if line.startswith("RESULT "):
                res.append(json.loads(line[7:])); got = True
        p.wait()
        if not got:
            errs[i].seek(0)
            tail = [l.strip() for l in errs[i].read().splitlines() if l.strip()][-3:]
            failures.append({"worker": i, "exit": p.returncode, "stderr_tail": tail})
    ok = [r for r in res if r["tokens"] > 0]
    wall = max(r["end"] for r in ok) - min(r["start"] for r in ok)
    lat = [r["end"] - r["start"] for r in ok]
    row = {"concurrency": c, "ok": len(ok), "wall_s": round(wall, 2), "total_tokens": sum(r["tokens"] for r in ok),
           "agg_tok_per_s": round(sum(r["tokens"] for r in ok) / wall, 1),
           "latency_s": {"min": round(min(lat), 2), "mean": round(sum(lat) / len(lat), 2), "max": round(max(lat), 2)},
           "ttft_s_mean": round(sum(r["first"] - r["start"] for r in ok) / len(ok), 2),
           "per_request_tok_per_s_mean": round(sum(r["tokens"] / (r["end"] - r["start"]) for r in ok) / len(ok), 1),
           "finish": sorted({r["finish"] for r in ok}), "requested": c, "failed": failures}
    rows.append(row)
    print(json.dumps(row), flush=True)
    json.dump({"pages": a.pages, "mode": "gundam" if a.cropping else "base", "ngram": False, "sweep": rows}, open(a.out, "w"), indent=1)
