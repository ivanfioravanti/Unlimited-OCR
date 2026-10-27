"""Concurrency sweep against the SGLang server: C simultaneous identical multi-page requests.
usage: python bench/bench_sglang_conc.py bench/doc --pages 6 --crop --conc 1 4 8 16 [--no-ngram] [--out DIR]
Aggregate tok/s = sum(output tokens) / (wall time from first send to last finish). Per-request latency = send -> last token."""
import argparse, base64, glob, json, os, re, threading, time
import requests
from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor

ap = argparse.ArgumentParser()
ap.add_argument("doc"); ap.add_argument("--pages", type=int, default=6); ap.add_argument("--crop", action="store_true")
ap.add_argument("--conc", type=int, nargs="+", default=[1, 4, 8, 16]); ap.add_argument("--no-ngram", action="store_true")
ap.add_argument("--url", default="http://127.0.0.1:10000"); ap.add_argument("--out", default="bench/out_gb10/conc")
ap.add_argument("--max-tokens", type=int, default=24576)
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.doc, "page_*.*")), key=lambda f: int(re.search(r"page_(\d+)", f).group(1)))[:a.pages]
imgs = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(f, "rb").read()).decode()}} for f in files]
payload = {"model": "Unlimited-OCR-2", "messages": [{"role": "user", "content": imgs + [{"type": "text", "text": "<PAGE><image>" * len(files) + "Multi page parsing."}]}],
    "max_tokens": a.max_tokens, "temperature": 0, "skip_special_tokens": False, "images_config": {"crop_mode": a.crop}, "stream": True,
    "stream_options": {"include_usage": True}}
cp = {"page_token_id": 128827, "eos_token_id": 1, "max_page_tokens": 20000}
if not a.no_ngram:
    payload["custom_logit_processor"] = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
    cp.update(ngram_size=35, window_size=1024 if len(files) > 1 else 128)
payload["custom_params"] = cp
os.makedirs(a.out, exist_ok=True)

def one(i, out):
    s = requests.Session(); s.trust_env = False
    t = time.time(); ttft = None; chunks = []; usage = {}; fin = None
    try:
        r = s.post(a.url + "/v1/chat/completions", json=payload, stream=True, timeout=6000); r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            ev = json.loads(d); usage = ev.get("usage") or usage
            for c in ev.get("choices") or []:
                fin = c.get("finish_reason") or fin
                x = (c.get("delta") or {}).get("content") or ""
                if x:
                    if ttft is None: ttft = time.time() - t
                    chunks.append(x)
        out[i] = {"latency_s": time.time() - t, "ttft_s": ttft, "tokens": usage.get("completion_tokens") or len(chunks), "finish": fin, "text": "".join(chunks)}
    except Exception as e:
        out[i] = {"error": repr(e)}

res = {"mode": "gundam" if a.crop else "base", "pages": len(files), "ngram": not a.no_ngram, "sweep": []}
for c in a.conc:
    out = [None] * c; th = [threading.Thread(target=one, args=(i, out)) for i in range(c)]
    t0 = time.time(); [x.start() for x in th]; [x.join() for x in th]; wall = time.time() - t0
    ok = [o for o in out if o and "error" not in o]
    texts = {o["text"] for o in ok}
    row = {"concurrency": c, "wall_s": round(wall, 2), "ok": len(ok), "errors": [o["error"] for o in out if o and "error" in o][:3],
           "total_tokens": sum(o["tokens"] for o in ok), "agg_tok_per_s": round(sum(o["tokens"] for o in ok) / wall, 1),
           "latency_s": {"min": round(min(o["latency_s"] for o in ok), 2), "mean": round(sum(o["latency_s"] for o in ok) / len(ok), 2), "max": round(max(o["latency_s"] for o in ok), 2)} if ok else None,
           "ttft_s_mean": round(sum(o["ttft_s"] for o in ok) / len(ok), 2) if ok else None,
           "per_request_tok_per_s_mean": round(sum(o["tokens"] / o["latency_s"] for o in ok) / len(ok), 1) if ok else None,
           "distinct_outputs": len(texts), "tokens_each": sorted({o["tokens"] for o in ok})}
    res["sweep"].append(row); print(json.dumps(row), flush=True)
    if ok: open(os.path.join(a.out, f"conc{c}_sample.txt"), "w").write(ok[0]["text"])
json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
