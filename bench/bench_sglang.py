"""SGLang client benchmark (server per README). usage: python bench/bench_sglang.py bench/doc [--runs 2] [--crop] [--out bench/out_sglang]
Run in the SGLang venv (needs sglang for the n-gram logit processor)."""
import argparse, base64, glob, json, os, re, time
import requests
from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor

ap = argparse.ArgumentParser()
ap.add_argument("doc"); ap.add_argument("--runs", type=int, default=2)
ap.add_argument("--crop", action="store_true", help="gundam mode (default: base)")
ap.add_argument("--url", default="http://127.0.0.1:10000"); ap.add_argument("--out", default="bench/out_sglang")
ap.add_argument("--tokens-per-page", type=int, default=8000)
ap.add_argument("--pages", type=int, default=0); ap.add_argument("--no-ngram", action="store_true", help="omit the no-repeat n-gram logit processor (HF/MLX do not use it)")
ap.add_argument("--max-tokens", type=int, default=0)
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.doc, "page_*.*")), key=lambda f: int(re.search(r"page_(\d+)", f).group(1)))
if a.pages: files = files[:a.pages]
imgs = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(f, "rb").read()).decode()}} for f in files]
s = requests.Session(); s.trust_env = False
payload = {"model": "Unlimited-OCR-2",
    "messages": [{"role": "user", "content": imgs + [{"type": "text", "text": "<PAGE><image>" * len(files) + "Multi page parsing."}]}],
    "max_tokens": a.tokens_per_page * len(files), "temperature": 0, "skip_special_tokens": False,
    "images_config": {"crop_mode": a.crop}, "stream": True, "stream_options": {"include_usage": True},
    "custom_logit_processor": DeepseekOCRNoRepeatNGramLogitProcessor.to_str(),
    "custom_params": {"ngram_size": 35, "window_size": 1024 if len(files) > 1 else 128,
                      "page_token_id": 128827, "eos_token_id": 1, "max_page_tokens": 20000}}
if a.no_ngram:
    payload.pop("custom_logit_processor"); payload.pop("custom_params")
    payload["custom_params"] = {"page_token_id": 128827, "eos_token_id": 1, "max_page_tokens": 20000}  # page guard only
if a.max_tokens: payload["max_tokens"] = a.max_tokens
os.makedirs(a.out, exist_ok=True)
res = {"device": "cuda", "gpu": "NVIDIA GB10", "backend": "sglang 0.5.20.dev10", "mode": "gundam" if a.crop else "base",
       "dtype": "bfloat16", "pages": len(files), "ngram_processor": not a.no_ngram, "runs": []}
for r in range(a.runs):
    t = time.time(); ttft = None; chunks = []; usage = {}; finish = None
    resp = s.post(a.url + "/v1/chat/completions", json=payload, stream=True, timeout=6000); resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"): continue
        d = line[5:].strip()
        if d == "[DONE]": break
        ev = json.loads(d); usage = ev.get("usage") or usage
        for c in ev.get("choices") or []:
            finish = c.get("finish_reason") or finish
            x = (c.get("delta") or {}).get("content") or ""
            if x:
                if ttft is None: ttft = time.time() - t
                chunks.append(x)
    dt = time.time() - t; n = usage.get("completion_tokens") or len(chunks)
    run = {"run": r, "seconds": round(dt, 2), "prefill_s": round(ttft, 2) if ttft else None, "prompt_tokens": usage.get("prompt_tokens"),
           "output_tokens": n, "finish": finish, "tok_per_s": round(n / dt, 2), "decode_tok_per_s": round(n / (dt - ttft), 2) if ttft else None}
    res["runs"].append(run); print(run, flush=True)
    open(os.path.join(a.out, f"raw_run{r}.txt"), "w").write("".join(chunks))
json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=2)
