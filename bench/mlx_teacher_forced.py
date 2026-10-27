"""Teacher-forced fidelity of the MLX model against a reference raw output (e.g. native HF / SGLang), per token position.
usage: python bench/mlx_teacher_forced.py DOC_DIR MODEL_DIR REF_RAW.txt [--pages 6] [--cropping]

The reference tokens are fed through the real MLX prefill + DASH decode path one step at a time, and at every step the model's own argmax is
recorded. Positions inside a bbox "[x0, y0, x1, y1]" are reported separately from text/structure positions: bbox digits are near-ties by
nature (+-1-2 px), text positions are what matters for transcription. The generation guards are switched off so they cannot touch the forcing
(page_count_guard / no_repeat_ngram_size exist in the mlx-vlm uocr2-generation-guards branch; drop those two kwargs on a build without it).
"""
import argparse, os, re, time
import mlx.core as mx
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("doc"); ap.add_argument("model"); ap.add_argument("ref")
ap.add_argument("--pages", type=int, default=6); ap.add_argument("--cropping", action="store_true")
a = ap.parse_args()

model_dir = os.path.expanduser(a.model)
tk = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
text = open(a.ref, encoding="utf-8").read()
enc = tk.encode(text, add_special_tokens=False)
ref = enc.ids
print(f"reference tokens={len(ref)}  decode(encode(text)) == text: {tk.decode(ref, skip_special_tokens=False) == text}", flush=True)
spans = [m.span() for m in re.finditer(r"\[\d+, \d+, \d+, \d+\]", text)]
in_bbox = [any(s <= lo and hi <= e for s, e in spans) for lo, hi in enc.offsets]

model, processor = load(model_dir)
prompt = apply_chat_template(processor, model.config, "Multi page parsing.", num_images=a.pages)
files = [os.path.join(a.doc, f"page_{i}.jpg") for i in range(1, a.pages + 1)]
rec, step = [], {"i": 0}


def force(tokens, logits):
    i = step["i"]
    if i >= len(ref):
        return logits
    lg = logits.reshape(-1).astype(mx.float32)
    top = mx.sort(mx.topk(lg, 2))
    rec.append((i, ref[i], int(mx.argmax(lg).item()), float(top[1].item()) - float(lg[ref[i]].item())))
    step["i"] += 1
    return mx.where(mx.arange(lg.shape[0]) == ref[i], 0.0, -1e9).reshape(logits.shape).astype(logits.dtype)


t0 = time.perf_counter()
for _ in stream_generate(model, processor, prompt, image=files, max_tokens=len(ref), temperature=0.0,
                         page_count_guard=False, no_repeat_ngram_size=0, logits_processors=[force],
                         **({"cropping": True} if a.cropping else {})):
    pass
bad = [r for r in rec if r[1] != r[2]]
bb = [r for r in bad if in_bbox[r[0]]]
tx = [r for r in bad if not in_bbox[r[0]]]
n_bb = sum(in_bbox[: len(rec)]); n_tx = len(rec) - n_bb
print(f"forced {len(rec)} steps in {time.perf_counter() - t0:.0f}s")
print(f"agreement overall      {1 - len(bad) / len(rec):.2%}  ({len(bad)} disagreements)")
print(f"agreement bbox tokens  {1 - len(bb) / n_bb:.2%}  ({len(bb)} of {n_bb})")
print(f"agreement other tokens {1 - len(tx) / n_tx:.2%}  ({len(tx)} of {n_tx})   <- text and structure")
dec = lambda t: tk.decode([t], skip_special_tokens=False).replace("\n", "\\n")
print("text/structure disagreements (position, reference token, MLX top choice, reference token's gap below MLX top, context):")
for i, r, m, gap in tx:
    ctx = "".join(dec(t) for t in ref[max(0, i - 6):i])
    print(f"  {i:5d}  ref={dec(r)!r:14} mlx={dec(m)!r:14} gap={gap:5.2f}  ...{ctx!r}")
print(f"largest gap among text disagreements: {max([g for *_, g in tx], default=0):.2f} logits; among bbox: {max([g for *_, g in bb], default=0):.2f}")
