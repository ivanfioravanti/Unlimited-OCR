"""Force the first K tokens of a reference output, then let MLX decode freely: does an early bbox-digit tie decide the branch?
usage: python bench/mlx_prefix_force.py DOC_DIR MODEL_DIR REF_RAW.txt K OUT.txt [--pages 6] [--cropping]

The page-1 header and the page-2 table of contents flip between "empty header, 3 blocks" and "Baidu header, 35 blocks" depending on exact
ties at output positions ~7 and ~13 (bbox digits). Forcing the first ~20 reference tokens shows whether the free run then follows the
reference's branch (compare the result with bench/compare_raw.py).
"""
import argparse, os
import mlx.core as mx
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("doc"); ap.add_argument("model"); ap.add_argument("ref"); ap.add_argument("k", type=int); ap.add_argument("out")
ap.add_argument("--pages", type=int, default=6); ap.add_argument("--cropping", action="store_true")
a = ap.parse_args()

model_dir = os.path.expanduser(a.model)
tk = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
ref = tk.encode(open(a.ref, encoding="utf-8").read(), add_special_tokens=False).ids
model, processor = load(model_dir)
prompt = apply_chat_template(processor, model.config, "Multi page parsing.", num_images=a.pages)
files = [os.path.join(a.doc, f"page_{i}.jpg") for i in range(1, a.pages + 1)]
step = {"i": 0}


def force(tokens, logits):
    i = step["i"]
    step["i"] += 1
    if i >= a.k:
        return logits
    return mx.where(mx.arange(logits.shape[-1]) == ref[i], 0.0, -1e9).astype(logits.dtype).reshape(logits.shape)


out = []
for g in stream_generate(model, processor, prompt, image=files, max_tokens=4096 * a.pages, temperature=0.0,
                         logits_processors=[force], **({"cropping": True} if a.cropping else {})):
    out.append(g.text)
open(a.out, "w").write("".join(out))
print(f"K={a.k} tokens={g.generation_tokens} finish={g.finish_reason} <PAGE>={''.join(out).count('<PAGE>')}")
