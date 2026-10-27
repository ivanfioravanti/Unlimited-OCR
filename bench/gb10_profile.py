"""Instrumented single run of Unlimited-OCR-2 via HF Transformers (CUDA), for the DGX brief tasks B/C/F/D4.

usage: python bench/gb10_profile.py --pages 6 --tag p6 [--stream] [--max-new-tokens 24576]
                                    [--kernels] [--teacher RAW.txt]
Writes bench/out_gb10/<tag>.{json,raw.txt,log.txt}.

Timers (all after torch.cuda.synchronize()):
  preprocess_s  : _uocr_build_inputs (image load->tensors, tokenization)
  sam_s/clip_s  : sum of SAM and CLIP forward over the prefill
  prefill_s     : first call of the decoder model (includes sam+clip+projector+12 layers over the prompt)
  step_fwd_ms   : per decode step, time inside self.model(...) only
  step_wall_ms  : per decode step, pre-hook to next pre-hook (adds lm_head, argmax+sync, page guard, python, streamer)
"""
import argparse, contextlib, io, json, os, re, sys, time
import torch
from transformers import AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--pages", type=int, default=6); ap.add_argument("--tag", default=None)
ap.add_argument("--doc", default=os.path.join(os.path.dirname(__file__), "doc"))
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out_gb10"))
ap.add_argument("--stream", action="store_true", help="HF default (TextStreamer prints every token)")
ap.add_argument("--max-new-tokens", type=int, default=24576)
ap.add_argument("--kernels", action="store_true", help="torch.profiler over 60 decode steps")
ap.add_argument("--kernels-start", type=int, default=3, help="decoder-call index where profiling starts (0=prefill)")
ap.add_argument("--teacher", default=None, help="raw output file whose tokens are fed (teacher forcing, step-wise decode path)")
ap.add_argument("--ngram", type=int, default=0)
ap.add_argument("--crop", action="store_true", help="gundam mode (crop_mode=True); default base")
a = ap.parse_args()
tag = a.tag or f"p{a.pages}"
os.makedirs(a.out, exist_ok=True)
sync = torch.cuda.synchronize

name = "baidu/Unlimited-OCR-2"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModel.from_pretrained(name, trust_remote_code=True, use_safetensors=True,
                                  torch_dtype=torch.bfloat16).eval().cuda()
files = [os.path.join(a.doc, f"page_{i}.jpg") for i in range(1, a.pages + 1)]
mod = sys.modules[type(model).__module__]

T = {"preprocess_s": 0.0, "sam_s": 0.0, "clip_s": 0.0}
calls = []   # (t_pre, t_post) per decoder call, synced

orig_build = mod._uocr_build_inputs
def build(*x, **k):
    sync(); t = time.time(); r = orig_build(*x, **k); sync(); T["preprocess_s"] += time.time() - t; return r
mod._uocr_build_inputs = build

def timed(module, key):
    st = {}
    def pre_(m, i):
        sync(); st["t"] = time.time()
    module.register_forward_pre_hook(pre_)
    def post(m, i, o):
        sync(); T[key] += time.time() - st["t"]
    module.register_forward_hook(post)
timed(model.model.sam_model, "sam_s"); timed(model.model.vision_model, "clip_s")
cur = {}
def pre(m, i, kw=None):
    sync(); cur["t"] = time.time()
def post(m, i, o):
    sync(); calls.append((cur["t"], time.time()))
model.model.register_forward_pre_hook(pre); model.model.register_forward_hook(post)

prof = None
if a.kernels:
    from torch.profiler import profile, ProfilerActivity
    prof_calls = {"n": 0, "p": None}
    def pre2(m, i):
        if prof_calls["n"] == a.kernels_start:
            prof_calls["p"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]); prof_calls["p"].__enter__()
        prof_calls["n"] += 1
    def post2(m, i, o):
        if prof_calls["n"] == a.kernels_start + 60 and prof_calls["p"] is not None:
            sync(); prof_calls["p"].__exit__(None, None, None)
            prof_calls["p"].export_chrome_trace(os.path.join(a.out, f"{tag}.trace.json"))
            ev = prof_calls["p"].key_averages()
            rows = sorted(ev, key=lambda e: -e.device_time_total)[:25]
            with open(os.path.join(a.out, f"{tag}.kernels.txt"), "w") as f:
                f.write("top CUDA kernels over 60 decode steps (device_time_total us, count, name)\n")
                for e in rows:
                    f.write(f"{e.device_time_total:10.0f} {e.count:6d} {e.key[:110]}\n")
                tot = sum(e.self_device_time_total for e in ev)
                tri = sum(e.self_device_time_total for e in ev if "triton" in e.key.lower())
                f.write(f"\nself device time total {tot:.0f} us; triton-named kernels {tri:.0f} us; "
                        f"cuda kernel names containing 'triton': {sorted({e.key[:80] for e in ev if 'triton' in e.key.lower()})[:15]}\n")
            prof_calls["p"] = None
    model.model.register_forward_pre_hook(pre2); model.model.register_forward_hook(post2)

log = io.StringIO()
class Tee(io.TextIOBase):
    def write(self, s): log.write(s); return len(s)
    def flush(self): pass
info = {"tag": tag, "mode": "gundam" if a.crop else "base", "pages": a.pages, "stream": a.stream, "max_new_tokens": a.max_new_tokens,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "dtype": "bfloat16",
        "attn_implementation": getattr(model.config, "_attn_implementation", None)}

if a.teacher:
    # teacher forcing through the SAME decode path: prefill, then feed given tokens one by one
    from transformers.cache_utils import DynamicCache
    text = open(a.teacher).read()
    ids_t = tok.encode(text, add_special_tokens=False)
    info["teacher_roundtrip_ok"] = tok.decode(ids_t, skip_special_tokens=False) == text
    prompt = "<PAGE><image>" * a.pages + "Multi page parsing."
    from PIL import Image
    imgs_pil = [Image.open(f).convert("RGB") for f in files]
    input_ids, imgs, seq_mask, spatial = mod._uocr_build_inputs(tok, prompt, imgs_pil, crop_mode=a.crop, base_size=1024, image_size=640, max_crops=16)
    dev = model.lm_head.weight.device
    ids = input_ids.unsqueeze(0).to(dev); S = ids.shape[1]
    kw = dict(images=[(imgs[0].to(dev), imgs[1].to(dev), spatial)], images_seq_mask=seq_mask.unsqueeze(0).to(dev), images_spatial_crop=spatial)
    cfg = model.config; saved = (getattr(cfg, "_ring_window", None), getattr(cfg, "sliding_window", None))
    cfg._ring_window = None; cfg.sliding_window = None
    st = mod.configure_dash_inference(model.model, True, topm=16)
    rows = []
    with torch.no_grad(), contextlib.redirect_stdout(Tee()):
        cache = DynamicCache()
        out = model.model(input_ids=ids, past_key_values=cache, use_cache=True, position_ids=torch.arange(S, device=dev)[None], return_dict=True, **kw)
        cache = out.past_key_values; h = out.last_hidden_state[:, -1:]
        for i, t in enumerate(ids_t):
            lg = model.lm_head(h).float()[0, -1]
            top2 = torch.topk(lg, 2)
            pred = int(top2.indices[0]); margin = float(top2.values[0] - top2.values[1])
            lt = float(lg[t]); rows.append((i, t, pred, margin, float(top2.values[0]) - lt))
            if i + 1 == len(ids_t): break
            out = model.model(input_ids=torch.tensor([[t]], device=dev), past_key_values=cache, use_cache=True,
                              position_ids=torch.tensor([[S + i]], device=dev), return_dict=True)
            cache = out.past_key_values; h = out.last_hidden_state
    mod.configure_dash_inference(model.model, False, state=st); cfg._ring_window, cfg.sliding_window = saved
    dis = [r for r in rows if r[1] != r[2]]
    import statistics as s
    info.update(teacher_tokens=len(rows), agree=len(rows) - len(dis), agree_pct=round(100 * (len(rows) - len(dis)) / len(rows), 2),
                disagree=len(dis), disagree_margin_lt_0_5=sum(1 for r in dis if r[3] < 0.5), disagree_margin_lt_1=sum(1 for r in dis if r[3] < 1.0),
                disagree_margin_ge_2=sum(1 for r in dis if r[3] >= 2.0), disagree_margin_median=round(s.median([r[3] for r in dis]), 3) if dis else None,
                disagree_margin_max=round(max([r[3] for r in dis]), 3) if dis else None,
                all_disagreements=[{"pos": r[0], "ctx": tok.decode(ids_t[max(0, r[0] - 8):r[0]]), "ctx_page": ids_t[:r[0]].count(128827), "forced": tok.decode([r[1]]), "argmax": tok.decode([r[2]]), "margin": round(r[3], 3), "forced_gap": round(r[4], 3)} for r in dis])
    json.dump(info, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1); print(json.dumps(info, indent=1)); sys.exit()

out_buf, err_buf = Tee(), Tee()
sync(); t0 = time.time()
with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
    res = model.infer(tok, image_file=files, output_path="", save_results=False, eval_mode=not a.stream,
                      crop_mode=a.crop, max_new_tokens=a.max_new_tokens, no_repeat_ngram_size=a.ngram,
                      ngram_window=1024 if a.ngram else 0, return_dict=True)
sync(); wall = time.time() - t0
n = len(res["token_ids"])
fw = [(b - a_) * 1000 for a_, b in calls[1:]]                                   # decode forward only
wl = [(calls[i + 1][0] - calls[i][0]) * 1000 for i in range(1, len(calls) - 1)]  # pre-hook to next pre-hook
def quart(x):
    q = len(x) // 4
    return [round(sum(x[i * q:(i + 1) * q if i < 3 else None]) / len(x[i * q:(i + 1) * q if i < 3 else None]), 2) for i in range(4)] if q else None
dec_wall = calls[-1][1] - calls[1][0] if len(calls) > 1 else 0
info.update(finish=res["finish_reason"], output_tokens=n, num_pages_out=res["num_pages_out"], expected_pages=res["expected_pages"],
            prompt_tokens=len(res["prompt_token_ids"]), total_wall_s=round(wall, 2), infer_internal_time_s=round(res["time"], 2),
            internal_prefill_to_first_token_s=round(res["prefill_time"], 2),
            preprocess_s=round(T["preprocess_s"], 3), sam_s=round(T["sam_s"], 3), clip_s=round(T["clip_s"], 3),
            prefill_forward_s=round(calls[0][1] - calls[0][0], 3), decode_forward_s=round(sum(fw) / 1000, 2), decode_wall_s=round(dec_wall, 2),
            tok_per_s_total=round(n / wall, 2), tok_per_s_decode_wall=round(n / dec_wall, 2) if dec_wall else None,
            ms_per_token_forward_mean=round(sum(fw) / len(fw), 2), ms_per_token_wall_mean=round(sum(wl) / len(wl), 2),
            ms_per_token_forward_quartiles=quart(fw), ms_per_token_wall_quartiles=quart(wl),
            peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
logtxt = out_buf.__class__ and log.getvalue()
fb = [l for l in logtxt.splitlines() if re.search("fallback|falling back|not supported|triton", l, re.I)]
info["fallback_lines"] = len(fb); info["fallback_sample"] = sorted(set(fb))[:10]
info["cpu_params"] = sum(1 for p in model.parameters() if p.device.type != "cuda")
json.dump(info, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
open(os.path.join(a.out, f"{tag}.raw.txt"), "w").write(res["text"])
open(os.path.join(a.out, f"{tag}.log.txt"), "w").write(logtxt[-200000:])
print(json.dumps(info, indent=1))
