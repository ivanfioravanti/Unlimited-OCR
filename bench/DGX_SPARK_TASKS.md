# DGX Spark (GB10): is it slow because of the setup, and is the output correct?

Context: same 6-page request (`bench/doc/page_1..6.jpg`, base mode, greedy, bf16) on three setups.

| | GB10, HF Transformers (your run) | M3 Ultra, mlx-vlm PR #2601 |
|---|---|---|
| Whole request | 108.5 s, 3934 tok, **36.3 tok/s** | 26.5 s, 4566 tok, **172 tok/s** |
| Decode only | not measured | 177.7 tok/s = 5.6 ms/token |
| Prefill (1649 prompt tokens, incl. image preprocessing) | not measured | 0.74-0.86 s |

On MLX the per-token time stayed flat (5.5 -> 5.65 ms/token) over 6,400 generated tokens, so the sparse-attention cost really is
constant. The GB10 run is ~27.5 ms/token. Unverified back-of-envelope from `config.json` (hidden 1280, 12 layers, 6+2 active experts of
896, dense layer 0 of 6848, vocab 129280): one decode step reads ~1.15 GB of weights. At the GB10's ~273 GB/s that is a ~4 ms/token
floor (~240 tok/s), so 27.5 ms/token suggests the HF run is not memory-bound. Treat this as a hypothesis to test, not a result.

Everything below goes in `bench/reference/` as `gb10_<what>_result.json` / `.md`, same schema as the existing result files.
Branch with my outputs and tools: `benchmark-m3ultra-mlx` on ivanfioravanti/Unlimited-OCR.

## A. Same input? (5 min)
`sha256sum bench/doc/page_*.jpg | cut -c1-12` must give `f74b84fc2c9c 0a3fbadd3951 4c49db49331d d57e2a04f6f1 3f883038f9d7 b678837bcbf7`
(pages 1-6). If not, say so and stop: nothing else is comparable.

## B. Where does the time go?
Split one `infer_multi` run into: image preprocessing, vision encoder, prefill, decode. Then decode ms/token **per quartile of the output**
(per-step timestamps with `torch.cuda.synchronize()`; there is a hand-written decode loop around lines 1135-1195 of
`modeling_unlimitedocr.py` that I only read through grep, so check which loop `infer_multi` really uses).
* Flat across quartiles -> constant-cost attention is engaged; the cost is elsewhere.
* Growing with output length -> sparse attention is NOT engaged and decode is attending to the full cache.
Also run `nvidia-smi dmon -s u` during decode: GPU busy or mostly idle (CPU/launch-bound on the Grace cores)?

## C. Is a fallback silently active?
`config.json` has `dash_use_triton: true`; `_sparse_base_impl.py` contains a dense "block-mask fallback" for unsupported shapes and a
`[PageRope layer=... ] fallback to 1d: <reason>` print. Triton on sm_121 / aarch64 is a plausible failure point (my guess, unverified).
* `grep -i fallback` the full stdout/stderr of a run; report any hit and its reason.
* Confirm the Triton gather kernels actually launch (profiler or a counter), and `attn_implementation`, dtype, and that nothing runs on CPU.

## D. Is the output correct? (what I checked, plus one test I have NOT done)
1. `python bench/compare_raw.py bench/reference/m3ultra_mlx_raw.txt <your raw_run0.txt> --tokenizer <model dir>`, and again against
   your stored reference. My result for GB10-reference vs MLX: first divergence at char 25; 90.7% of tokens match in order; identical prefix
   7 tokens; per-page text similarity 0.995 / 0.995 / 0.998 / 0.973 / 0.998 / 0.928; bbox median drift 1 px.
2. Determinism: your two runs gave 3934 and 3932 tokens. Diff `raw_run0` vs `raw_run1`: where do they first differ, and why (CUDA atomics,
   MoE routing ties)? MLX gave byte-identical output across runs.
3. Two content differences to arbitrate (I cannot tell which side is right): (a) page 1 header: GB10 reference has **empty** text,
   MLX has `Baidu 百度 | ERNIE`; (b) page 2 (table of contents): GB10 emits 1 text block, MLX emits 35 per-line blocks plus empty `other`
   blocks (+511 of the +630 extra MLX tokens). A third engine (SGLang, task E) or the same run on another GPU would settle it.
4. **Teacher-forced check (decisive, not done yet):** feed prompt + the MLX output tokens through the HF/CUDA model in one forward pass and
   record, per position, argmax agreement and the top-1 minus top-2 logit margin where it disagrees. Disagreements only at tiny margins
   mean numerics; confident disagreements mean a real implementation difference (in the port or in your path). Re-tokenize
   `m3ultra_mlx_raw.txt` and verify `decode(encode(text)) == text` first; if it does not round-trip, ask me for the exact token ids.

## E. A faster engine
Try the README's SGLang path (`wheel/sglang-0.5.20.dev10...whl`, CUDA graphs, `--max-running-requests 1` is fine) on the same 6 images,
same metrics. Note the README client adds the no-repeat n-gram logit processor, which the HF and MLX runs do not use; run with and without
it and say which. If the wheel does not install on aarch64 / sm_121, report the exact error.

## Rules
* Same inputs, base mode, greedy, bf16. Quote tok/s together with the output token count (outputs differ in length, so compare rates not seconds).
* Say what each timer includes. Report negative results too.
