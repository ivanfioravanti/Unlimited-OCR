# GB10 (DGX Spark) findings for DGX_SPARK_TASKS.md

Hardware/software: NVIDIA GB10 (sm_121, aarch64, 121 GB unified), torch 2.13.0+cu130, transformers 4.52.1 (HF path), SGLang wheel 0.5.20.dev10 (own venv).
Same inputs everywhere: `bench/doc/page_1..6.jpg`, base mode, greedy, bf16. Scripts: `bench/gb10_profile.py` (HF, instrumented), `bench/bench_sglang.py`, `bench/make_sweep_table.py`.
Raw results are `bench/reference/gb10_*` (JSON/raw text). Numbers are single runs unless stated.

## A. Inputs identical: YES
`sha256sum bench/doc/page_*.jpg | cut -c1-12` = `f74b84fc2c9c 0a3fbadd3951 4c49db49331d d57e2a04f6f1 3f883038f9d7 b678837bcbf7` (matches the brief).

## Headline
* HF Transformers on GB10: **26.1 ms/token decode forward, 28.0 ms/token incl. python/lm_head/argmax**, 35 tok/s. Not memory-bound: the weight-read floor is ~4 ms/token.
* SGLang on the same GB10: **~6.9 ms/token = 145 tok/s decode, 140 tok/s end-to-end** (n-gram processor off), 129 tok/s with it on. About 4x the HF path.
* The HF output and the SGLang output agree (99.6% of tokens in order, same blocks). **The two content differences in D3 are on the MLX side**, not GB10's.

## B. Where does the time go (HF, 6 pages, 3934 tokens, 112.2 s total wall)
Timers are `torch.cuda.synchronize()`-bracketed; "forward" = inside `model.model(...)` only; "wall" = pre-hook to next pre-hook (adds `lm_head`, `argmax`+sync, page guard, python).

| phase | time | what it includes |
|---|---|---|
| image preprocessing (`_uocr_build_inputs`) | 0.09 s | image load, transforms, tokenization |
| SAM encoder | 0.71 s | inside prefill |
| CLIP encoder | 0.09 s | inside prefill |
| prefill (whole first decoder call) | 1.96 s | vision encoders + projector + 12 layers over 1649 prompt tokens |
| decode, forward only | 102.75 s | 3933 steps, mean 26.1 ms |
| decode, wall | 110.0 s | mean 28.0 ms; 1.9 ms/token is lm_head + argmax sync + guard + python |

* Streaming is NOT the cause: `eval_mode=True` (no TextStreamer) gives 35.08 tok/s vs 35.0-36.3 with streaming.
* **Per-quartile decode forward ms/token (6 pages): 23.6 / 25.0 / 27.0 / 28.8** (wall 25.5 / 26.9 / 28.9 / 30.7). It **grows ~22%** over 3.9k generated tokens (MLX was flat 5.5 -> 5.65). 3 pages: 22.5 / 22.6 / 22.9 / 23.5; 4 pages: 22.6-25.1; 5 pages: 23.3-27.3. So per-token cost depends on sequence length in the HF path; it is not a dense-attention blow-up (22%, not several x), but it is not constant either.
* Profiler (`torch.profiler`, 60 decode steps; early = steps 3-63, late = steps 3500-3560; profiler overhead inflates wall): ~3,340 CUDA kernels per decode step (354 matmul calls/step, one per expert/projection, mostly bf16 `gemv`). GPU kernel busy time **13.0 ms/step early, 21.0 ms/step late**, while an un-profiled step takes ~24-29 ms of forward wall time. So early on roughly half of each step the GPU is idle (launch/CPU bound), and the growth later is GPU-side: `direct_copy` 86 -> 267 ms and `Cat` 58 -> 162 ms per 60 steps (dtype copies and cache concatenation scale with cache length), plus fp32 `gemv2T` (107 ms) and fp32 gemv (25 -> 71 ms) appearing late (index/score matmuls over the whole cache). bf16 weight gemv is ~6.3 ms/step, near the memory floor, and constant. See `gb10_hf_kernels_decode_{early,late}.txt`.
* `nvidia-smi dmon -s u` during the 6-page run: SM-active mean **59.9%** over the 63 non-zero 2 s samples (`gb10_hf_p6_dmon.txt`). Note that this metric means "a kernel was running", not occupancy. It is consistent with launch gaps, not with a saturated GPU.
* Not done: a CPU-side profile to prove the Grace-core launch-bound hypothesis (the evidence above is indirect: 3.3k kernels/step, ~50% GPU idle early). Decode is not CUDA-graphed in the HF path.

## C. Fallbacks
* `grep -iE "fallback|falling back|not supported|triton"` over full stdout+stderr: 12 hits, all `[DASH] Triton block-sparse path active (S=1649, block_size=32, remote_topk=12, local_window=128)` (once per layer, **prefill only**). For 3/4/5 pages the only hit is a `tl.make_block_ptr is deprecated` UserWarning. No `[PageRope ...] fallback to 1d`, no `[SparseAttn] triton gather path failed ... falling back` line, no dense-fallback message.
* Config at runtime: `attn_implementation = eager`, bf16, 0 parameters on CPU, peak 6.83 GB.
* **Decode runs no custom Triton kernel at all.** Confirmed in the profiler trace (not by name matching): in both the early and late windows the only kernels are PyTorch ATen/cuBLAS kernels plus `bitonicSortKVInPlace`; there is no Triton kernel launch. Prefill's Triton path is confirmed by the log line only (I did not profile prefill). So the decode path is the eager PyTorch implementation of DASH (topk/entmax/gather via aten ops), which is what the thousands of small kernels are.
* Unexplained: `[PageRope layer=0 row0] ... ok=False` and `[PageAwareKL ...] ... ok=False` diagnostic lines in the prefill log (all pages, all layers). I do not know what `ok` checks; the output quality is unaffected.

## D. Correctness
1. `compare_raw.py m3ultra_mlx_raw.txt gb10 run0 --tokenizer`: reproduces the brief (first divergence at char 25, 4567 vs 3937 tokens, identical prefix 7, 90.7% of GB10 tokens matched in order).
2. Determinism (HF, run 0 vs run 1 of the original run): first diff at char 2177, page 2, block 10, `text[112, 139, ...` vs `[112, 138, ...` (one bbox digit). 3937 vs 3935 tokens, 99.6% matched, per-page text similarity 1.000 on 5 of 6 pages (0.998 on page 6). Cause not isolated (bf16 + non-deterministic reduction order in CUDA; my guess only). The new un-streamed HF runs give 3934 tokens again. SGLang is deterministic across my runs (3933 tokens each time, with and without the n-gram processor).
3. Arbitration (third engine = SGLang, no-ngram and ngram both): page 1 header is **empty** (`header[117, 40, 396, 73]<FORK>\n`) and page 2 is **3 blocks**, exactly like the GB10 HF reference. So (a) `Baidu 百度 | ERNIE` and (b) the 35 per-line page-2 blocks are MLX-specific. SGLang vs HF: 99.6% of tokens match in order, per-page text similarity 1.000 except page 6 (0.997); first divergence at char 35 (a bbox coord 72 vs 73). SGLang vs M3 Ultra shows the same differences as HF vs M3 Ultra.
4. **Teacher-forced check** (`gb10_d4_teacher_forced_mlx_result.json`): prompt + the 4567 MLX output tokens (round-trip `decode(encode(text)) == text` verified TRUE) fed through the HF/CUDA decode path step by step (same DASH decode config as generation). **Argmax agrees at 4514/4567 positions (98.84%)**; 53 disagree.
   * 38 of 53 have top1-top2 margin < 1.0 (29 < 0.5; median 0.25; many exactly 0.0, which are ties at bf16 logit granularity). They are almost all bbox digits (`38` vs `39`, `371` vs `372`): numerics.
   * Only **2 have margin >= 2**, and they are the page-3/page-4 sentence "Document parsing ...":
     - pos 1448 (page 3): MLX wrote " recess" (in "recessions have undergone"); HF's top choice is " is", margin 2.94, and the forced token is **10.3 logits below** HF's top. Confident disagreement.
     - pos 2294 (page 4): MLX wrote " is"; HF's top is " has", margin 7.1, forced token 12.4 logits below. Confident disagreement.
   * Reading: apart from those two spots (the fragile sentence), MLX and HF/CUDA disagree only at numerical noise level. The " recess" token is something the HF/CUDA model assigns very low probability to (10 logits), so that one is a real divergence (or an extreme numeric amplification) in the MLX run at that sentence, not tie-breaking. Also note HF itself (and SGLang) get this sentence wrong, in a different way ("is having undergone", see F), so the model is fragile there on every engine.

## E. SGLang (works on aarch64/sm_121)
`wheel/sglang-0.5.20.dev10+g8d62f5fb9c-py3-none-any.whl` is stored in git as an LFS pointer (133 bytes): fetch the real file from GitHub (sha256 `a3f3e2c9...ba70d`). Then `uv pip install` it with Python 3.10 pulls torch 2.13.0+cu130, flashinfer 0.6.18, sglang-kernel 0.4.7: installs and serves with the README flags (CUDA graphs on, no changes). No error to report.

Client = `bench/bench_sglang.py` (streaming, base mode, `max_tokens=24576`). "End-to-end" = POST to last token (includes server-side image preprocessing, prefill, decode, HTTP); prefill = time to first streamed token; decode tok/s = tokens / (total - TTFT).

| 6 pages | tokens | end-to-end tok/s | decode tok/s | prefill s |
|---|---|---|---|---|
| n-gram processor ON (README client) | 3933 | 129.2-131.3 | 133.2-135.0 | 0.7-1.5 |
| n-gram processor OFF (HF/MLX parity) | 3933 | 140.2-140.9 | 144.5 | 0.7-0.8 |

The processor costs ~8% and does not change the content: outputs are byte-identical with and without at 4, 5 and 6 pages; at 3 pages they differ by one bbox pixel (577 vs 576, same text); no loops either way. Sweep: see `gb10_page_sweep.md` (all pages 3-6, both modes). SGLang memory (~80 GB reserved by `--mem-fraction-static 0.7`) is not comparable to HF's 6.83 GB.

## F. Page sweep, ground truth
See `gb10_page_sweep.md`. Summary (HF and SGLang agree):
* **GB10 never looped** (3, 4, 5, 6 pages; all EOS; `<PAGE>` count == N; HF 1669/2623/3231/3934 tokens, SGLang 1670/2624/3230/3933).
* Per-page PDF similarity is identical to the earlier reference and to M3 Ultra within 0.004 (6 pages: 0.926 1.000 0.998 0.979 0.925 0.898 HF / ...0.900 SGLang).
* The page-3 glitch is reproduced: 3 pages -> "Document parsing **has undergone** a major paradigm shift" (correct, page 3 = 1.000); **4, 5 and 6 pages -> "is having undergone"** on both engines, with and without the n-gram processor (text identical in all cases). It is deterministic and the same wording at 4, 5 and 6 pages, so it is not more frequent with more pages beyond the 3 -> 4 transition. It appears once the request has >= 4 pages; the M5 Max loop starts at the same 4-page transition.

## Caveats
* Single runs; HF run-to-run differs by 2 tokens (bf16), SGLang matched exactly.
* The kernel-time profile has profiler overhead; absolute ms are indicative, ratios early/late are the point.
* I did not determine whether the HF decode being eager is intended; the HF code has a Triton prefill path (`dash_use_triton: true`) and a separate eager decode path.
* tok/s for HF includes everything inside `infer()` (preprocessing, prefill, decode, python); for SGLang it is HTTP-to-HTTP.
