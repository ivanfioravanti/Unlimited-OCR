# GB10 gundam + quality checks (brief section G)

Same 6 JPEGs (hashes verified earlier), greedy, bf16, `bench/check_quality.py <raw> --pages N --pdf bench/report.pdf` (the repo PDF is an LFS pointer). 1 run per cell.
Timers: HF `prefill s` = first decoder call (vision encoders + projector + prompt, `cuda.synchronize`d), `total s` = whole `infer()` (preprocess + prefill + decode); SGLang `prefill s` = time to first streamed token (includes HTTP + server-side preprocessing), `total s` = POST to last token. HF used `eval_mode=True` (no streamer). Raw outputs: `gb10_hf_[gundam_]p<N>_raw.txt`, `gb10_sglang_[gundam_]p<N>_{ng,nong}_raw.txt`.

## 1. Quality verdicts

| engine | mode | N | tokens | finish | pages | no-loop | prose | canary | words | prefill s | total s | tok/s (end-to-end) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HF | base | 3 | 1669 | EOS | PASS | PASS | PASS | PASS | PASS | 1.545 | 42.85 | 38.95 |
| HF | base | 4 | 2623 | EOS | PASS | PASS | PASS | FAIL | WARN | 1.468 | 68.76 | 38.15 |
| HF | base | 5 | 3231 | EOS | PASS | PASS | PASS | FAIL | WARN | 1.543 | 88.51 | 36.5 |
| HF | base | 6 | 3934 | EOS | PASS | PASS | PASS | FAIL | WARN | 1.956 | 112.15 | 35.08 |
| HF | gundam | 3 | 2234 | EOS | PASS | PASS | PASS | PASS | PASS | 1.894 | 61.85 | 36.12 |
| HF | gundam | 4 | 3217 | EOS | PASS | PASS | PASS | PASS | PASS | 1.989 | 96.49 | 33.34 |
| HF | gundam | 5 | 3856 | EOS | PASS | PASS | PASS | PASS | PASS | 2.244 | 128.56 | 29.99 |
| HF | gundam | 6 | 4569 | EOS | PASS | PASS | PASS | PASS | PASS | 2.417 | 154.99 | 29.48 |
| SGLang (n-gram off) | base | 3 | 1670 | EOS | PASS | PASS | PASS | PASS | PASS | 0.36 | 11.9 | 140.37 |
| SGLang (n-gram off) | base | 4 | 2624 | EOS | PASS | PASS | PASS | FAIL | WARN | 0.71 | 18.86 | 139.16 |
| SGLang (n-gram off) | base | 5 | 3230 | EOS | PASS | PASS | PASS | FAIL | WARN | 0.74 | 23.09 | 139.87 |
| SGLang (n-gram off) | base | 6 | 3933 | EOS | PASS | PASS | PASS | FAIL | WARN | 0.68 | 27.91 | 140.94 |
| SGLang (n-gram off) | gundam | 3 | 2235 | EOS | PASS | PASS | PASS | PASS | PASS | 1.12 | 16.76 | 133.34 |
| SGLang (n-gram off) | gundam | 4 | 3218 | EOS | PASS | PASS | PASS | PASS | PASS | 1.22 | 23.75 | 135.48 |
| SGLang (n-gram off) | gundam | 5 | 3857 | EOS | PASS | PASS | PASS | PASS | PASS | 2.7 | 29.66 | 130.05 |
| SGLang (n-gram off) | gundam | 6 | 4570 | EOS | PASS | PASS | PASS | PASS | PASS | 1.8 | 33.82 | 135.11 |
| SGLang (n-gram on) | base | 3 | 1670 | EOS | PASS | PASS | PASS | PASS | PASS | 0.52 | 12.96 | 128.83 |
| SGLang (n-gram on) | base | 4 | 2624 | EOS | PASS | PASS | PASS | FAIL | WARN | 1.43 | 21.05 | 124.67 |
| SGLang (n-gram on) | base | 5 | 3230 | EOS | PASS | PASS | PASS | FAIL | WARN | 0.88 | 25.1 | 128.68 |
| SGLang (n-gram on) | base | 6 | 3933 | EOS | PASS | PASS | PASS | FAIL | WARN | 0.9 | 30.44 | 129.22 |
| SGLang (n-gram on) | gundam | 3 | 2235 | EOS | PASS | PASS | PASS | PASS | PASS | 1.07 | 17.92 | 124.69 |
| SGLang (n-gram on) | gundam | 4 | 3218 | EOS | PASS | PASS | PASS | PASS | PASS | 2.49 | 26.81 | 120.03 |
| SGLang (n-gram on) | gundam | 5 | 3857 | EOS | PASS | PASS | PASS | PASS | PASS | 1.39 | 30.57 | 126.18 |
| SGLang (n-gram on) | gundam | 6 | 4570 | EOS | PASS | PASS | PASS | PASS | PASS | 3.02 | 37.6 | 121.53 |

Reference points: MLX M3 Ultra gundam raw (`m3ultra_mlx_gundam_raw.txt`) passes all five checks at N=6 (re-run here).

**Answer: yes, native gundam also passes, on both engines.** HF and SGLang in gundam mode pass pages/no-loop/prose/canary/words at every N=3..6 (canary "Document parsing has undergone a major paradigm shift" correct at 4, 5, 6 pages). Native base fails the canary at N>=4 on both HF and SGLang (WARN words: `having`, `recessing`), exactly as expected. So the glitch is a property of base mode in the model, not of MLX or of GB10 numerics. The n-gram logit processor makes no difference to the output in gundam mode (byte-identical for N=3..6) and costs ~7-10% speed. No loops anywhere on GB10 (base or gundam).

Gundam cost: prompt tokens 2717 / 3621 / 4525 / 5429 (N=3..6) vs 827..1649 in base. HF decode ms/token (forward) in gundam rises with the longer cache, 6 pages: 28.6 / 30.3 / 32.4 / 34.7 per quartile (base 23.6..28.8); HF is 29.5 tok/s at N=6 vs SGLang 135 tok/s (n-gram off).

## 2. Teacher-forced fidelity, gundam (`gb10_d4_teacher_forced_mlx_gundam_result.json`)
MLX tokens from `m3ultra_mlx_gundam_raw.txt` (3988 tokens; `decode(encode(text)) == text` verified TRUE) fed through the HF decode path in gundam mode, prompt built with `crop_mode=True` (same as native HF gundam: 5429 prompt tokens), same step-by-step DASH decode as generation.

* **Argmax agreement: 3925 / 3988 = 98.42%** (base-mode run earlier: 98.84%). Rule was >= 98.5%, so **by the stated rule this is a FAIL, by 4 tokens** (3929 would pass).
* 63 disagreements: 56 with margin < 1.0 (35 < 0.5, median 0.25, many exact bf16 ties at 0.0); **60 of 63 are bbox digits** (e.g. `130` vs `126`), 3 are text: pos 915 `BB` vs `BW` (inside a model name on page 3, gap 1.6), pos 3496 ` \\` vs ` S` and pos 3651 `v` vs `emptyset` (both inside LaTeX on page 6, gaps 0.5 and 0.25).
* **Disagreements where MLX's token trails my top choice by > 3 logits: exactly one**, and it is a bbox coordinate, not text: pos 2647, page 5, context `]<FORK>4\n<PAGE><BLOCK>chart[`, MLX `130`, HF top `126`, top1-top2 margin 0.0 (a tie at the top), MLX token 4.5 logits below. **No text disagreement beyond 3 logits.**
* Per rule ("if it fails, say where and stop"): the agreement criterion narrowly fails and I stopped the fidelity investigation here (no hidden-state dumps). My reading, for you to judge: the miss is entirely bbox digit noise plus 3 low-gap text tokens, the no-text->3-logit criterion passes, and the earlier base run (98.84%) had two real >3-logit text divergences on the canary sentence that do NOT appear in gundam. Counts per output page: 15 / 5 / 9 / 11 / 10 / 13 (no clustering).

## 3. Unexpected
* **The header / page-2 differences flip sides between modes.** Base: HF and SGLang give an empty page-1 header and 3 page-2 blocks, MLX gives `Baidu 百度 | ERNIE` and 35 page-2 blocks. Gundam: MLX gives the empty header and 3 page-2 blocks; HF *and* SGLang gundam give `Baidu 百度 | ERNIE` and 35 page-2 blocks (compare_raw MLX-gundam vs HF-gundam: page 2 blocks 3/35). So these content differences are not "an engine is right/wrong": they follow the (mode, run), and look like a bifurcation of the free-running output. Hypothesis, not tested: it is triggered by near-tie bbox digits in the first tokens (teacher-forced logs show exact ties, margin 0.0, at positions 7 and 13 of the output, `38` vs `39`, `73` vs `74`), which then condition whether the header is emitted. A free-run test with a forced first-token tie-break would settle it.
* HF gundam and SGLang gundam agree on 99.9% of tokens in order (4570 tokens each, same blocks; first difference one bbox pixel at char 4315).
* HF native gundam = 4569 tokens (EOS) with page 2 as 35 blocks, MLX gundam = 3988 tokens: the +581 tokens are those 35 page-2 blocks again, not a quality difference (both pass all checks).
