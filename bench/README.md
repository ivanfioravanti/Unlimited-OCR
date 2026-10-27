# Cross-machine benchmark

Same input (first 6 pages of `Unlimited-OCR-2.pdf`, 150 dpi), same metrics, any backend.

```shell
pip install pymupdf
python bench/prepare_doc.py            # -> bench/doc/page_1..6.jpg
python bench/bench_transformers.py bench/doc --runs 2 --out bench/out_cuda   # CUDA/MPS via HF Transformers
```

## Protocol for other backends (e.g. mlx-vlm)
* Input: `bench/doc/page_*.jpg` in page order, one multi-page request, prompt `"<PAGE><image>" * 6 + "Multi page parsing."`, base mode (no tiling), greedy (temperature 0), bf16 if available.
* Do 2 runs in one process after loading; run 0 includes warmup. Time only generation (not model load or image decode if separable; say which).
* Write `result.json` with the same schema as `reference/gb10_cuda_transformers_result.json`:
  `device, gpu (chip name), torch/framework version, dtype, pages, load_s, runs:[{run, seconds, output_tokens, tok_per_s, peak_mem_gb}]`. Add `prefill_s`/`prompt_tokens` if your backend reports them.
* Save raw output of run 0 as text and compare to `reference/gb10_cuda_transformers_raw.txt` (16.7 KB, ~3934 tokens). Report how many tokens/blocks match and any divergence (e.g. first differing block); small drift is expected from bf16 numerics.

## Reference (NVIDIA GB10 / DGX Spark, Transformers 4.52.1, torch 2.13+cu130, bf16)
108.5 s / 3934 tokens = 36.3 tok/s (run 0), 112.3 s / 3932 tokens = 35.0 tok/s (run 1), 6.83 GB peak GPU mem, 18.1 s load.

## Apple M3 Ultra (512 GB), mlx-vlm, bf16
Run with `python bench/bench_mlx.py bench/doc --model <Unlimited-OCR-2 dir> --runs 2 --out bench/out_mlx`, using mlx-vlm with
[Blaizzy/mlx-vlm PR #2601](https://github.com/Blaizzy/mlx-vlm/pull/2601) (page2d RoPE + DASH sparse attention; not in a release yet; tested at `223c2980`).
Files: `reference/m3ultra_mlx_result.json`, `reference/m3ultra_mlx_raw.txt`, `reference/m3ultra_mlx_vs_gb10_compare.txt` (from `compare_raw.py`).

| | GB10 CUDA, HF Transformers | M3 Ultra, mlx-vlm |
|---|---|---|
| Time, run 0 / run 1 | 108.5 s / 112.3 s | 26.6 s / 26.4 s |
| Output tokens | 3934 / 3932 | 4566 / 4566 |
| Tokens/s, whole request incl. prefill | 36.3 / 35.0 | 172.0 / 172.8 |
| Decode-only tokens/s | n/a | 177.7 |
| Prompt tokens, time to first token | n/a | 1649, 0.74-0.86 s (incl. image preprocessing) |
| Peak memory | 6.83 GB (CUDA allocated) | 8.95 GB (MLX peak, unified memory) |
| Load | 18.1 s | 0.6 s (checkpoint already in page cache) |

Tokens/s over the whole request is ~4.7-4.9x higher on the M3 Ultra. Wall time is only ~4.1-4.3x lower because the Mac emitted 16% more
tokens (see below), so compare the rates, not the seconds. The two backends are not equivalent engines: the GB10 number is eager HF
Transformers, the Mac number is an mlx-vlm implementation with custom Metal kernels, so this is not a hardware-only comparison.

### Output agreement with the CUDA reference (run 0)
The outputs are **not** token-identical: first divergence at char 25 (header bbox `[117, 40, 396, 72]` vs `[117, 38, 396, 73]`, and the
Mac transcribes the header logo text `Baidu 百度 | ERNIE` where the reference leaves it empty). Token level: 90.7% of the reference tokens
reappear in order in the MLX output, but the identical prefix is only 7 tokens. By page:
* Prose agrees (text similarity 0.995-0.998 on pages 1, 2, 3, 5), and matched blocks' bboxes differ by a median of 1 px (max 2-5 px on
  pages 2-5). The 316 / 247 px maxima reported for pages 1 and 6 are `image` blocks segmented differently (page 1: one wide figure box
  in the reference vs two figure boxes in MLX), not coordinate drift.
* Page 2 (table of contents): same entries, but the reference emits one 3-block text list, MLX emits 35 per-line blocks plus empty
  `other` blocks. Tokens per page, MLX minus reference: +25, **+511**, +17, +30, +36, +11 (total +630), so page 2 is 81% of the extra tokens.
* Pages 4 and 6 (equations): similarity 0.973 / 0.928, from LaTeX style (`$ x $` spacing, `\tag {1}` vs `<equation>(1)`,
  `\mathcal{Y}` vs `y`) and some differently transcribed sub-expressions.
Both backends are greedy, but the CUDA reference is itself not run-to-run identical (3934 vs 3932 tokens), so some near-tie divergence is
expected. This comparison cannot separate bf16 drift from a porting difference; a teacher-forced check (feed the reference tokens through
the MLX model and compare argmax per position) would.

## Quality: base mode vs gundam (mlx-vlm; Apple M3 Ultra and M5 Max)
The 6-page protocol above uses base mode (one 1024 view per page), the auto policy for several pages. The official evaluation layout is
**gundam**: each page tiled (1024 global view + up to 16 640px tiles). `python bench/bench_mlx.py bench/doc --cropping` runs it.
Ground truth is the PDF's own text layer (`score_vs_pdf.py` = per-page similarity, `unknown_words.py` = words the output has but the page
does not; a misread word shows up in the latter, the former barely moves).

| 6 pages, M3 Ultra | base | gundam |
|---|---|---|
| page 3, first body sentence (PDF: "Document parsing **has** undergone...") | "recessions have undergone" | **"has undergone"** |
| words absent from the PDF page, pages 1-6 | `hundred`, `snarcv recessions`, `cross`, `experts` | `hundred`, `snarcv ubbw`, `cross`, `experts` |
| prompt tokens / output tokens | 1649 / 4566 | 5429 / 3987 |
| whole request / decode-only | 26.5 s / 178 tok/s | 25.0 s / 171 tok/s (prefill 1.9 s) |

* 15-page document, M3 Ultra: both modes finish with 15 pages. Words absent from the PDF: 10 (base) vs 8 (gundam); gundam fixes the page-3
  sentence and page 13 (`alvettri dshattention` in base, clean in gundam). Time-to-first-token 1.9 s vs 5.1 s, total 84.5 s vs 86.9 s,
  decode 174 vs 168 tok/s. Per-page similarity is within 0.009 everywhere (pages 9-10 are tables/figures and score low in both).
* Hardware stability: in gundam mode the M3 Ultra and M5 Max outputs have identical content (text similarity 1.000 on all pages, 11 of 62
  bbox coordinates differ by 1-2 px). In base mode they diverged on page 3 and, without the generation guards, the M5 Max looped.
* The native engines (HF and SGLang on the GB10) also misread that sentence at 4+ pages in base mode, so base mode's resolution is the
  limit, not only this port. Gundam on the native side: see `DGX_SPARK_TASKS.md` task G.
* What gundam does not fix: the tiny italic attribution line on page 3 ("Snarcy on huggingface (Adapted from UBW)") is misread by every
  engine and mode tried (`snarcv`, `ubbw`).
* Scope: one document (this technical report), 15 pages. Treat it as evidence, not a benchmark.

## Cross-engine results, 6 pages (GB10 numbers from `benchmark-gb10-findings`; greedy, bf16, same JPEGs)
Whole-request tokens/s (prefill included). Timers differ: HF = inside `infer()`, SGLang = HTTP POST to last token, MLX = in-process
including image preprocessing. Output lengths differ by engine and run (the page-2 block layout, see below), so compare rates, not seconds.

| engine | base | gundam | decode-only (base) |
|---|---|---|---|
| GB10, HF Transformers (eager) | 35 tok/s | 29.5 tok/s | ~26 ms/token (grows ~22% over the run) |
| GB10, SGLang, n-gram off | 140 | 135 | ~6.9 ms/token |
| GB10, SGLang, n-gram on | 129 | 122 | |
| M3 Ultra, mlx-vlm, guards on | 168 | 157 | 5.6 ms/token (flat) |
| M5 Max, mlx-vlm, guards on | 206 | 177-189 | ~4.8 ms/token |

Native gundam passes `check_quality.py` at 3, 4, 5 and 6 pages on HF and SGLang (re-run independently from the raw files); native base fails the
canary from 4 pages on both. MLX gundam passes at 6 pages on both Macs.

### Fidelity of the MLX port, gundam mode
* Native HF tokens through MLX (`mlx_teacher_forced.py`): 98.84% overall; bbox-coordinate tokens 95.69% (47 of 1091), **all other tokens
  99.83%** (6 of 3479), largest text gap 2.25 logits, none above 3 (`reference/m3ultra_mlx_teacher_forced_vs_hf_gundam.txt`).
* MLX tokens through native HF (GB10 side): 98.42% overall, 60 of 63 disagreements are bbox digits; the same split gives **99.91%** on the
  other 3261 tokens (3 disagreements, gaps 1.6 / 0.5 / 0.25). One bbox digit trails by 4.5 logits (a tie at the top, page-5 chart).
  Overall agreement is dominated by bbox digits (18% of positions, +-1-2 px), so the text/structure rate is the meaningful number.
* One small text difference: MLX gundam writes "Adapted from U**BB**W" where native HF and SGLang write "UBW" (native prefers it by 1.2-1.6
  logits). MLX base mode reads it correctly. The "Snarcv" (for "Snarcy") misread is shared by every engine and mode.
* The page-1 header and page-2 table of contents flip between "empty header, 3 blocks" and "Baidu header, 35 blocks" with the (mode, run),
  not with the engine. Forcing only the first 20 native tokens (`mlx_prefix_force.py`, ties at output positions 7 and 13) makes MLX
  take the native branch (header `Baidu 百度 | ERNIE`, page 2 as 37 blocks) and then match the native document at 0.993-0.999 per-page text
  similarity (`reference/m3ultra_mlx_gundam_prefix20_raw.txt`). So neither layout is a port error.

### Page-count sweep on the Macs (`mlx_sweep.py`, `reference/sweep_m3ultra`, `reference/sweep_m5max`, `reference/mlx_sweep_verdicts.json`)
Pages 3-6 x {base, gundam}, one request per cell, scored with `check_quality.py`. M3 Ultra and M5 Max agree with the native engines cell for
cell: base passes at 3 pages and fails the canary sentence from 4 pages on; gundam passes everywhere. With the generation guards on, the M5
Max base runs at 4 and 6 pages no longer run away but still fail (the filler paragraph on page 3). Quality verdicts and the M3 Ultra timings
are reliable; the **M5 Max timings in that sweep are not** (the laptop was in interactive use: 6-page gundam took 32 s against 21 s idle,
with per-token time rising inside the request). Use the idle runs above for M5 Max speed.

### Re-test on the updated PR head (mlx-vlm PR #2601 @ 95878139, which now contains the generation guards)
The PR author cherry-picked the guards and README commits (tree identical to our `uocr2-generation-guards` branch) and rewrote the PR history
underneath. Everything was re-run from that head on both Macs: unit tests pass (Unlimited-OCR model and guard tests, 537 in `test_generate.py`,
processor tests); all 8 M3 Ultra and all 8 M5 Max sweep outputs (pages 3-6, base and gundam) are **byte-identical** to the earlier sweeps, with
the same verdicts; native HF gundam tokens through MLX give the same 98.84% overall / 99.83% on text tokens. Across chips, gundam content is
identical at every page count (3-6; only 5-11 bbox digits differ by 1-2 px) while base content differs on every page count. M5 Max timings in
that re-run are not usable (the laptop was under heavy load: gundam at 3 pages took 33 s instead of 9 s).

### Speed re-measured on the PR head (mlx-vlm PR #2601 @ 95878139), 6 pages, 3 runs each (`reference/pr_head_95878139/`)
Whole request, prefill and image preprocessing included, greedy, bf16, `bench_mlx.py` (add `--cropping` for gundam).

| | base tok/s (runs) | gundam tok/s (runs) | decode-only |
|---|---|---|---|
| M3 Ultra | 170.4 (170.1, 170.5, 170.6) | 160.4 (160.3, 160.6, 160.5) | 5.7 ms/token |
| M5 Max | 199.2 (195.5, 201.2, 201.0) | 184.3 (192.3, 185.3, 175.1) | 4.9 / 5.1 ms/token |

Against SGLang on the GB10 (141 base, 135 gundam, n-gram off): 1.21x / 1.19x (M3 Ultra), 1.41x / 1.36x (M5 Max). Against HF eager (35 / 29.5): 4.9x to 6.2x.
The M5 Max is a MacBook: its gundam runs fell from 192 to 175 tok/s over three back-to-back runs (heat), and `m5max_load_log.txt` shows the
machine was nearly idle (WindowServer 26% falling to 4%). Earlier M5 Max runs on the pre-rewrite head gave 205.5 / 206.7 (base) and 176.3 / 188.6 (gundam).

### 15 pages (M3 Ultra, PR head 95878139; `reference/sweep_m3ultra_p15/`, produced by `mlx_sweep.py ... --pages 15`)
| mode | prompt / output tokens | finish | time to first token | total | tok/s | check_quality |
|---|---|---|---|---|---|---|
| base | 4115 / 14405 | stop, 15 pages | 1.8 s | 84.4 s | 170.7 | pages, no-loop, prose pass; canary fails (page-3 sentence) |
| gundam | 13565 / 13759 | stop, 15 pages | 5.0 s | 87.6 s | 157.0 | all pass |

Per-token time over the output: base 5.69 to 5.76 ms, gundam 5.79 to 6.13 ms (about +6% over 13.8k tokens, so close to flat but not exactly).
GB10 SGLang, gundam, whole document in one window (`SGLANG_UOCR_EANABLE_DOC_LIMIT=0`): 14.3k tokens in 104.4 s (`benchmark-gb10-findings`).
GB10 SGLang base mode loops from page 3 at 15 pages in every configuration tried with the n-gram rule off. The MLX run above had the n-gram rule on, so
"MLX finishes where SGLang loops" is not yet a like-for-like comparison (it needs SGLang base at 15 pages with the n-gram processor on).

### Post cards (`x_cards.html`, `x_cards/*.png`)
`x_cards.html` holds the data for three 1600x900 cards (open `x_cards.html#c1`, `#c2` or `#c3` to see one card at exact size). The PNGs in `x_cards/`
are 3200x1800 exports made with headless Chrome (`--force-device-scale-factor=2 --window-size=1600,900 --screenshot`).

### Can the MLX port batch? Concurrency on the Macs (`mlx_concurrency.py`, `reference/mlx_concurrency/`)
Not inside one process: the port's DASH attention raises `supports batch size 1` (`dash.py`), and mlx-vlm's generic `batch_generate` path does not
accept this model's page-aware prompt (it fails even with one request: "image tokens in the prompt do not match the number of images").
mlx-vlm's server has continuous batching in general, but that path needs batch support in the model, which this port does not have yet.
What does work is several independent processes sharing the GPU. C identical 6-page gundam requests started together (n-gram rule off, as in the
SGLang sweep; every worker loaded and warm before the start):

| C | M3 Ultra aggregate tok/s (mean latency) | M5 Max aggregate tok/s (mean latency) | GB10 SGLang aggregate tok/s (mean latency) |
|---|---|---|---|
| 1 | 159.5 (25 s) | 193.4 (21 s) | 134.9 (34 s) |
| 2 | 245.4 (33 s) | 259.5 (31 s) | n/a |
| 4 | 319.5 (50 s) | 272.7 (58 s) | 474.0 (39 s) |
| 8 | 329.9 (97 s) | 173.5 (183 s) | 549.7 (65 s) |
| 16 | 325.9 (195 s) | not run (128 GB) | 658.6 (111 s) |

The M3 Ultra roughly doubles its single-request throughput and then saturates near 330 tok/s; the M5 Max peaks at 273 with 4 processes and gets
slower at 8 (cause not investigated: a laptop, ~10 GB per process, 8 processes use ~80 of 128 GB). SGLang keeps scaling to 659 (4.9x). So the
DGX Spark's lead in aggregate throughput holds, at about 1.5x (C=4) to 2x (C=16) the M3 Ultra, but the Mac is not limited to one request at a time.
In a first run of this sweep on the M3 Ultra, 1 of 4 and 2 of 16 workers produced no result (stderr was not captured, so the cause is unknown); the
harness now records failed workers, and the re-run above had all 31 workers succeed. Only the re-run is reported.

## Results on mlx-vlm main (5061bd7f, PR #2601 merged on 2026-10-27) (`reference/main_5061bd7f/`)
**Main is not the PR head we tested before (95878139).** Before merging, the maintainer added two commits: `00b1ca91` removed our generation guards and the
`generate_step` hook ("no model-specific generation guards are installed automatically"; callers can still pass `logits_processors`), and `c3ae82e2` made
**tiled (Gundam) pages the default for every page count** and added uniform KV-cache quantization (`--kv-bits`). Everything below was re-run on main on both
Macs. The scripts now pass `cropping` explicitly (main would otherwise silently run "base" tests in Gundam mode) and only pass guard kwargs when the
build has them. Unit tests on main: 19 Unlimited-OCR model tests, 21 processor tests, 536 in `test_generate.py`, all pass.

| 6 pages, whole request, tok/s | base (opt-in, `cropping=False`) | Gundam (default) |
|---|---|---|
| M3 Ultra, 3 runs | 175.1 (174.3, 175.6, 175.6), decode 5.55 ms/token | 164.4 (165.1, 165.2, 162.8), decode 5.60 ms/token |
| M5 Max, 3 runs | **runs away**: 24,576 tokens, 122 s, no EOS | 189.1 (190.0, 194.6, 182.8), 3,987 tokens, EOS |

Dropping the guards makes the M3 Ultra about 2.5% faster than on the PR head (170.4 / 160.4). Against SGLang (141.0 / 134.9) that is 1.24x (M3 base),
1.22x (M3 Gundam) and 1.40x (M5 Gundam); against HF eager (35.4 / 29.1) 4.9x to 6.5x.
* **Quality sweep, pages 3-6, base and Gundam, both Macs** (`sweep_*`, `verdicts.json`, cap 1,500 tokens per page): Gundam passes every check at every
  page count on both Macs, with content identical across the two chips (only bbox digits differ) and byte-identical to the earlier guarded runs. Base fails the
  page-3 sentence from 4 pages on, as on the native engines. **On the M5 Max base mode at 4 and 6 pages hits the cap again** (the loop the guards used to stop);
  the M3 Ultra base outputs do not loop and are byte-identical to the guarded runs.
* **15 pages, M3 Ultra** (`p15_m3ultra`): base 14,400 tokens in 84.1 s, EOS, 15 pages, fails only the page-3 sentence; Gundam 13,664 tokens in 85.1 s (160.5 tok/s),
  all checks pass. GB10 SGLang Gundam, whole document in one window: 104.4 s.
* **Fidelity** (native HF Gundam tokens through MLX main, M3 Ultra): unchanged, 98.84% overall, 99.83% on text tokens, largest text gap 2.25 logits.
* **Concurrency** (`concurrency/`, separate processes, Gundam, 6 pages, no guards): M3 Ultra 161.6 / 253.9 / 324.6 / 333.9 / 329.2 tok/s at 1 / 2 / 4 / 8 / 16 (all
  31 workers succeeded). M5 Max, two runs: 196 / 253 / 232 / 205 and 202 / 294 / 293 / 196 (mean 199 / 273 / 262 / 200). One of the 8 workers in the second run failed with
  `Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)` at prefill: eight Gundam processes (about 10 GB each) exhaust GPU memory on the 128 GB laptop, which
  explains the drop at 8. SGLang: 135 / 474 / 550 / 659 at 1 / 4 / 8 / 16.

**Correction to an earlier claim.** I said the proposed guards "never fire on healthy output". That held for the 3- to 6-page text runs I compared byte for byte, but
not for a long document with tables: in the 15-page Gundam run the n-gram rule changed page 10 (a results table; `^{\text{Edit}}` instead of `^{Edit}`, 13,759 vs 13,664 tokens)
and the unguarded output is closer to the PDF text on that page (0.814 vs 0.802). That is consistent with main not enabling them by default.
