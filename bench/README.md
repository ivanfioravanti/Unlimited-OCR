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
