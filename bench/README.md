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
