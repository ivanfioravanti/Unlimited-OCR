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
