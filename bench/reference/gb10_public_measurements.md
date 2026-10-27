# GB10 measurements for the public write-up

Hardware/software: NVIDIA GB10 (DGX Spark, sm_121, aarch64, 20 cores, 121 GB unified memory), driver 580.178.04, CUDA 13.0, torch 2.13.0+cu130, transformers 4.52.1 (HF), SGLang 0.5.20.dev10+g8d62f5fb9c (flashinfer 0.6.18, sglang-kernel 0.4.7).
Inputs: the same 6 JPEGs (hashes verified), `bench/doc/page_1..15.jpg` for the 15-page runs (pages 1-6 unchanged), greedy, bf16, SGLang n-gram processor OFF unless stated.
Raw files: `bench/reference/gb10_*` (listed per section). Scripts: `gb10_profile.py` (HF), `bench_sglang.py`, `bench_sglang_conc.py`.

## 0. Clocks, power, exact flags
* Power state during the SGLang session (`gb10_sglang_session_clocks_power.csv`, 1157 samples at 2 s, `nvidia-smi --query-gpu`): while busy, SM clock mean **2444 MHz** (min 2411, max 2463; the chip's listed max is 3003 MHz), P0, board power mean **33.5 W** (peak sample 74 W), max temperature **72 C**, clocks-throttle-reasons bitmask `0x0` in **every** sample (no thermal or power throttling). Idle is P8 at ~208 MHz / ~5 W. GPU persistence mode enabled; CPU governor `performance`. nvidia-smi reports power limit and memory clock as N/A on this platform; no power-mode setting was changed by me. The HF repeats were run before the logger was started, so they have no clock log.
* SGLang server, exact command: `bench/gb10_sglang_serve.sh` (README flags verbatim, host 127.0.0.1): `--mem-fraction-static 0.7 --max-running-requests 64 --max-prefill-tokens 2048 --enable-custom-logit-processor --enable-streaming-session --allow-auto-truncate --disable-radix-cache --mm-preprocess-cache-size-mb 0 --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 64 --cuda-graph-backend-prefill full --cuda-graph-max-bs-prefill 16384 --cuda-graph-bs-prefill 1024 2048 --cuda-graph-config '{"prefill": {"backend": "full", "full_prefill_max_req": 64}}' --chat-template unlimited-ocr --tokenizer-backend fastokens --tokenizer-worker-num 8`, env `SGLANG_UOCR_EANABLE_DOC_LIMIT=1 SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW=10 SGLANG_UOCR_DOC_BASE_PAGE_WINDOW=10 SGLANG_UOCR_MAX_PAGE_TOKENS=20000 SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB=64 SGLANG_UOCR_SHM_WRAP_MODEL_DATA=1 SGLANG_UOCR_SHM_WRAP_MIN_MB=1 SGLANG_VLM_CACHE_SIZE_MB=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The only variant is `bench/gb10_sglang_serve_nolimit.sh` (`SGLANG_UOCR_EANABLE_DOC_LIMIT=0`), used for the 15-page gundam result (section 3).
* Client request: streaming chat completion, `temperature 0`, `skip_special_tokens false`, `images_config.crop_mode` false (base) / true (gundam), `custom_params {page_token_id 128827, eos_token_id 1, max_page_tokens 20000}` (page guard only; n-gram processor omitted).

## 1. Repeats, 6 pages, 3 runs each (`gb10_rep_*`)
All 12 outputs pass/fail their quality checks identically across runs (`check_quality.py`). Timers: HF `total` = whole `infer()` (preprocess + prefill + decode, `eval_mode`, no streamer), HF `prefill` = first decoder call (incl. vision encoders). SGLang `total` = POST to last streamed token, `prefill` = time to first token. Run 1 of SGLang was on a freshly started server (cold), runs 2-3 warm; the HF run 1 values are from earlier sessions (same code).

| engine | mode | run | tokens | total s | tok/s end-to-end | prefill s | decode tok/s |
|---|---|---|---|---|---|---|---|
| HF | base | 1 / 2 / 3 | 3934 / 3934 / 3934 | 112.15 / 110.19 / 110.78 | 35.08 / 35.70 / 35.51 | 1.96 / 1.38 / 1.40 | 35.76 / 36.22 / 36.03 (fwd 26.1 / 25.8 / 25.9 ms) |
| HF | gundam | 1 / 2 / 3 | 4569 / 4569 / 4569 | 154.99 / 154.54 / 161.80 | 29.48 / 29.57 / 28.24 | 2.42 / 2.21 / 2.19 | 30.03 / 30.08 / 28.70 (fwd 31.5 / 31.4 / 33.0 ms) |
| SGLang | base | 1 / 2 / 3 | 3933 / 3933 / 3933 | 27.93 / 27.87 / 27.89 | 140.82 / 141.11 / 141.01 | 0.79 / 0.67 / 0.67 | 144.91 / 144.56 / 144.49 |
| SGLang | gundam | 1 / 2 / 3 | 4570 / 4570 / 4570 | 34.56 / 33.67 / 33.42 | 132.24 / 135.73 / 136.73 | 2.63 / 1.72 / 1.45 | 143.15 / 143.02 / 142.94 |

Spread (min-max over 3): HF base 110.2-112.2 s (1.8%), HF gundam 154.5-161.8 s (4.7%), SGLang base 27.87-27.93 s (0.2%), SGLang gundam 33.4-34.6 s (3.4%, all in the prefill of the cold first run; decode tok/s spread 0.1%). Output text: all 3 HF runs are byte-identical per mode. SGLang runs 2 and 3 are byte-identical per mode, but the cold first run after server start differs from them by a single bbox coordinate (base: `header[117, 40, 396, 72]` vs `73`; gundam: title y `134` vs `135`; same text, 99.8% token match), so SGLang is deterministic once warm and not necessarily on its very first request (cause not investigated). Token counts are the same in all repeats.

## 2. SGLang concurrency, 6 pages, gundam (`gb10_sglang_concurrency_gundam_result.json`)
C identical simultaneous requests, one burst each (default server flags, `--max-running-requests 64`). Aggregate = total output tokens / wall from first send to last finish. Latency = send to last token of that request. No errors. Every request produced 4570 tokens and EOS. (Not quality-checked individually; I kept only request 0 per C, and the C=16 sample passes all five checks.)

| C | wall s | aggregate tok/s | latency s (min / mean / max) | mean TTFT s | per-request tok/s | distinct outputs |
|---|---|---|---|---|---|---|
| 1 | 33.9 | 134.9 | 33.9 | 1.9 | 134.9 | 1 |
| 4 | 38.6 | 474.0 | 38.5 / 38.6 / 38.6 | 4.2 | 118.5 | 1 |
| 8 | 66.5 | 549.7 | 62.6 / 65.0 / 66.5 | 15.5 | 70.3 | 2 |
| 16 | 111.0 | 658.6 | 108.8 / 110.8 / 111.0 | 40.8 | 41.2 | 2 |

Aggregate throughput is 3.5x at C=4 and 4.9x at C=16 relative to C=1; latency grows from 34 s to 111 s. The jump from C=4 to C=8 is mostly TTFT: prefill of 8-16 gundam requests (5.4k tokens each) is serialized (`--max-prefill-tokens 2048`), 15.5 s and 40.8 s mean TTFT. At C>=8 two distinct outputs appeared among identical requests (same token counts; presumably batch-composition numerics in bbox digits, not inspected). One burst per C, no repeats.

## 3. Decode ms/token per quarter, and 15 pages
### Per-quarter decode, 6 pages (SGLang, chunk timestamps; 1 streamed chunk = 1 token, 3932-3933 chunks vs 3933 tokens)
| mode | run | ms/token Q1 / Q2 / Q3 / Q4 |
|---|---|---|
| base | 1 / 2 / 3 | 6.89 6.89 6.92 6.91 / 6.93 6.91 6.93 6.92 / 6.93 6.91 6.93 6.92 |
| gundam | 1 / 2 / 3 | 6.99 6.99 6.99 6.99 / 6.99 6.99 6.99 6.99 / 7.00 7.00 7.00 7.00 |

Flat to 0.5%: per-token cost does not grow with output length in SGLang. (HF, same 6 pages: 23.6 -> 28.8 ms base and 28.6 -> 34.7 ms gundam per quartile, growing; see gb10_findings.md.)

### 15 pages (`gb10_p15_sglang_*`)
| config | prompt tokens | output tokens | finish | quality (pages/no-loop/prose/canary/words) | total s | notes |
|---|---|---|---|---|---|---|
| base, n-gram off (default flags, cap 60000) x2 | 3019 | 60000 | **length** | **FAIL / FAIL / FAIL / FAIL / WARN** | 453 / 442 | **runaway**: only 3 `<PAGE>`, loops "The new technology is a key to the new technology." from page 3; decode stays 7.3-7.9 ms/token for 60k tokens |
| base, `DOC_LIMIT=0`, cap 20000 | 4115 | 20000 | length | FAIL (same loop) | 139.9 | same failure, so not caused by the page window |
| gundam, default flags (10-page window), cap 60000 x2 | 14615 | 60000 | length | PASS / PASS / PASS / PASS / WARN | 489 / 468 | document is complete after ~14.1k visible tokens (15 `<PAGE>`, ends with page number 15) but the server then keeps emitting ~46k invisible tokens until the cap; visible text done at 112.8 s (cap-20000 run); 3rd quartile slows to 9.4 ms/token around the window handoff |
| **gundam, `DOC_LIMIT=0` (whole document, one window), cap 40000, x3** | **13565** | **14299 / 14299 / 14299** | **EOS** | **PASS / PASS / PASS / PASS / WARN** | **104.4 / 105.4 / 104.1** | prefill 4.3 / 5.2 / 4.0 s; decode 142.9 tok/s; ms/token Q1-Q4 7.01 / 7.00 / 6.99 / 7.00 (flat); 3 outputs byte-identical |

* MLX comparison point: M3 Ultra gundam 15 pages, 13.8k tokens in 87 s. GB10 SGLang (single window): **14.3k tokens in 104.4 s** (137 tok/s end-to-end, 143 decode); output length differs by 4% so compare rates (137 vs ~159 tok/s) or seconds with that in mind.
* The words WARN is `popo`, `fired` (the second is in a real reference title, "Fired-orc technical report"; not investigated further). Per-page similarity vs PDF text, 15-page gundam: 0.923 1.000 1.000 0.997 0.925 0.932 0.966 0.974 **0.635** 0.819 0.899 0.946 1.000 1.000 0.999 (page 9 is low; not investigated; pages with equations/figures score lower for every engine).
* **Unexpected: with the README's default 10-page window (`SGLANG_UOCR_EANABLE_DOC_LIMIT=1`, window 10) a 15-page gundam request does not stop on EOS** in my setup: the text is complete but generation continues with empty tokens until `max_tokens`. With `DOC_LIMIT=0` it stops on EOS at 14.3k tokens. I did not look into why; if you use the default flags for long documents, set a `max_tokens` close to the expected output and report time to the last visible chunk (`last_visible_chunk_s` in the result JSON), not total request time.
* 15-page **base** mode fails on every engine configuration tried (SGLang, both flag sets), the same runaway signature as the M5 Max at 4 and 6 pages; on GB10 the failure starts at 15 pages, not 4 (6-page base passes the loop checks, fails only the canary).

## What I did not do
* No HF 15-page run (HF decode is ~26-35 ms/token, ~7-9 min for 14k tokens; say if you want it). No concurrency for base mode, no repeats of the concurrency bursts, no per-request quality check at C>1 other than the one C=16 sample.
