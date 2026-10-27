source ~/venvs/uocr-sgl/bin/activate
export SGLANG_UOCR_EANABLE_DOC_LIMIT=1 SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW=10 SGLANG_UOCR_DOC_BASE_PAGE_WINDOW=10 SGLANG_UOCR_MAX_PAGE_TOKENS=20000
export SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB=64 SGLANG_UOCR_SHM_WRAP_MODEL_DATA=1 SGLANG_UOCR_SHM_WRAP_MIN_MB=1 SGLANG_VLM_CACHE_SIZE_MB=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python -m sglang.launch_server --model-path baidu/Unlimited-OCR-2 --served-model-name Unlimited-OCR-2 \
 --chat-template unlimited-ocr --tokenizer-backend fastokens --tokenizer-worker-num 8 --mem-fraction-static 0.7 \
 --max-running-requests 64 --max-prefill-tokens 2048 --enable-custom-logit-processor --enable-streaming-session \
 --allow-auto-truncate --disable-radix-cache --mm-preprocess-cache-size-mb 0 \
 --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 64 --cuda-graph-backend-prefill full \
 --cuda-graph-max-bs-prefill 16384 --cuda-graph-bs-prefill 1024 2048 \
 --cuda-graph-config '{"prefill": {"backend": "full", "full_prefill_max_req": 64}}' --host 127.0.0.1 --port 10000
