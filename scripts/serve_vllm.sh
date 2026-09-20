#!/usr/bin/env bash
# vLLM server for the generic prefix-cache comparison (Appendix A.3.5).
vllm serve "${QWEN_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}" --port 8000 --enable-prefix-caching --block-size 16
