#!/usr/bin/env bash
set -e
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
source ~/ai-systems-venv/bin/activate

SPEC_CONFIG='{"method": "draft_model", "model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 3, "use_heterogeneous_vocab": true}'

echo "Starting vLLM Speculative Decoding Server on port 8000..."
exec vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --speculative-config "$SPEC_CONFIG" \
  --enable-prefix-caching \
  --port 8000 \
  --enforce-eager