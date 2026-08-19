#!/bin/bash
# Same server, three env vars more. FRT_REPO points at the FlashRT
# checkout. The long-context boot adds the explicit KV budget the
# docs describe (utilization-based sizing leaves no activation
# headroom at 200K).
MODEL=${MODEL:?set MODEL to the checkpoint path or hub id}
FRT_REPO=${FRT_REPO:?set FRT_REPO to the FlashRT checkout}
CTX=${CTX:-262144}
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHONPATH="$HERE" FRT_VLLM_ATTACH=1 FRT_VLLM_STRUCTURES_PATH="$FRT_REPO" \
vllm serve "$MODEL" --served-model-name demo --port 8000 \
  --trust-remote-code --max-model-len "$CTX" \
  --gpu-memory-utilization 0.90 --max-num-seqs 1 \
  --kv-cache-memory 10400000000 --max-num-batched-tokens 4096 \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":6}'
