#!/bin/bash
# Race arm B: the attached server at the model's native maximum,
# same flags otherwise. Three env vars are the whole difference.
MODEL=${MODEL:?set MODEL to the checkpoint path or hub id}
FRT_REPO=${FRT_REPO:?set FRT_REPO to the FlashRT checkout}
CTX=${CTX:-262144}
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHONPATH="$HERE" FRT_VLLM_ATTACH=1 FRT_VLLM_STRUCTURES_PATH="$FRT_REPO" \
vllm serve "$MODEL" --served-model-name demo --port 8000 \
  --trust-remote-code --max-model-len "$CTX" \
  --gpu-memory-utilization 0.90 --max-num-seqs 1 \
  --kv-cache-memory 10400000000 --max-num-batched-tokens 4096 \
  --enable-prefix-caching
