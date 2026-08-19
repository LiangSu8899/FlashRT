#!/bin/bash
# Stock vLLM server. MODEL points at the checkpoint directory or hub id.
# At CTX=262144 on a 32 GB card this boot REFUSES — that refusal is
# scene one of the demo.
MODEL=${MODEL:?set MODEL to the checkpoint path or hub id}
CTX=${CTX:-262144}
vllm serve "$MODEL" --served-model-name demo --port 8000 \
  --trust-remote-code --max-model-len "$CTX" \
  --gpu-memory-utilization 0.90 --max-num-seqs 1 \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":6}'
