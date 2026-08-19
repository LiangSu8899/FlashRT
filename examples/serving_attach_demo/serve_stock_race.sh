#!/bin/bash
# Race arm A: the stock server at the best context it can actually
# boot on this card (its own error at 262144 estimates ~102K), with
# prefix caching on so each turn prefills only its increment. The
# race form runs without speculative decode (this vLLM series
# disables prefix caching under it).
MODEL=${MODEL:?set MODEL to the checkpoint path or hub id}
CTX=${CTX:-102400}
vllm serve "$MODEL" --served-model-name demo --port 8000 \
  --trust-remote-code --max-model-len "$CTX" \
  --gpu-memory-utilization 0.90 --max-num-seqs 1 \
  --enable-prefix-caching
