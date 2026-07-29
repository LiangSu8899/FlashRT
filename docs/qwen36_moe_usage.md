# Qwen3.6-35B-A3B text inference

FlashRT runs the language backbone from the official
`Qwen/Qwen3.6-35B-A3B` BF16 checkpoint on an RTX 5090. The checkpoint uses the
same `qwen3_5_moe` text architecture as Nex-N2-mini, so both models share the
same weight loader, prefill, attention, MoE, recurrent-state, and CUDA Graph
decode implementation.

This entry is text-only. It does not load the vision tower and it validates but
does not execute the checkpoint's MTP head. Image/video input and speculative
decode are not part of this interface.

## Requirements

| | |
|---|---|
| Checkpoint | `Qwen/Qwen3.6-35B-A3B` BF16 safetensors |
| Hardware | RTX 5090 / SM120 |
| GPU memory | 32 GB |
| Framework | PyTorch |
| Runtime quantization | NVFP4 |
| Build flag | `-DFLASHRT_ENABLE_QWEN35MOE=ON` |

Configure and build the gated `qwen3_5_moe` kernels:

```bash
cmake -S . -B build -DFLASHRT_ENABLE_QWEN35MOE=ON
cmake --build build -j
pip install -e ".[torch]"
```

## Usage

```python
from flash_rt.frontends.torch.qwen36_moe_rtx import (
    Qwen36MoeTextFrontendRtx,
)

frontend = Qwen36MoeTextFrontendRtx(
    "/models/Qwen3.6-35B-A3B",
    device="cuda:0",
    max_seq=4096,
    kernelized=True,
    quant_scope="experts",
)
frontend.set_prompt("Explain why deterministic reductions matter.")
token_ids = frontend.generate(max_new_tokens=64)
print(frontend.tokenizer.decode(token_ids))
```

`set_prompt()` accepts already-rendered text. For chat requests, render the
checkpoint's own template first:

```python
messages = [{"role": "user", "content": "Write a CUDA reduction checklist."}]
prompt = frontend.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
frontend.set_prompt(prompt)
```

The direct frontend is intentional. `flash_rt.load_model()` wraps VLA models
with a `predict(images, ...)` API and therefore redirects
`config="qwen36_moe"` to the class above.

## Checkpoint contract

Before allocating GPU memory, the frontend checks:

- the exact 40-layer `qwen3_5_moe` text geometry;
- the 30 linear-attention / 10 full-attention schedule;
- all 693 text-backbone tensors consumed by the shared pipeline;
- all 19 official MTP tensors;
- every safetensors shard referenced by the index.

Extra vision tensors are allowed and ignored by this text-only entry. The
validation can be run without loading model weights:

```bash
PYTHONPATH=. python - <<'PY'
from flash_rt.frontends.torch.qwen36_moe_rtx import (
    validate_qwen36_moe_checkpoint,
)
print(validate_qwen36_moe_checkpoint("/models/Qwen3.6-35B-A3B"))
PY
```

## Runtime controls

The shared architecture uses:

- `FLASHRT_QWEN35MOE_PREFILL_CHUNK` — chunked-prefill block size, default
  `8192`; `0` disables chunking.
- `FLASHRT_QWEN35MOE_GRAPH_CACHE_MAX` — decode CUDA Graph LRU capacity,
  default `256`.

The older `FLASHRT_NEXN2_PREFILL_CHUNK` and
`FLASHRT_NEXN2_GRAPH_CACHE_MAX` names remain compatible aliases.

## Validation

The repository smoke test is checkpoint-independent:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  pytest -q -p no:cacheprovider tests/test_qwen36_moe_smoke.py
```

Set `FLASHRT_QWEN36_MOE_CKPT_DIR` to include the official checkpoint contract
test. Performance and precision numbers must be measured on Qwen3.6 weights;
Nex-N2-mini measurements are not interchangeable even though the compute
pipeline is shared.

The optional real-model test loads the checkpoint, checks finite logits, and
compares cold and warm CUDA Graph generation with an official Transformers
BF16 greedy-token fixture:

```bash
FLASHRT_QWEN36_MOE_CKPT_DIR=/models/Qwen3.6-35B-A3B \
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
pytest -q -p no:cacheprovider tests/test_qwen36_moe_gpu.py
```

First-light data on an RTX 5090, PyTorch 2.9.1, CUDA 12.8 runtime, and the
official BF16 checkpoint:

| Measurement | Result |
|---|---:|
| Runtime weight load | 47.96 s |
| Resident allocated memory after load | 21.44 GiB |
| Peak allocated memory during load | 22.94 GiB |
| First 21-token prefill, including warmup | 230.95 ms |
| Subsequent 20–45-token prefill | 28.99–35.12 ms |
| 64-token prompt, 32-token eager decode | 48.14 tok/s |
| 64-token prompt, 32-token warm CUDA Graph decode | 195.49 tok/s |

The eager, first-capture, and warm-graph runs produced the same 32 token IDs.
These numbers are a first-light correctness run, not a context-length sweep.

Four chat prompts from 12 to 45 tokens were also compared with the official
Transformers BF16 implementation:

| Precision check | Result |
|---|---:|
| Last-token logit cosine, minimum | 0.95635 |
| Last-token logit cosine, mean | 0.96455 |
| First-token argmax matches | 4 / 4 |
| Greedy generation matches | 64 / 64 tokens |

The logit cosine is lower than the Nex-N2-mini measurement, but the tested
greedy sequences were token-exact for 16 generated tokens on all four prompts.

## Limitations

- Text only; the vision tower is not loaded.
- Greedy decode only on the kernelized path.
- The MTP tensors are validated but not loaded, so speculative decode is not
  enabled.
- Only the BF16 source checkpoint with runtime NVFP4 conversion is supported.
- SM120 only.
