# Qwen3.6-27B DFlash Speculative Decoding

This document covers the DFlash block-diffusion drafter path for
Qwen3.6-27B NVFP4. DFlash replaces the sequential MTP draft chain with
a single drafter forward per speculation cycle: a 5-layer 2B drafter
proposes an entire 15-token block, and the target model verifies the
block in one S=16 forward.

For the general Qwen3.6 NVFP4 model contract and parameter reference,
see [`qwen36_nvfp4.md`](qwen36_nvfp4.md) and
[`qwen36_usage.md`](qwen36_usage.md).

## Requirements

- Qwen3.6-27B NVFP4 main checkpoint (same as the MTP path).
- The z-lab DFlash drafter checkpoint:

```bash
hf download z-lab/Qwen3.6-27B-DFlash --local-dir /models/Qwen3.6-27B-DFlash
```

  The drafter ships as a single BF16 `model.safetensors` (~3.3 GB,
  5 layers, `block_size=16`, target hidden taps at layers
  1/16/31/46/61). FlashRT quantizes every drafter linear to NVFP4 at
  load time (~825 MB resident); no separate conversion step.
- On Thor the DFlash verify runs over the persistent FP8 KV cache.
  The frontend allocates it automatically at drafter load if the
  construction did not already enable long-context mode.

## Usage

```python
import os

from flash_rt.frontends.torch.qwen36_thor import Qwen36TorchFrontendThor

os.environ["FLASHRT_QWEN36_MTP_CKPT_DIR"] = "/models/Qwen3.6-27B-FP8"
os.environ["FLASHRT_QWEN36_DFLASH_CKPT_DIR"] = "/models/Qwen3.6-27B-DFlash"
os.environ["FLASHRT_QWEN36_LONG_KV_CACHE"] = "fp8"

fe = Qwen36TorchFrontendThor(
    "/models/Qwen3.6-27B-NVFP4",
    quant="nvfp4",
    max_seq=32768,
)
fe.init_dflash_drafter()          # reads FLASHRT_QWEN36_DFLASH_CKPT_DIR

ids = fe._tokenizer.apply_chat_template(
    [{"role": "user", "content": "Plan the pick-and-place task."}],
    add_generation_prompt=True, return_tensors="pt").to(fe.device)

out = fe.generate_own_speculative_DFlash_nvfp4(
    ids,
    max_new_tokens=256,
    K=15,                          # speculative tokens per cycle
)
```

The RTX frontend exposes the same entry point; the drafter and verify
kernels are shared, only the KV plumbing differs per arch.

## Drafter context window

The drafter conditions on fc-projected target hidden features of the
committed context. Two window modes exist:

- **Per-token window** (Thor default): one feature entry per committed
  token, appended in bulk after each verify (N+1 entries per cycle).
  On Thor the prompt prefill seeds the window with the features of the
  last `min(window, prompt_len)` prompt tokens, so the drafter starts
  at full context instead of ramping from empty.
- **Per-cycle shift window** (legacy, RTX default): one entry per
  speculation cycle. Kept for compatibility; acceptance length is
  measurably lower because window entries end up ~AL tokens apart.

| Env | Default | Meaning |
|---|---|---|
| `FLASHRT_QWEN36_DFLASH_CKPT_DIR` | unset | Drafter checkpoint directory (required). |
| `FLASHRT_QWEN36_DFLASH_PERTOKEN` | `1` on Thor | Per-token window mode. |
| `FLASHRT_QWEN36_DFLASH_WINDOW` | `128` | Per-token window length (tokens, <= 256). |
| `FLASHRT_QWEN36_DFLASH_WINDOW_SEED` | `1` | Seed the window from the prompt tail at prefill (Thor). |

## Measured performance (Thor, SM110)

Steady-state decode at short context against the FP8-KV MTP spec path
(`generate_own_speculative_KN_nvfp4`, K=6) in the same process, greedy
decoding, 64/256-token delta method:

| prompt | MTP AL / tok/s | DFlash AL / tok/s |
|---|---:|---:|
| robot task -> JSON plan | 2.87 / 33.7 | **4.57 / 48.9** |
| robot navigation plan | 2.59 / 30.5 | 3.25 / 34.8 |
| prose explanation | 2.43 / 28.5 | 3.00 / 31.7 |

Cycle anatomy on Thor: one S=16 verify (~86 ms, weight-read bound) +
one drafter graph replay (~7 ms). A partial accept costs two
constant-time state copies from the per-step checkpoints written
during the verify itself — there is no recovery forward. The accept
decision includes one host synchronization per cycle
(`argmin().item()` on the match mask); at ~10 us it is three orders
of magnitude below the verify cost and is included in every number
above. A device-side accept loop is possible follow-up work, not a
prerequisite.

Output quality is lossless: the verify pass is the greedy ground
truth, and generated tokens are byte-identical to the FP8-KV MTP
reference on all measured prompts.

## Serving

A stateless OpenAI-compatible host for this path lives in
[`serving/qwen36_dflash_agent`](../serving/qwen36_dflash_agent) —
single-stream request/response serving with per-request DFlash
generation and accept-length telemetry:

```bash
python -m serving.qwen36_dflash_agent.server \
  --checkpoint /models/Qwen3.6-27B-NVFP4 --max-seq 32768 --K 15
curl -s http://127.0.0.1:8000/health
```

For long-running agent sessions (prefix reuse, tool calling, SSE
streaming) use [`serving/qwen36_agent`](../serving/qwen36_agent),
which serves the MTP spec path.

## Notes

- Structured output (JSON plans, code) accepts much better than free
  prose; the gains above track the drafter's training distribution.
- Degenerate prompts that repeat one sentence verbatim can steer the
  seeded window into drafting more repetition. If you benchmark with
  synthetic repeated text, disable the seed
  (`FLASHRT_QWEN36_DFLASH_WINDOW_SEED=0`) for representative numbers.
- Greedy-parity comparisons must use the FP8-KV MTP route as the
  reference (`FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ=0` forces it for
  short prompts). The BF16 short route stores KV in a different
  format, so token-exact comparison across the two is not meaningful.
- The published drafter checkpoint is marked by z-lab as still under
  training; acceptance lengths should improve by dropping in a newer
  drafter checkpoint without code changes.
