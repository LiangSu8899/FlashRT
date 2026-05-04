# Qwen3.6-27B NVFP4 on RTX 5090

This document covers the FlashRT NVFP4 inference path for Qwen3.6-27B,
including model dependencies, the speculative-decode `K` selection, real
measured throughput data, and reproduction commands. Numbers come from
the v1 release on `feat/qwen36-integration` (commit `1d4e102`),
RTX 5090 (sm_120, 32 GB HBM, BW 1.79 TB/s).

## 1. Model dependencies

The NVFP4 inference path needs **two** checkpoints:

| Role | Format | Source |
|---|---|---|
| Main model | NVFP4 W4A16 (`compressed-tensors` `nvfp4-pack-quantized`) | [`prithivMLmods/Qwen3.6-27B-NVFP4`](https://huggingface.co/prithivMLmods/Qwen3.6-27B-NVFP4) |
| MTP head | FP8 e4m3 block-128 (only `mtp.safetensors` is consumed) | the upstream FP8 ckpt (Qwen3.6-Next-27B-FP8) |

Pass the main NVFP4 ckpt directory as the `checkpoint_path` argument
to `Qwen36TorchFrontendRtx`. The NVFP4 ckpt does **not** ship an MTP
module — `compressed-tensors` strips it — so the FP8 ckpt directory
that contains `mtp.safetensors` is loaded separately via the
`FLASHRT_QWEN36_MTP_CKPT_DIR` environment variable; we convert
FP8 → BF16 → NVFP4 once at load (no FP8 in the hot path). Without
the env var, MTP is None and speculative decode is disabled (pure
single-token decode still works at ~36 tok/s).

### Can I use a different NVFP4 Qwen3.6 checkpoint?

Yes, **as long as** all four conditions hold:

1. **Architecture** = Qwen3.6-Next 27B (`num_hidden_layers=64`,
   `num_v_heads=48`, `hidden_size=5120`, head_dim=128, vocab 248,320).
   The frontend hard-codes these for buffer allocation.
2. **Quantization** = `compressed-tensors` `nvfp4-pack-quantized` with
   the standard schema:
   - `<prefix>.weight_packed` u8 `(out, in/2)` (FP4 e2m1, 2-per-byte)
   - `<prefix>.weight_scale` fp8_e4m3 `(out, in/16)` (per-block-16 SF)
   - `<prefix>.weight_global_scale` fp32 scalar (= `448 / amax`)
3. **Quant scope** = MLP gate/up/down + full-attn q/k/v/o.
   Linear-attn projections (`in_proj_qkv`, `in_proj_z`, `out_proj`,
   `in_proj_a`, `in_proj_b`, `conv1d`) **must stay BF16** — that's
   what the lin-attn kernels expect.
4. **MTP head from a paired FP8 ckpt**. Without it, speculative decode
   is unavailable; pure-decode (no spec) still works at ~36 tok/s.

If any of those is different (e.g. AWQ, GGUF, GPTQ, MXFP4, full-tensor
NVFP4 of lin-attn), the loader will reject or produce wrong outputs.

The loader source of truth is [`flash_rt/frontends/torch/_qwen36_rtx_nvfp4_weights.py`](../flash_rt/frontends/torch/_qwen36_rtx_nvfp4_weights.py)
— see the module docstring for the exact key list.

## 2. Headline numbers (decode tok/s)

Decode tok/s = `(N_OUT - 1) × 1000 / decode_time_ms`. **Excludes**
prefill (TTFT). Same metric vLLM and TensorRT-LLM report.

Single representative prompt (`"Explain quantum entanglement in one
short paragraph."`, 11 tokens), max_new_tokens=128, default `K=6`:

```
TTFT (prefill)        :   ~237 ms      (one-shot, doesn't recur)
TPOT                  :   ~7.74 ms/token
★ decode tok/s        :   128.87       ← v1 headline
end-to-end tok/s      :   ~104         (with prefill amortized)

spec stats: K=6  attempts=31  p_full=0.290  p_ind=0.522  AL=4.10
```

`AL=4.10` means each spec cycle emits 4.1 tokens on average; `p_full`
is the fraction of cycles where the full draft chain is accepted.

Pre-session baseline (post-α-S3, K=3 default) on the same prompt:
**92.10 tok/s**. v1 release improvement: **+39.9%**.

## 3. Choosing `K` (speculative chain length)

`K` is the MTP draft chain length per spec cycle. Verify processes
`K+1` tokens, the spec loop accepts the longest matching prefix. The
right `K` depends on prompt distribution and target output length.

### Measured K-curve (single prompt, NTOK=128)

```
 K   decode tok/s   AL    p_ind   vs original 92.10
 3   119.23         3.17  0.733   +29.5%
 4   112.68         3.17  0.556   +22.4%   (drafter trough)
 5   124.15         3.74  0.559   +34.8%
 6   129.44         4.10  0.522   +40.5%   ★ peak (NTOK=128)
 7   119.16         3.97  0.429   +29.4%   (rolls off)
```

Why `K=6` wins at NTOK=128: AL keeps growing through `K=6` faster than
verify cost grows; at `K=7` drafter `p_ind` drops far enough that AL
plateaus while verify cost dominates. `K=4` is a local minimum because
`p_ind` crashes from 0.733 to 0.556 at the new "deepest" position
without enough total AL gain to offset.

### Length sensitivity

Drafter quality decays as generation goes on (drift from the original
prompt distribution). Longer outputs → smaller `K` becomes safer.

```
                    NTOK=128   NTOK=256   NTOK=512
 K=3  decode tok/s  119.23     113.66     113.98
 K=5  decode tok/s  124.15     117.37     114.40
 K=6  decode tok/s  129.44     109.68     112.34   ← peak shifts
 K=7  decode tok/s  119.16     107.28     110.65
```

Peak shifts from `K=6` (NTOK=128) to `K=5` (NTOK=256), and by NTOK=512
all values converge around 113. For workloads with mostly short outputs
(< 256 tokens), `K=6` is best; for sustained generation, `K=5` is more
robust.

### Per-prompt variance (5 prompts × 2 NTOK)

Speculative decode is sensitive to prompt-text distribution. The
drafter aligns better with structured prompts (math, code) than
free-form ones (creative writing).

```
    prompt  NTOK   K   prompt_len   decode tok/s     AL  p_full
   explain   128   3      11             119.11   3.17   0.575
   two_sum   128   3      41             110.28   2.95   0.558
      sort   128   3      22             115.85   3.10   0.512
      math   128   3      17             122.01   3.26   0.538
   summary   128   3      19              94.90   2.54   0.220
   explain   128   6      11             128.87   4.10   0.290
   two_sum   128   6      41             110.69   3.53   0.139
      sort   128   6      22             102.29   3.26   0.051
      math   128   6      17             117.32   3.74   0.000
   summary   128   6      19              83.15   2.65   0.000
   explain   256   3      11             113.41   3.04   0.512
   two_sum   256   3      41             119.08   3.19   0.613
      sort   256   3      22             123.82   3.31   0.623
      math   256   3      17             130.58   3.49   0.671
   summary   256   3      19             103.49   2.77   0.348
   explain   256   6      11             109.48   3.49   0.123
   two_sum   256   6      41             121.05   3.86   0.152
      sort   256   6      22             115.83   3.70   0.101
      math   256   6      17             131.09   4.18   0.098
   summary   256   6      19              96.28   3.07   0.060
```

Aggregate (mean ± CV across 5 prompts):

| NTOK | K | min | median | max | mean | CV |
|---|---|---:|---:|---:|---:|---:|
| 128 | 3 | 94.90 | 115.85 | 122.01 | 112.43 | 9.5% |
| 256 | 3 | 103.49 | 119.08 | 130.58 | 118.07 | 8.7% |
| 128 | 6 | 83.15 | 110.69 | 128.87 | 108.46 | 15.8% |
| 256 | 6 | 96.28 | 115.83 | 131.09 | 114.74 | 11.3% |

**Reading the table:**
- `K=6` peak (131.09 on math/256) > `K=3` peak (130.58) — captures the
  best case, which is the headline-worthy number.
- `K=6` mean ≈ `K=3` mean across diverse prompts. The expected speedup
  for "any prompt" is roughly even.
- `K=6` is **more sensitive** to prompt distribution (CV 11-16% vs
  K=3's 9%). Predictable workloads (single prompt class) favor K=6;
  varied workloads favor K=3.
- "summary"-style creative prompts are 1.6× slower than math/code
  prompts at the same K. This is normal speculative-decoding
  variance — not a bug.

### Recommendation

| Workload | Suggested K |
|---|:---:|
| Mostly short generations (≤ 256 tokens), single prompt class | **6** (default) |
| Mixed workloads, longer generations | **5** |
| Ultra-conservative (tightest variance) | **3** |

Set via `TEST_K=<n>` env var when running `standard_bench`, or pass
`K=<n>` to `generate_own_speculative_KN_nvfp4`.

## 4. Long-context throughput

Decode tok/s at fixed context length (synthetic-filled KV cache, no
spec — single-token forward then reported as if spec amortization
applied at AL=3.17). Uses the TurboQuant packed KV cache.

```
   ctx     forward ms    decode tok/s  (eager)    CUDA-Graph replay
  8 K        26.6              119.3                36.0 ms /  88.6
 16 K        30.4              105.4                ─    (capture only at 32 K+)
 32 K        38.7               81.3                35.7 ms /  88.7
 64 K        55.5               57.1                51.8 ms /  61.2
128 K        87.7               36.2                85.3 ms /  37.1
200 K       120.5               26.3                ─
256 K       153.4               20.7               150.8 ms /  21.0
```

Replay cos vs eager = **1.000000** at every ctx (32 K / 64 K / 128 K /
256 K) — bit-identical token output across replays.

Spec decode (K=3..6) is **not yet integrated with the long-ctx
TurboQuant path** — that's a Phase 3D follow-up. The numbers above are
single-token forward throughput at long ctx, projected to AL=3.17.

## 5. TTFT (prefill latency)

Prefill is one S=1 forward per prompt token, captured per-cur_pos as
CUDA Graphs. Cost scales linearly:

```
prompt_len   TTFT (ms)       per-token (ms)
        11   ~237            21.5
        17   ~370            21.8
        22   ~480            21.8
        41   ~890            21.7
       100   ~2200           22.0
```

≈ 22 ms / token of prefill, prompt-length-independent rate. For a 128
prompt-token call, TTFT ≈ 2.8 s. The TPOT (decode) measured from the
spec loop is independent of TTFT and dominates total wall time when
output ≫ prompt.

## 6. Reproduction

Build (from the FlashRT repo root):

```bash
cmake -S . -B build
cmake --build build -j --target flash_rt_kernels
# flash_rt_kernels*.so lands in flash_rt/ via CMake's
# LIBRARY_OUTPUT_DIRECTORY — no manual cp needed.
```

Run (replace `<NVFP4_CKPT>` with the `prithivMLmods/Qwen3.6-27B-NVFP4`
directory and `<FP8_CKPT>` with the directory that contains the FP8
ckpt's `mtp.safetensors`):

```python
import torch
from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

import os
os.environ['FLASHRT_QWEN36_MTP_CKPT_DIR'] = '<FP8_CKPT>'
fe = Qwen36TorchFrontendRtx('<NVFP4_CKPT>', quant='nvfp4')

prompt = 'Explain quantum entanglement in one short paragraph.'
input_ids = fe._tokenizer(prompt, return_tensors='pt').input_ids.cuda()
out = fe.generate_own_speculative_KN_nvfp4(
    input_ids, max_new_tokens=128, K=6)
print(fe._tokenizer.decode(out[0, input_ids.shape[1]:].tolist()))
```

Recommended runtime env vars:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

`internal-tests/` is gitignored — these probe scripts ship as part of
the v1 release for reproduction but are dev-local.

## 7. Optimization history (this release vs prior)

```
phase                                            decode tok/s  Δ
α-S3 baseline (CUTLASS EVT dequant, K=3)            92.10      —
A2c-2 fuse in_proj_a/b BF16 matvec                  92.40      +0.3%
A1'-S0 per-step lin state save + skip recovery     117.62      +27.7%
A2c-3 chained per-step state via in/out kernels    119.24      +29.5%
A1'-S1 K_save_max=8 + spec K=6 default             128.87      +39.9%
```

The biggest single lever was eliminating the partial-accept recovery
forward (A1'-S0): the spec loop's `restore + recovery forward` path
fired on ~43% of cycles and cost ~21 ms each. By saving the lin/conv
state at every step **inside** the verify K-iter recurrent loop, the
spec loop reads the correct state directly on partial accept — no
recovery forward needed. This collapsed average cycle time from 34.4
ms to 27.0 ms.

Detailed per-commit notes are in the `git log` for commits `ea2be34`,
`1825d93`, `372b37c`, `a1e59e1`, `1d4e102` on `feat/qwen36-integration`.
