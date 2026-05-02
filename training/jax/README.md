# FlashRT Training — JAX path (BETA)

Status: **v0.1.0 BETA — API not stable**. JAX companion to the
PyTorch FP8+LoRA training stack at [`training/`](../). Both paths
share the same algorithm primitives in
[`flash_rt/core/rl/`](../../flash_rt/core/rl/) (ACP tags, advantage,
soft VF loss) and the same FP8 cuBLASLt kernel in
[`csrc/gemm/gemm_runner.cu`](../../csrc/gemm/gemm_runner.cu); only
the framework wiring differs.

## Why this exists

FlashRT's PyTorch path is one of several ways to fine-tune pi0.5
([openpi](https://github.com/Physical-Intelligence/openpi) is the
canonical JAX reference, [lerobot](https://github.com/huggingface/lerobot)
is the PEFT-based PyTorch port). The JAX path here is for users
who *already* train in JAX via openpi and want to add FP8 forward
acceleration without leaving the JAX ecosystem. Concretely:

* The trained checkpoint is **Orbax**, not PyTorch safetensors.
* Inference loads the merged Orbax via
  [`flash_rt.frontends.jax.pi05_rtx`](../../flash_rt/frontends/jax/pi05_rtx.py)
  — JAX-native end to end, no torch in the train→serve loop.
* The PyTorch path at [`training/`](../) is a parallel, independent
  lane — they never need to meet.

The work the JAX path adds on top of openpi:

1. An **XLA FFI handler** that exposes FlashRT's cuBLASLt FP8 GEMM
   to JAX ([`csrc/training/jax_ffi/`](../../csrc/training/jax_ffi/)).
2. A **monkey-patch** that routes openpi's
   `lora.Einsum` + `lora.FeedForward` matmuls through the FFI when
   the layer carries a `LoRAConfig`.
3. A **JAX-native train→serve LoRA bridge** (Orbax in, Orbax out).
4. **Parity tests** against the PyTorch reference — the architecture
   port (StandaloneVF, ValueFunctionHead, Pi05ValueFunction) reaches
   cosine = 1.000000 under transferred weights; the ACP hook is
   byte-equal across stacks.

## Layout

```
training/jax/
├── fp8/
│   ├── ffi_loader.py          # ctypes-load + jax.ffi.register_ffi_target
│   ├── fp8_jax.py             # custom_vjp wrapping the FP8 FFI op
│   └── lora_patch.py          # patches openpi.models.lora.{Einsum,FeedForward}
├── rl/
│   ├── value_function.py      # StandaloneVF + ValueFunctionHead + Pi05VF (nnx)
│   ├── train_value.py         # synthetic-data VF training driver
│   ├── value_infer.py         # acp_indicator annotation pass
│   ├── acp_hook.py            # JaxACPPromptHook (byte-equal vs PyTorch)
│   └── train_recap.py         # namespace mirror — driver lives upstream
│                                 (train_libero_recap.py kept as back-compat alias)
├── checkpoint.py              # save/load lora_weights.npz + lora_metadata.json
├── merge_lora.py              # Orbax → Orbax LoRA fold (no PyTorch hop)
├── scripts/
│   └── run_baseline_with_fp8_patch.py   # subprocess wrapper: enable patch + runpy upstream
└── tests/                     # gitignored — parity / smoke / round-trip
```

The shared inference primitives (`acp_tags`, `cfg_sampler`,
`advantage`, numpy parts of `reward`) are imported directly from
[`flash_rt/core/rl/`](../../flash_rt/core/rl/); the JAX path only
reimplements the bits that touched torch tensors.

## Quick start

Same env-var convention as the PyTorch README. Add one JAX-only entry:

| Resource | Env var | Required by |
|---|---|---|
| pi0.5 JAX (Orbax) base ckpt | `FLASHVLA_PI05_CKPT_JAX` | upstream baseline + merge |
| LIBERO RECAP-annotated dataset | `FLASHVLA_RECAP_DATASET` | upstream baseline |
| openpi `train_jax_lora_recap.py` | `FLASHVLA_JAX_BASELINE_SCRIPT` | wrapper |
| Pi0.5 LIBERO ckpt (norm_stats source) | `FLASHVLA_PI05_LIBERO` | inference round-trip |

The trained driver is the upstream
`openpi-compiler/RL/scripts/train_jax_lora_recap.py`. FlashRT
contributes a wrapper that enables the FP8 patch before the
upstream's argparse runs:

```bash
python -m training.jax.scripts.run_baseline_with_fp8_patch \
    --baseline-script $FLASHVLA_JAX_BASELINE_SCRIPT \
    --checkpoint_path $FLASHVLA_PI05_CKPT_JAX \
    --dataset_root    $FLASHVLA_RECAP_DATASET \
    --output_dir      <your-run-dir> \
    --steps 30000 --batch_size 4 --lr 2.5e-5 \
    --lora_rank 16 --acp_dropout 0.30 --log_freq 50
```

Toggle the patch off with `FLASHVLA_JAX_FP8=0` (still installed,
routed through the original `jnp.einsum` / `jnp.dot`). The default
`FLASHVLA_JAX_FP8_MIN_M=64` keeps small-M shapes (action-expert
attention at small batch) on BF16 — see "Memory and throughput
notes" below for why.

`JAX_PLATFORMS=cuda` is required when the container ships the
experimental `mlir_tensorrt` plugin: its clustering pass rejects
XLA FFI custom calls. The wrapper does not set this for you so the
user can choose their backend explicitly.

### Train → serve (JAX-native, Orbax → Orbax)

After a training run, the upstream script saves both a full Orbax
state and a LoRA-only `lora_weights.npz`. To produce a clean Orbax
that the JAX inference frontend loads directly:

```python
from training.jax.merge_lora import merge_lora_into_base

merge_lora_into_base(
    trained_orbax_dir="<your-run-dir>/step_030000/params",
    output_orbax_dir="<your-merged-dir>/params",
    scaling=1.0,           # openpi pi05 default (alpha == rank)
    overwrite=False,
)
```

The merged directory is a drop-in replacement for `pi05_base/params`
plus the LIBERO finetune. Load via:

```python
from flash_rt.frontends.jax.pi05_rtx import Pi05JaxFrontendRtx

pipe = Pi05JaxFrontendRtx("<your-merged-dir>", num_views=2)
pipe.set_prompt("pick up the red block")
pipe.calibrate_with_real_data([sample_obs])
out = pipe.infer(obs)             # {"actions": (chunk_size, 7) np.ndarray}
```

Note that the inference frontend's
[`flash_rt/frontends/jax/pi05_rtx.py`](../../flash_rt/frontends/jax/pi05_rtx.py)
loader expects norm_stats next to the merged checkpoint at
`<dir>/assets/physical-intelligence/libero/norm_stats.json`. The
upstream pi05_libero ckpt already ships this file — symlink its
`assets/` next to your merged dir.

There is **no PyTorch in the train→serve path**. Users on the
PyTorch line have a separate, parallel lane at
[`training/rl/merge_lora.py`](../rl/merge_lora.py) that produces a
`model.safetensors` for `Pi05TorchFrontendRtx`. The two stacks
never need to meet.

For **classifier-free-guidance inference** on the merged JAX
checkpoint (the RECAP / ACP test-time recipe), see
[`docs/rl_inference.md`](../../docs/rl_inference.md).
`Pi05JaxFrontendRtx` inherits the CFG pipeline machinery from the
PyTorch frontend, so `set_rl_mode(cfg_enable=True, cfg_beta=1.5)`
+ `set_prompt(...)` + `infer(obs)` works on a JAX-loaded
checkpoint without any extra wiring (verified at ~37 ms / call
on RTX 5090, β=1.5 — same path as the PyTorch frontend).

### What is the RECAP / ACP pipeline?

Same recipe as the PyTorch README's
["What is the RECAP / ACP pipeline?"](../README.md#what-is-the-recap--acp-pipeline)
section. The shared algorithm primitives live under
[`flash_rt/core/rl/`](../../flash_rt/core/rl/); both stacks import
the same `build_acp_tagged_task` from `acp_tags.py`, the same
N-step advantage from `advantage.py`, and the same numpy reward
helpers from `reward.py`. The byte-equality test in
`training/jax/tests/test_acp_hook_parity.py` (gitignored) confirms
the JAX hook produces identical tag strings given the same
`(tasks, indicators, seed)` triple.

`Pi05ValueFunction` (the paper §IV-A 3-layer head over a frozen
pi0.5 prefix embedding) is implemented in
[`training/jax/rl/value_function.py`](rl/value_function.py) as a
`flax.nnx.Module` mirroring the PyTorch reference at
[`flash_rt/core/rl/value_function.py`](../../flash_rt/core/rl/value_function.py).
Forward parity at cosine = 1.000000 under transferred weights —
see Status table below.

## Status

| Component | State | Notes |
| --- | --- | --- |
| Per-shape FP8 forward correctness | ✅ | cos ≥ 0.997 on all six pi0.5 GEMM shapes (matches PyTorch parity floor) |
| LoRA grad correctness via FP8 forward | ✅ | grad_x cos ≥ 0.997 vs all-BF16 reference |
| openpi `lora.Einsum` patch | ✅ | cos ≥ 0.997 on `qkv_fused`, `q_only`, `kv_fused`, `attn_vec` einsum patterns |
| openpi `lora.FeedForward._dot` patch | ✅ | cos ≥ 0.997 on FFN gate / up / down via real `FeedForward` module |
| Threshold gate (`FLASHVLA_JAX_FP8_MIN_M`) | ✅ | small-M shapes fall back to `jnp.einsum` cleanly |
| Routing fraction on real pi0.5 forward | ✅ | 24/24 = 100 % at B = 8 (gate ≥ 70 %) |
| 50-step JAX RECAP smoke (FP8 on vs off) | ✅ | final-loss ratio 1.00× (gate < 5×), no NaN |
| 100-step gate (peak GPU < 24 GB) | ✅ | 21.04 GB measured with `XLA_PYTHON_CLIENT_ALLOCATOR=platform` |
| 30k stability run | ✅ | 166.75 min, peak 21.06 GB, loss 0.123 → 0.013, 0 NaN |
| Architecture parity (`StandaloneVF`, `ValueFunctionHead`, `Pi05VF`) | ✅ | logits + predict_value cos = 1.000000 under transferred weights |
| Indicator annotation parity | ✅ | Hamming distance 2 / 12962 = 0.015 % vs PyTorch (gate ≤ 1 %) |
| ACP hook byte-equality vs PyTorch | ✅ | 23/23 across 5 seeds × 4 dropouts |
| JAX-native train→serve round-trip | ✅ | merge → `Pi05JaxFrontendRtx.infer` finite, abs_mean 0.288, latency 19.15 ms |
| FP8 backward (E5M2) | 🚫 intentionally not implemented | matches PyTorch decision; same reasoning as [training/README.md](../README.md#why-fp8-backward-is-not-implemented) |

## Per-shape FP8 forward microbench (RTX 5090 SM120, M = 512)

Standalone matmul-vs-matmul comparison against the JAX
`jnp.einsum` BF16 baseline. JIT'd, 50-iteration mean. **These do
NOT translate to end-to-end training-step speedup** — the
end-to-end number is in the next section, where XLA scan-fusion
loss eats most of the per-shape win at small batch.

| Shape | dims | fwd-only | fwd+bwd |
|---|---|---|---|
| `enc.attn.qkv` | 2048 → 2048 | 1.92× | 1.83× |
| **`enc.ffn.gate_up`** | 2048 → 16384 | **3.74×** | **2.91×** |
| `enc.ffn.down` | 16384 → 2048 | 3.31× | 2.31× |
| `dec.attn.qkv` | 1024 → 1024 | 2.42× | 1.97× |
| `dec.ffn.gate_up` | 1024 → 4096 | 1.69× | 1.05× |
| `dec.ffn.down` | 4096 → 1024 | 2.57× | 1.16× |

Phase-1 gate (≥ 1.3× fwd on the FFN gate_up shape, the largest
LoRA-bearing matmul) is met at 3.74×. The FFN-down + decoder-attn
rows hover around 1× on fwd+bwd because the activation re-quantize
on the backward path costs roughly what the FP8 GEMM saves on
those small-N shapes.

## End-to-end training-step throughput (RTX 5090 SM120, B = 4, lora_rank = 16)

Single 30k LIBERO RECAP run with the FP8 patch enabled:

| Stack | step/s | samples/s | peak GPU | wall (30k) | loss first / last |
|---|---:|---:|---:|---:|---:|
| **JAX FP8 + LoRA via this path** | **3.00** | **11.99** | **21.06 GB** | **166.75 min** | 0.123 → **0.013** |

Run config: B=4, `lr=2.5e-5`, `lora_rank=16`, `acp_dropout=0.30`,
`seed=42`, scaling factor 1.0 (alpha == rank, openpi pi05 default).
Throughput is steady — first-50-step warmup ≈ 1.0 step/s, then 3.0
step/s held flat for the remaining 29950 steps. No NaN over 30k
steps. Peak GPU is the actual working set under
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` (without that flag JAX's
default 75% preallocation makes `nvidia-smi` report ~24 GB instead).

For reference, the PyTorch path's 30k stability data
([training/README.md](../README.md#end-to-end-training-step-throughput-rtx-5090-sm120-b4-lora_rank16))
on the same hardware reports `step/s = 3.95`, `peak GPU = 14.65 GB`,
`wall ≈ 128 min`, `loss mean 0.044 → 0.012`. The two stacks
converge to the same loss band; the throughput / memory difference
between them reflects framework + allocator overhead, not the
FP8 kernel itself.

### Loss trajectory (FP8 on, 30k steps)

| Step | loss | comment |
|---:|---:|---|
| 0 | 0.1229 | initial — warmup LR ramp |
| 1000 | 0.0356 | warmup-cosine peak (lr = 2.49e-5) |
| 5000 | 0.0220 | well below initial; LR decaying |
| 10000 | 0.0173 | midway plateau |
| 15000 | 0.0175 | flat band |
| 20000 | 0.0145 | minor convergence |
| 25000 | 0.0147 | LR near floor (lr ≈ 5e-6) |
| 30000 | 0.0129 | final — loss converged to floor |

Last-half median loss is 0.0135. The shape matches the PyTorch
RECAP curve 1-for-1 in direction; absolute values differ by a few
tenths of a milli-loss because the JAX path runs on the openpi
optimizer + dataset pipeline rather than the FlashRT PyTorch one.

## Memory and throughput notes

The JAX path's measured peak (21.06 GB at B=4) is structurally
larger than the PyTorch path's (14.65 GB) for two reasons,
neither involving the FP8 kernel:

1. **Allocator behaviour.** PyTorch's caching allocator releases
   intermediates as soon as ref-count hits zero; XLA's pool keeps
   buffers alive across the JIT step boundary so the high-water
   mark accumulates more than one step's worth of activations.
2. **`nnx.state(model)` shape.** The optimizer's `optax.adamw`
   tracks two moments per LoRA parameter, materialised as a
   separate JAX param tree; PyTorch's optimizer stores moments
   in-place on the param tensor.

`XLA_PYTHON_CLIENT_PREALLOCATE=false` +
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` are required for an honest
peak measurement (otherwise JAX reserves 75 % of the GPU upfront
and `nvidia-smi` reports the reservation, not the working set).
These two env vars are baked into all the gate tests under
`training/jax/tests/`.

The throughput at B=4 (3.0 step/s vs PyTorch 3.95) reflects two
costs the JAX path pays that PyTorch does not:

* Per-GEMM **FFI call overhead.** Every patched matmul is a
  Python → C++ FFI round-trip plus a cuBLASLt descriptor lookup.
  At small batch the overhead is non-trivial vs the FP8 GEMM
  savings.
* **XLA scan-fusion break.** openpi stacks the 18 transformer
  layers via `flax.linen.scan`, which XLA fuses into a single
  big kernel. Per-GEMM FFI breaks that fusion. The
  [`flashvla-ft/`](../../flashvla-ft/) prototype documented this
  effect at length — its measured speedup grew with batch size
  (`0.83×` at B=4 → `2.03×` at B=32). The same shape is expected
  here. We have not benchmarked B=32 because B=4 is the recipe
  the upstream `train_jax_lora_recap.py` ships.

The `FLASHVLA_JAX_FP8_MIN_M` threshold (default 64) is the
mitigation: shapes with `M < 64` (action-expert attention at
small B) fall back to the original `jnp.einsum`, where XLA's
fusion stays intact and the FFI overhead would otherwise be
larger than the FP8 win.

### Why FP8 backward is not implemented

Same reasoning as the PyTorch path — see
[training/README.md](../README.md#why-fp8-backward-is-not-implemented).
Memory savings are already covered by the cached BF16 mirror in
the `custom_vjp`'s residuals (or by a future
on-the-fly dequant variant — same trade-off the PyTorch
`cache_bf16_weight=False` recipe makes). Speed envelope at B=4
is small for the same scan-fusion reason as the forward.
Numerical risk on flow-matching MSE gradients in FP8 E5M2 is real
and would need a separate study. Re-evaluate at larger batch.

## Tests

`training/jax/tests/` is `.gitignore`d (per-developer local
validation, requires LIBERO RECAP + pi0.5 ckpts). Set the
`FLASHVLA_*` env vars from the [PyTorch README](../README.md#quick-start)
plus `FLASHVLA_JAX_BASELINE_SCRIPT` (path to upstream
`train_jax_lora_recap.py`), then run from the repo root::

    JAX_PLATFORMS=cuda PYTHONPATH=.:<openpi-src-path> \
        python3 -m pytest training/jax/tests/ -v

Fast suite (≈ 70 seconds): 55 parametric cases across parity,
patch routing, ACP byte-equality. Slow suite (≈ 5 + 3 + 167
minutes for 50-step / 100-step / 30k gates): set the `FLASHVLA_*`
env vars first.

`JAX_PLATFORMS=cuda` is required when the JAX install ships an
experimental `mlir_tensorrt` plugin that grabs the CUDA platform
on import — the plugin's clustering pass rejects FlashRT's XLA
FFI custom calls, so an explicit platform pin keeps the run on
the canonical CUDA backend.
