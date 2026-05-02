# Kernel Library and Fusion Patterns

> **Target audience**: engineers who need to pick kernels while writing a new model's `pipeline.py` forward, optimize an existing forward path, or evaluate whether a proposed fusion is viable.
>
> **TL;DR**:
> - The library ships **~120 pybind entries across three modules** — `flash_rt_kernels` (98 memory-bound / fp8 / cuBLAS-wrapper / FMHA entries), `flash_rt_fp4` (23 NVFP4 entries), and `flash_rt_fa2` (2 FA2 forward entries). Counts can drift ±a handful between releases; treat the number as "about 100", not exact. `flash_rt_kernels` is organized into 10 groups by purpose (see §2 below).
> - Fusion is not free: **any rewrite that changes the kernel launch sequence carries Myelin tactic risk**. All 4 historical regressions (see §5) were caused by this.
> - Current production fusion shape: see [`optimization-details.md`](../optimization-details.md) §1.1; this doc complements it with **how to choose / whether a new model can reuse** the existing kernels.
> - Adding a brand-new fused kernel requires editing `csrc/` and rebuilding the `.so` (out of scope for this round; separate CUDA dev workflow).

**Read first**: [`docs/optimization-details.md`](../optimization-details.md) has the full Pi0.5 optimization accounting (where the 44ms vs 70ms gap comes from). This doc is not about "why Pi0.5 is fast" — it's about "**how you wire kernels together in forward when adding a new model**".

---

## 1. Kernel naming conventions

### 1.0 Two naming layers: C++ math-primitive vs Python semantic-role

FlashRT uses **two distinct naming conventions** for the same kernel depending on layer:

- **C++ layer** (`csrc/`, symbol names, file names): named after the **math primitive** the kernel implements, e.g. `flash_rt::fused_fp4::silu_mul_two_fp4_to_fp4` — "apply `silu(a) * b` to two fp4 streams, output fp4". The kernel does not know (or care) what a caller uses it for.
- **Python layer** (`flash_rt_kernels` / `flash_rt_fp4` pybind modules, docs, user-facing API): named after the **semantic role** in the model architecture, e.g. `gate_geglu_merged_fp4`. Different architectures can bind the same C++ primitive under different Python names when the semantics differ.

Concrete example: Pi0.5 / Pi0 use **GEGLU** (tanh-approx GELU gate) in the encoder FFN. The historical C++ name `silu_mul_two_*` reflects the shared math skeleton (`activation(a) * b`), and the Python binding is exposed as `gate_geglu_*` to reflect the actual activation the model expects. For a true-SiLU architecture (Qwen3, BAGEL) the Python binding is `silu_mul_split_fp8_fp16` — a different Python entry, sometimes the same C++ primitive.

If you are reading C++ source and see `silu_mul_*` names, you are looking at the math-primitive layer. The Python-visible name is the contract; the C++ name is an implementation detail. **Do not rename C++ symbols to match Python** — downstream builds, profiling traces, and historical git history all reference the C++ names.

This split mirrors the PyTorch / CUTLASS convention (math-named kernels, semantic-named functional wrappers) and keeps the kernel library reusable across architectures.

### 1.1 Suffix/infix cheatsheet

**You can read most of what a kernel does straight from its name**, no need to grep the source:

| Suffix / infix | Meaning |
|----------------|---------|
| `_fp16` | fp16 in / fp16 out |
| `_fp8` | fp8 input |
| `_fp8_fp16` | fp8 in, fp16 out (where the descale sits decides which end) |
| `_bf16out` / `_f32out` | bf16 / fp32 output (default is fp16) |
| `_static` | activation scale is a compile-time constant (passed in as a `d_scale` pointer) |
| `_noweight` | RMSNorm does not multiply by the `(1 + learnable_weight)` vector (Pi0.5 / GROOT AdaRMSNorm use "1" as the weight and a separate Dense layer for modulation) |
| `_inplace` | in-place update (input buffer == output buffer) |
| `_merged` | several primitives collapsed into a single kernel (classic: gate_geglu) |
| `_wide` | CUTLASS GEMM tactic tuned for a "wide" output dim (used for Gate+Up of shape [2H, D]) |
| `_sq` | "square-ish" CUTLASS GEMM, balanced shape |
| `_t1` | CUTLASS GEMM tactic for "tall" shapes (high first dim) |
| `_plain` | CUTLASS GEMM without descale, raw fp32 accumulate output (rarely used) |

Examples:
- `residual_add_rms_norm_fp8_noweight_fp16` = residual add + RMSNorm (no weight) + quantize to FP8, fp16 input → three operations in one kernel.
- `gate_geglu_merged_fp8_fp16` = Gate+Up where the gate output is fp16; after GELU+Mul the result is quantized to FP8 (ready for the next GEMM).

---

## 2. Kernel inventory, grouped by purpose

### 2.1 GEMM family (8 variants)

FP8 in → fp16 / bf16 / f32 out, per-tensor static descale:

```
cutlass_fp8_plain         A: f32 out, no descale
cutlass_fp8_sq            A: fp16 out, "square-ish" tactic
cutlass_fp8_sq_bf16out    A: bf16 out
cutlass_fp8_sq_f32out     A: fp32 out
cutlass_fp8_t1            A: "tall" (high M) tactic
cutlass_fp8_t1_bf16out    A: bf16 out
cutlass_fp8_wide          A: "wide" (high N, e.g. Gate+Up [2H, D]) tactic
cutlass_fp8_wide_bf16out  A: bf16 out
cutlass_fp8_wide_f32out   A: fp32 out
cutlass_fp8_gelu          A: fp16 out + GELU epilogue (only used in the DiT FFN GELU path)
```

**Which one to pick**:
- Square-ish (M ≈ N ≈ K, e.g. the O projection) → `_sq`
- Tall and skinny (N = 2H >> M, e.g. Gate+Up) → `_wide`
- High M (decoder running multiple steps at once) → `_t1`

Descale parameter: alpha = act_scale × weight_scale (see [`calibration.md §2.3`](calibration.md)).

**JAX consumer (training).** The `cublasLtMatmul` FP8 path used by
PyTorch (`GemmRunner::fp8_nn_dev` —
`D_bf16 = scale_a · scale_b · A_fp8 @ B_fp8`, K-N layout, no
transpose) is also exposed to JAX via the XLA FFI handler
`flashvla::fp8_gemm_bf16_out` in
[`csrc/training/jax_ffi/fp8_gemm_ffi.cu`](../csrc/training/jax_ffi/fp8_gemm_ffi.cu).
**No new GEMM kernel** — the FFI handler is a 30-line wrapper
that forwards XLA's PlatformStream + buffer pointers to the same
`fp8_nn_dev` entry point. The JAX-side training driver
([`training/jax/`](../training/jax/)) routes openpi's
`lora.Einsum` / `lora.FeedForward` matmuls through this handler;
performance numbers (cos parity, microbench, 30k bench) are in
[`training/jax/README.md`](../training/jax/README.md).

Other GEMMs:
- `gmm_fp16` — grouped matmul for batches with different shapes, fp16 (used by the DiT batched head projection).
- `fp8_gemm_descale_bf16out` / `fp8_gemm_descale_f32out` / `fp8_gemm_descale_fp16` — non-CUTLASS path (cuBLASLt) with descale; decoder and AE go through this one (using the `.t().contiguous()` layout).

### 2.2 Norm family (13 variants)

| kernel | fusion | typical use |
|--------|--------|-------------|
| `rms_norm` / `rms_norm_fp16` | plain RMSNorm | — |
| `rms_norm_inplace` | in place | debugging |
| `rms_norm_fp8` / `rms_norm_fp8_fp16` | RMSNorm + quantize to fp8 | before a GEMM |
| `rms_norm_fp8_noweight_bf16/fp16` | weightless RMSNorm + quantize | AdaRMS style (weight is 1, modulation done by a Dense layer) |
| `residual_add_rms_norm` | residual + RMSNorm | post-residual layer |
| `residual_add_rms_norm_fp8` / `_fp8_fp16` | above + quantize | between two consecutive layers |
| `residual_add_rms_norm_fp8_noweight_fp16/bf16` | weightless + residual + quant | ★ workhorse fusion for Pi0.5 / Pi0 encoder |
| `plain_rms_fp8_fp16` | RMSNorm + quant (no residual) | first layer |
| `plain_res_rms_fp8_fp16` | residual + RMSNorm + quant, alternate layout | — |
| `ada_rms_norm_style` / `_fp8` | AdaRMSNorm style (scale + shift params) | Pi0.5 decoder |
| `adarms_fp16` | AdaRMSNorm with pure fp16 out | Pi0.5 decoder (layers that skip the quant) |
| `fused_adarms_fp8_static_fp16` | AdaRMSNorm + fp8 quant + descale | workhorse fusion for Pi0.5 decoder |
| `layer_norm` / `_fp16` / `_fp8` / `_no_affine_fp16` | LayerNorm (non-RMS) | SigLIP attention / FFN LN |
| `ada_layer_norm_fp16` | AdaLN (scale + shift) | DiT adaLayerNorm |
| `gate_residual_ada_norm_fp8` | gate×value + residual + AdaNorm + quant | mid-layer fusion in Pi0.5 decoder |
| `gate_res_adarms_fp8_static_fp16` | same idea, AdaRMS variant | Pi0.5 decoder |

### 2.3 Gate + GELU/SiLU family

> **Activation reference (READ THIS FIRST):**
> - `gate_geglu_*` kernels = **GELU (tanh-approx)** — Paligemma / Gemma / Pi0 / Pi0.5 FFN.
> - `silu_mul_split_*` kernels = **true SiLU (swish)** — GROOT Qwen3 / BAGEL FFN.
> - `fused_add_silu_*` / `silu_inplace_*` = **true SiLU**.

```
# GELU-based (GEGLU)
gate_geglu                                 # split version (separate gate, up)
gate_geglu_merged                          # merged: single [N, 2H] input, split in half
gate_geglu_merged_fp16
gate_geglu_merged_fp8                      # quantized output
gate_geglu_merged_fp8_fp16                 # ★ workhorse for Paligemma encoder / decoder
gate_geglu_fp4_sfa_v2_fp16                 # NVFP4 variant (merged GEGLU → FP4 + SFA)
gate_geglu_mul_fp4_sfa_v2_fp16             # same + AWQ inv_s multiply
geglu_two_fp4_to_fp4                       # P1 split-GU combiner (two FP4 inputs → FP4)
geglu_two_mul_fp4_to_fp4                   # same + AWQ-Down inv_s multiply

# True SiLU (swish)
silu_mul_split_fp8_fp16                    # ★ Qwen3 (GROOT) / BAGEL path
silu_inplace_fp16                          # plain SiLU, no mul
fused_add_silu_fp16                        # residual + SiLU fused

# Plain GELU in-place
gelu_inplace / gelu_inplace_fp16

# Other gating
gate_mul_residual                          # gate × value + residual (no activation)
gate_res_fp16
geglu_fp8_static_fp16                      # DiT GELU-variant FFN fusion
```

**GEGLU vs true SiLU**: `gate_geglu_*` and `geglu_two_*` compute tanh-approx GELU (`x/(1+exp(-1.5957·x·(1+0.044715·x²)))`), matching Paligemma / Gemma / Pi0 / Pi0.5 FFN. **GROOT Qwen3** (which actually needs SiLU) must use `silu_mul_split_fp8_fp16` — the split path — not the merged GEGLU kernels.

### 2.4 Attention family (5 backend variants)

| kernel | scenario | protocol site |
|--------|----------|---------------|
| `fmha_strided_full` | SigLIP 2D visual attention (standard Q×K matmul) | `siglip` |
| `fmha_strided_forward` / `fmha_forward` | legacy path | — |
| `attention_qkv_fp16` | standard GQA encoder / decoder | `encoder` / `decoder` kernel=`"standard"` |
| `attention_qkv_fp16_padded` | padded variant | — |
| `attention_qkv_fp16_state_masked` | Pi0 decoder where the first token is a state (masked) | `decoder` kernel=`"state_masked"` |
| `attention_mha_fp16` | full MHA (GROOT Qwen3 + DiT) | kernel=`"mha"` |

**New-model forwards do not call these directly** — they go through the [`AttentionBackend` protocol](../flash_rt/hardware/backend.py) via `attn.run(site=..., layer_idx=..., ...)`. See [`adding_new_model.md §2.1`](adding_new_model.md).

### 2.5 QKV split / RoPE family

```
qkv_split                                # split QKV → three buffers
qkv_split_rope                           # split + RoPE (does not write KV cache)
qkv_split_rope_kvcache_fp16              # ★ encoder / decoder workhorse: split + RoPE + KV cache write
rope_apply                               # apply RoPE to existing Q/K
rope_rotate_half_fp16                    # GPT-NeoX style rotate_half
gpu_repeat_interleave_heads              # expand GQA key/value (when implicit broadcast is not used)
```

**Picking a RoPE style**:
- Pi0.5 / Pi0: pair-interleave (Q/K weights stored already `interleave_qk`-transformed) + `qkv_split_rope_kvcache_fp16`.
- Qwen3 / GROOT: `rope_rotate_half_fp16` (Qwen3 standard) + `attention_mha_fp16`.

If your new model is in the Gemma / Paligemma family → pair-interleave path. If it's Qwen / LLaMA family → rotate_half path. Do not mix them.

### 2.6 Quantization family (6)

```
quantize_fp8                   # host scale (legacy)
quantize_fp8_static            # static device scale
quantize_fp8_static_fp16       # ★ fp16 input → fp8, static d_scale pointer
quantize_fp8_device            # scale computed dynamically on device
quantize_fp8_device_fp16
cast_fp16_fp8                  # plain type cast (scale = 1.0)
```

The production path only ever uses `_static_fp16` (after calibration the `d_scale` is always a device-side f32 pointer). The `_device` variants are only used inside the calibration forward (to measure amax).

The same `quantize_fp8_static` kernel is also re-exported on the
JAX side via the XLA FFI handler `flashvla::quantize_fp8_static`
in `flash_rt_jax_ffi.so` — the binding is in
[`csrc/training/jax_ffi/activation_quantize_ffi.cu`](../csrc/training/jax_ffi/activation_quantize_ffi.cu)
and contains zero new compute logic (it just forwards the
caller's CUDA stream + buffer pointers to the same kernel
declared at [`csrc/kernels/quantize.cuh:21-22`](../csrc/kernels/quantize.cuh)).
See [`kernel_catalog.md § 11`](kernel_catalog.md) for the JAX-side
listing.

### 2.7 Miscellaneous kernels

```
patch_im2col + patch_embed_bias_pos     # SigLIP patch unfolding
add_bias_fp16                            # bias add
bias_residual / bias_residual_fp16      # bias + residual
residual_add / residual_add_fp16        # pure residual
softmax_fp16                             # used by Qwen3 / DiT (non-FMHA path)
gpu_copy / gpu_strided_copy_fp16        # D2D copy
gpu_fill_neginf_fp16                    # attention mask initialization
gpu_euler_step                          # flow-matching Euler integrator (action projection)
gpu_cast_fp32_to_fp16
```

### 2.8 Runtime queries

```
has_cutlass_fmha     # bool — CUTLASS FMHA available on SM120+? (usually false on Thor)
has_cutlass_sm100    # bool — SM100-specific kernels
has_nvfp4            # bool — NVFP4 support (false on Thor)
get_sm_version       # int — e.g. 110
load_fmha_library    # load libfmha_fp16.so (legacy)
load_fmha_strided_library  # load libfmha_fp16_strided.so (required by SigLIP)
```

**Pi0-FAST dispatches on hardware via `hasattr(fvk, 'cutlass_fp8_sq')`** (see [`frontends/torch/pi0fast.py`](../flash_rt/frontends/torch/pi0fast.py)). If your new model needs to run on both Thor and RTX, follow the runtime-fork pattern used by Pi0-FAST.

---

## 3. Current production fusion shape (one Pi0.5 encoder layer)

One Pi0.5 encoder layer is currently **9 kernel launches**, in this order (a trimmed view of `shared_primitives.encoder_forward_calibrate`):

```
1. residual_add_rms_norm_fp8_noweight_fp16      # res + RMSNorm + quant  (fused 3→1)
2. cutlass_fp8_t1                                # QKV GEMM
3. qkv_split_rope_kvcache_fp16                   # split + RoPE + KV write
4. attention_qkv_fp16                            # GQA attn
5. quantize_fp8_static_fp16                      # attn_out → fp8
6. cutlass_fp8_sq                                # O proj GEMM
7. residual_add_rms_norm_fp8_noweight_fp16      # res + RMSNorm + quant  (fused)
8. cutlass_fp8_t1                                # Gate+Up GEMM [2H, D]
9. gate_geglu_merged_fp8_fp16                 # GELU + mul + quant  (fused 3→1)
10. cutlass_fp8_wide                             # Down GEMM
```

Compared against the Myelin-compiled version (70.2ms) which implements the same logic in **~20 kernels**, FlashRT has folded 3+3 elementwise + quant ops into the fused RMSNorm (steps 1 and 7) and another 3 into `gate_geglu` (step 9).

**Default template for a new model**: if the architecture is a Paligemma / Gemma variant, fork the Pi0.5 or Pi0 encoder forward. Swapping the dim constants is usually enough to get it running. **Do not** redesign the kernel sequence from scratch.

---

## 4. What is fusable, and what isn't

### 4.1 Fusable patterns (verified)

| fusion | why it works | representative kernel |
|--------|--------------|------------------------|
| residual + rms + quant | all three are elementwise plus one reduction, memory-bound, share loads | `residual_add_rms_norm_fp8_noweight_fp16` |
| gate + silu/gelu + mul + quant | one elementwise chain | `gate_geglu_merged_fp8_fp16` |
| AdaRMS + scale + quant | dense output flows straight into norm | `fused_adarms_fp8_static_fp16` |
| CUTLASS GEMM + GELU epilogue | CUTLASS supports epilogue fusion | `cutlass_fp8_gelu` |
| patch_embed im2col + GEMM + bias | two kernels + elementwise | currently 3 kernels (im2col + GEMM + bias_pos), not yet fused — open optimization opportunity |

### 4.2 Non-fusable patterns (or too risky)

| attempted fusion | why it fails | evidence |
|------------------|--------------|----------|
| **GEMM + the second GEMM inside GeGLU** | GeGLU has the shape `down(gate(x_gate) × x_up)`; the middle per-element mul is not epilogue-shaped | `v1.3-opt-3dm3d-plan.md §9`, mlir-tensorrt does not support it |
| **RMSNorm + FP8 cast + GEMM** | Myelin has a fusion barrier at cast boundaries; moving the cast degrades fusion | OPT-3 Softmax Cast: +3.30ms regression |
| **Merge Q + KV GEMMs** | KV is too small (few GQA heads), merged shape gets a worse tactic | v1.5-QKV: +0.47ms regression |
| **Inline `time_mlp` into the decoder graph** | changes graph topology → Myelin picks worse tactics | v1.5-B2: +2.39ms regression |
| **Pre-compute mask and bypass the graph** | the real improvement is drowned by ±2ms tactic noise | v1.5-B4: real gain ~0.3ms, reverted |
| **Remove the attention QDQ** | reorders the Q/K/V fp8 → fp16 attention-input conversion | M6: +1.03ms regression |

**Key lessons**:
1. **Myelin fusion depends on a specific Cast topology** — any change in Cast placement is high risk.
2. **Moving or removing a GEMM changes graph topology → Myelin tactic regresses.**
3. **Safe work = pre-compute non-GEMM ops** (mask, positions, RoPE tables).
4. **Risky work = altering the GEMM call chain.**

Full list of failures in §5.

---

## 5. Known failed optimizations (lessons learned) <a name="failed-optimizations"></a>

| code | attempt | result | reason |
|------|---------|--------|--------|
| OPT-5 | refactor `einsum → matmul` | **+5.18ms** | reshape/transpose altered topology → fusion degraded |
| OPT-3 | drop the post-softmax f32→fp16 cast | **+3.30ms** | saved 198 converts but Myelin fusion got worse |
| v1.5-B2 | batch `time_mlp` + inline `embed_suffix` | **+2.39ms** | changed graph topology |
| v1.5-QKV | merge Q + KV | +0.47ms | GQA KV too small |
| v1.5-P1 | einsum subscript `BTGKS` | +0.60ms | — |
| M6 | remove attention QDQ | +1.03ms | broke Myelin's fusion assumptions |
| v1.5-B4 | pre-compute mask / positions | ~0ms (lost in noise) | real gain < ±2ms tactic fluctuation |

For contrast, successful optimizations:
- M1 GQA implicit broadcast: -2.66ms
- M5 tiny-matmul skip QDQ: -2.83ms
- OPT-4 SigLIP attention FP8: -0.21ms
- v1.4 Gate+Up Wab merge: -0.40ms
- v1.6-L2 AdaRMSNorm Dense pre-compute: -4.8ms

**Rules of thumb for whether a fusion is worth trying**:
- If the change only touches **non-GEMM elementwise ops** (bias, mask, RoPE pre-compute, activation-scale loading) → safe.
- If it touches the **GEMM call chain** (merging, splitting, reordering) → extremely high risk, must be A/B-tested 5× to confirm.
- If it shifts a **FP8 cast position** (moving a quantize kernel before/after another op) → extremely high risk.

---

## 6. CUDA Graph capture rules <a name="cuda-graph-rules"></a>

1. **Pre-allocate every buffer** inside `_load_weights`; forward only reads `.data_ptr()`. Dynamic allocation causes capture to fail or misbehave.

2. **No Python-side conditionals in forward.** For example, `if layer_idx == 0:` will break capture. (Pi0.5 encoder skips attention on the last layer by computing `last = (l == L-1)` **inside** the forward and using a Python conditional to select **which kernels to launch** — what capture sees is still a fixed launch sequence, which is fine.)

3. **No `_gpu_sync` / `.cpu()` / `.numpy()` in forward.** These are allowed in the calibration forward since it is not captured.

4. **No `torch.empty()` or `CudaBuffer.device_empty()` in forward.**

5. **GEMM descale alpha is a host scalar argument**, not a device pointer. Each layer's alpha is effectively a compile-time constant (read from `self._enc_alpha_host`); it cannot change during capture.

6. **Warmup shapes must exactly match the real inference shapes**: tokens, `num_views`, `seq_len`, everything. Pi0.5 does warmup in `_capture_siglip_graph` using zeroed images:

   ```python
   self._img_buf.zero_()  # zero input to avoid accumulating inf
   ...
   fvk.patch_im2col(...)
   self._sig_x.zero_()    # force zero so warmup never produces nan
   siglip_forward(...)    # capture
   ```

7. **Capture once, replay many times.** Pi0.5's 10-step flow-matching decoder is a **single graph** that is replayed 10 times — the contents of `self._g_noise` are just updated before each replay.

---

## 7. Checklist for a new pipeline forward

- [ ] Every kernel argument is `int` (pointer) / `float` (scalar) / `stream_int` — no `torch.Tensor`, no numpy.
- [ ] Before each GEMM call, **alpha = act_scale × weight_scale is already a host float scalar** (see `calibration.md §2.3`).
- [ ] QKV split + RoPE done in one shot (`qkv_split_rope_kvcache_fp16`) — do not split into 3 kernels manually.
- [ ] residual + RMSNorm + FP8 quant **use the fused kernel** (`residual_add_rms_norm_fp8_*`) — do not hand-roll the three steps.
- [ ] Gate+Up GEMM + GELU + Mul + Quant **use the fused kernel** (`gate_geglu_merged_fp8_fp16`).
- [ ] Attention goes through `attn.run(site=..., layer_idx=..., ...)` — do not call `fvk.attention_qkv_fp16` directly.
- [ ] Every intermediate buffer name **matches the bufs dict key used in `_calibrate`** — a mismatch will segfault.
- [ ] `last = (i == L-1)` is handled in **Python**, not baked into a kernel launch.
- [ ] Zero `_gpu_sync` / `.cpu()` / `block_until_ready` calls inside forward.
- [ ] Start by copying the Pi0.5 or Pi0 encoder/decoder forward template, then adjust dim constants layer by layer. **Do not** redesign the kernel sequence.

---

## 8. Want to add a new fused kernel?

**Ask yourself first**:
1. Are all the ops you want to fuse on **the same memory stream** (all fp16 in/out, or fp8 in + fp16 out)? If not, it won't fuse cleanly.
2. Taken together, are these ops bandwidth-bound? If there's a GEMM in the middle (compute-bound), fusion gains are tiny (GEMMs are already compute-dominated).
3. Will your change affect how the Myelin compiler picks tactics for the rest of the pipeline? It almost certainly will — even though you are writing a hand-written CUDA kernel, you are also changing the Python / pybind11 call sequence.

**If you decide to do it**:
1. Write a standalone benchmark first (not inside a CUDA Graph), using CUDA events to measure the kernel in isolation. **Median of 3-5 runs.**
2. Add the new kernel on a shadow forward path and compare **absolute end-to-end latency** (A/B), not per-kernel time.
3. Accuracy cosine must be ≥ the original path.
4. Before committing, clear the Myelin timing cache and rerun 3 times to make sure tactic jitter is not masking a regression as a win.

**Compiling the new kernel**: modify `csrc/` and rebuild the `.so`. **Always back up the original `.so` first** (so a regression can be reverted instantly), then verify the symbols and dependencies of the new build with `nm` / `ldd` before swapping it into the deployed pipeline.

---

## 9. Related documents

- [`optimization-details.md`](../optimization-details.md) — full Pi0.5 optimization accounting + fusion comparison vs Myelin
- [`adding_new_model.md`](adding_new_model.md) — top-level guide for adding a new model
- [`calibration.md`](calibration.md) — FP8 scale and alpha mechanics
- [`stable_api.md`](../stable_api.md) — `AttentionBackend`, `WEIGHT_SPEC` public interfaces
- §5 of this document — all known failed optimizations, captured as project lessons
