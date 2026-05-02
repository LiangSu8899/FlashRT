# FlashRT Optimization Details

> Pi0.5 on Jetson AGX Thor (SM110, 20 SMs, 225 GB/s DRAM, 32 MB L2)
> Baseline: mlir-tensorrt + Myelin compiler engine v1.6-L2 = **70.2 ms**
> FlashRT: Hand-written C++/CUDA pipeline = **44 ms (CUDA Graph)**
> Improvement: **-26.2 ms (37.3% faster)**

---

## 1. Optimization Taxonomy

### 1.1 Kernel Fusion (Memory-Bound Op Elimination)

FlashRT fuses memory-bound ops that the compiler treats as separate kernels. On Thor, all non-GEMM ops are bandwidth-bound (D=2048 or 1024, 20 SMs). Each standalone kernel incurs DRAM read + write + launch overhead.

| Fused Kernel | Replaces (Compiler) | Passes Saved | Where |
|---|---|---|---|
| `rms_norm_fp8_static` | rms_norm + memset + amax_reduce + apply_scale + fp8_quantize | 4→1 | Encoder (18L) |
| `residual_add_rms_norm_fp8_static` | elementwise_add + rms_norm + memset + amax + apply + quant | 5→1 | Encoder (18L), Decoder (18L×10) |
| `fused_adarms_fp8_static` | adarms_modulate + memset + amax + quant + descale | 4→1 | Decoder (18L×10) |
| `gate_res_adarms_fp8_static` | gate_mul + residual + adarms_mod + memset + amax + quant + descale | 6→1 | Decoder (18L×10) |
| `gate_geglu_merged_fp8` | gelu_approx + gate_mul + memset + amax + quant + descale | 5→1 | Encoder (18L), Decoder (18L×10) |
| `siglip_layernorm_1read` | 3-pass layernorm (mean, var, normalize) | 3→1 | SigLIP (27L) |

**Total kernel launches eliminated**: Compiler engine has **~5,300 layers / ~21,000 kernel launches** per inference. FlashRT has ~13 kernels/decoder-layer × 180 blocks + ~500 encoder+siglip = **~2,840 kernel launches**. Reduction: **~18,000 launches eliminated (85%)**.

### 1.2 FP8 Static Quantization (Dynamic→Static Conversion)

The compiler engine uses **dynamic FP8 quantization** at every GEMM boundary:

```
Compiler (per GEMM):
  activation → amax_reduce → compute_scale → quantize_fp8 → GEMM → dequantize
  = 4 extra kernels + 1 device-sync (amax) per GEMM

FlashRT (per GEMM):
  activation (already FP8 from fused preceding kernel) → GEMM(descale=precomputed)
  = 0 extra kernels, scale is a compile-time constant
```

**Calibration**: Run 1 forward pass with amax measurement per quantization point, store scales. All subsequent inference uses static scales fused into preceding kernels.

**Impact**: Eliminates ~630 standalone quantize/dequantize kernel launches across the full pipeline.

### 1.3 RMSNorm Weight Fusion (Graph-Level Precomputation)

Standard RMSNorm: `y = x / rms(x) * (1 + weight)`. The `(1 + weight)` is constant per layer.

**FlashRT**: Fuse `(1 + weight)` into the **following GEMM weight** at load time:

```python
# At weight load (once):
fused_qkv_weight = qkv_weight * (1 + rms_norm_weight)

# At inference:
x_norm = rms_norm_no_weight(x)  # simpler kernel, no weight multiply
qkv = x_norm @ fused_qkv_weight  # weight already absorbs norm scale
```

**Applied to**: Encoder QKV (18L), Encoder GateUp (18L), Pi0 Decoder QKV (18L), Pi0 Decoder GateUp (18L).

**Impact**: Eliminates per-element multiply in RMSNorm kernel. The `rms_norm_fp8_noweight` kernel is ~15% faster than weighted version.

### 1.4 AdaRMSNorm Dense Precomputation (Pi0.5 Decoder)

Pi0.5 decoder uses AdaRMSNorm with learned modulation: `y = adarms(x, style)` where `style = time_emb @ mod_weight + mod_bias`.

**Compiler (v1.0-v1.4)**: 370 Dense GEMMs computed **inside the engine** every inference.

**FlashRT + Compiler v1.6-L2**: Precompute all `style` vectors on CPU at prompt-set time. Inject as constants → 370 Dense GEMMs removed from GPU graph.

**Impact**: -5.5 ms (compiler v1.4→v1.6-L2), replicated in FlashRT as Python precompute in `set_prompt()`.

### 1.5 Vectorized Memory Access

| Kernel | Before | After | Speedup |
|---|---|---|---|
| GELU+gate+FP8 | scalar fp16 load/store + per-element FP8 store | `half2` load + `uint32` packed 4×FP8 store | 2.3x |
| RMSNorm+FP8 | 2-pass (mean, normalize) + scalar FP8 store | 1-pass register-cached + `uint16` packed 2×FP8 | 1.4x |
| SigLIP LayerNorm | 3-pass (mean, var, norm) with `float32` gamma/beta | 1-pass register-cached + `bfloat16x2` gamma/beta | 2.1x |
| FP8 quantize | separate amax reduce + scale compute + cast | fused into preceding norm/activation kernel | eliminated |

### 1.6 CUDA Graph (Zero Dispatch Overhead)

The entire pipeline (SigLIP + PostLN + Encoder + Decoder) is captured as CUDA Graph.

**Compiler engine**: Already uses CUDA Graph internally (TRT engine execution). However, the graph contains **21,000+ nodes** with Myelin's fine-grained kernel scheduling.

**FlashRT**: Graph contains **~2,840 nodes** (7.4x fewer). Fewer nodes = faster graph instantiation + less scheduling overhead.

**Graph autotune**: CUDA Graph instantiation on Thor is non-deterministic (±2ms variance). FlashRT recaptures until a fast schedule is found. Typical: 0-1 retries needed.

---

## 2. End-to-End Timing Breakdown (Pi0.5, 2-view LIBERO)

### 2.1 Component Comparison

| Component | Compiler v1.6-L2 | FlashRT | Delta | Notes |
|---|---|---|---|---|
| **SigLIP** (27L vision) | 1.0 ms | 6.3 ms | +5.3 ms | See §2.2 |
| **Encoder** (18L VLM) | 36.9 ms | 19.8 ms | **-17.1 ms** | See §2.3 |
| **Decoder** (18L×10 steps) | 36.1 ms | 24.5 ms | **-11.6 ms** | See §2.4 |
| **Graph pipelining** | (included) | -3.9 ms | — | SigLIP overlaps with upload |
| **Python/Host** | ~0 ms | 2.1 ms | +2.1 ms | Image preprocessing + output |
| **Total** | **~74 ms** (nsys) / **70.2 ms** (CUDA events) | **44 ms** | **-26 ms** | |

> Note: Compiler's "74 ms nsys" vs "70.2 ms CUDA events" gap is due to nsys profiling overhead. FlashRT's 44 ms is CUDA events measurement (wall-clock `perf_counter` with `cudaStreamSynchronize`).

### 2.2 SigLIP: Why Compiler is Faster (1.0 ms vs 6.3 ms)

The compiler engine benefits from **Myelin's global fusion** across SigLIP's 27 layers:
- QKV GEMM + bias + reshape fused into single kernel
- LayerNorm + GELU + bias fused as GEMM epilogues
- Entire 27-layer stack shares L2 cache optimally (Myelin schedules for L2 residency)

FlashRT uses **cuBLASLt FP8 GEMM** + separate custom kernels for norm/activation:
- 27 layers × ~6 kernels/layer = ~162 kernel launches (vs compiler's ~138 with deeper fusion)
- FP8 weight reduces DRAM traffic (15 MB/layer < 32 MB L2), partially compensating

**This is the one area where the compiler approach wins.** SigLIP's large batch (S=512) and simple attention (no RoPE, no cross-attention) is ideal for Myelin's fusion.

### 2.3 Encoder: FlashRT 17.1 ms Faster

| Sub-component | Compiler v1.6-L2 | FlashRT | Delta | Root Cause |
|---|---|---|---|---|
| FFN GEMMs (gate_up + down) | 15.8 ms | 9.2 ms | **-6.6 ms** | cuBLASLt tactic vs Myelin CUTLASS |
| Attention (QKV + softmax + attn@V) | 7.1 ms | 4.3 ms | **-2.8 ms** | cuBLAS attention vs Myelin kgen+GEMM |
| RMSNorm + QDQ + RoPE | 5.4 ms | 1.4 ms | **-4.0 ms** | Fused norm+FP8, eliminated QDQ |
| Data movement (transpose, concat) | 6.9 ms | 0 ms | **-6.9 ms** | No explicit transposes needed |
| Other kgen | 1.7 ms | 0 ms | **-1.7 ms** | Fused into preceding kernels |
| Kernel dispatch overhead | — | 4.9 ms | +4.9 ms | ~290 kernel launches × ~17μs gap |
| **Total** | **36.9 ms** | **19.8 ms** | **-17.1 ms** | |

**Key wins**:
1. **No data movement**: Compiler inserts MoveConcat/Transpose for layout conversion between ops. FlashRT uses consistent row-major layout throughout — zero explicit transposes.
2. **Fused norm+FP8**: Compiler has separate RMSNorm → QDQ → Quantize → GEMM. FlashRT: `rms_norm_fp8_static` produces FP8 output directly.
3. **cuBLASLt vs Myelin tactic**: For Se=522 (encoder sequence), cuBLASLt selects better CUTLASS configurations than Myelin's tactic search for some shapes.

### 2.4 Decoder (AE): FlashRT 11.6 ms Faster

| Sub-component | Compiler v1.6-L2 | FlashRT | Delta | Root Cause |
|---|---|---|---|---|
| FP8 GEMMs (QKV+O+GateUp+Down) | 24.2 ms | 15.8 ms | **-8.4 ms** | cuBLASLt + static descale vs Myelin FP8 |
| Norm + activation + fusion kgen | 11.9 ms | 1.8 ms | **-10.1 ms** | All fused into 3 custom kernels/layer |
| Kernel dispatch overhead | — | 6.9 ms | +6.9 ms | 2,340 graph nodes × ~3μs |
| **Total** | **36.1 ms** | **24.5 ms** | **-11.6 ms** | |

**Key wins**:
1. **10.1 ms from fusion**: Compiler's ~9,700 kgen launches (norm, GELU, softmax, QDQ) across 180 blocks. FlashRT replaces with 3 fused kernels per layer: `fused_adarms_fp8_static` + `gate_res_adarms_fp8_static` + `gate_geglu_merged_fp8_fp16`. Despite GPU overlap making kgen "free" in the compiler, FlashRT's fused approach reduces total graph nodes and improves scheduling.
2. **8.4 ms from GEMM**: At S=10 (decoder batch), cuBLASLt FP8 with pre-baked descale outperforms Myelin's CUTLASS FP8 + separate descale. Myelin's per-GEMM tactic search advantage diminishes at very small batch sizes where launch overhead dominates.

---

## 3. Per-Layer Kernel Comparison (Decoder, 1 Layer)

### Compiler v1.6-L2 (per decoder layer, ×180 total)

```
#   Kernel                              Time     Type
1.  AdaRMSNorm kgen                     6 μs     elementwise (Myelin fused)
2.  QKV proj FP8 GEMM (CUTLASS)        17 μs    compute
3.  RoPE kgen                           5 μs     elementwise
4.  RoPE kgen                           5 μs     elementwise
5.  Q@K^T GEMM                          4 μs     compute
6.  Transpose+Select kgen               3 μs     data movement
7.  Softmax kgen                        4 μs     elementwise
8.  Attn@V GEMM                         4 μs     compute
9.  QDQ kgen                            3 μs     quantize
10. Transpose kgen                      3 μs     data movement
11. O proj FP8 GEMM                    12 μs    compute
12. Gate mul + residual kgen            5 μs     elementwise
13. AdaRMSNorm kgen                     6 μs     elementwise
14. Gate+Up FP8 GEMM (Wab, CUTLASS)   17 μs    compute
15. GELU kgen                           8 μs     elementwise
16. Down FP8 GEMM (CUTLASS)           17 μs    compute
17. Gate mul + residual kgen            5 μs     elementwise
18-22. QDQ/Cast kernels (various)      15 μs    quantize/cast
────────────────────────────────
     ~22 kernels/layer                ~139 μs   total
     ×180 layers                      ~25 ms    (+ GPU overlap → 36ms wall)
```

### FlashRT (per decoder layer, ×180 total)

```
#   Kernel                              Time     Type
1.  fused_adarms_fp8_static             8 μs     AdaRMSNorm+FP8 quantize (fused)
2.  fp8_gemm_descale (QKV)             14 μs    cuBLASLt FP8 + static descale
3.  qkv_split_rope_kvcache             5 μs     split Q/K/V + RoPE + KV cache write
4.  cublasGemmEx (Q@K^T)               4 μs     cuBLAS attention
5.  softmax_fp16                        3 μs     vectorized softmax
6.  cublasGemmEx (attn@V)              4 μs     cuBLAS attention
7.  quantize_fp8_static                 2 μs     O proj input quantize
8.  fp8_gemm_descale (O proj)          12 μs    cuBLASLt FP8
9.  gate_res_adarms_fp8_static          9 μs     gate×res + AdaRMSNorm + FP8 (fused)
10. fp8_gemm_descale (Gate+Up)         14 μs    cuBLASLt FP8
11. gate_geglu_merged_fp8_fp16          4 μs     GELU(gate)×up + FP8 (fused)
12. fp8_gemm_descale (Down)            14 μs    cuBLASLt FP8
13. gate_res_adarms_fp8_static          9 μs     residual + cross-layer norm (fused)
────────────────────────────────
     13 kernels/layer                  ~102 μs   per layer
     ×180 layers                       ~18.4 ms  (+ dispatch → 24.5ms wall)
```

### Key Differences

| Metric | Compiler | FlashRT | Ratio |
|---|---|---|---|
| Kernels per layer | 22 | 13 | 0.59x |
| Compute time per layer | ~139 μs | ~102 μs | 0.73x |
| Total kernel launches (decoder) | 16,480 | 2,340 | 0.14x |
| Data movement kernels | 4/layer (transpose, QDQ) | 0/layer | eliminated |
| Standalone quantize kernels | 3-4/layer | 0/layer (fused) | eliminated |

---

## 4. Optimization Progression

### Compiler Path (mlir-tensorrt + Myelin)

| Version | Latency | Delta | Method |
|---|---|---|---|
| v1.0 (initial FP8) | 85.3 ms | — | JAX export → MLIR → TRT FP8, dynamic quantize |
| v1.2-OPT4 (GQA + skip QDQ) | 78.5 ms | -6.8 ms | Implicit GQA broadcast, skip small-GEMM QDQ, SigLIP FP8 |
| v1.4-Wab (gate+up merge) | 75.7 ms | -2.8 ms | 2 GEMM → 1 wide GEMM, halve input QDQ |
| **v1.6-L2** (Dense precompute) | **70.2 ms** | **-5.5 ms** | 370 Dense removed, graph simplification → better Myelin tactic |

### FlashRT Path (C++/CUDA hand-written)

| Phase | Latency | Delta | Method |
|---|---|---|---|
| Phase 2 (initial port) | 83.0 ms | — | Replace Myelin with cuBLASLt + manual kernels |
| Phase 3 (kernel fusion) | 64.9 ms | -18.1 ms | Fused norms, 1-pass register cache, packed FP8, SigLIP 1-read LN |
| Phase 4 (SigLIP FP8) | 63.6 ms | -1.3 ms | BF16→FP8 weight (30→15 MB/layer, L2 residency) |
| Phase 5 (FP16 + FMHA) | 62.0 ms | -1.6 ms | BF16→FP16 throughout, CUTLASS strided FMHA |
| **Phase 6** (full static) | **47.8 ms** | **-14.2 ms** | Calibrate→static FP8, PostLN in graph, fused encoder |
| + CUDA Graph autotune | **44 ms** | **-3.8 ms** | Recapture for optimal graph schedule |

---

## 5. Where the 26 ms Comes From

Summary of FlashRT's advantage over compiler v1.6-L2 (70→44 ms):

| Category | Savings | Mechanism |
|---|---|---|
| **Encoder norm+QDQ elimination** | -4.0 ms | `rms_norm_fp8_static` replaces 4 separate kernels |
| **Encoder data movement elimination** | -6.9 ms | No MoveConcat/Transpose (consistent layout) |
| **Encoder GEMM improvement** | -6.6 ms | cuBLASLt tactic advantage at Se=522 |
| **Decoder kgen→fused kernels** | -10.1 ms | 3 fused kernels replace ~10 kgen per layer |
| **Decoder GEMM improvement** | -8.4 ms | cuBLASLt static descale at S=10 |
| **SigLIP regression** | +5.3 ms | Myelin's 27-layer global fusion is unbeatable |
| **Host/Python overhead** | +2.1 ms | Image preprocessing, output download |
| **Graph pipelining** | -3.9 ms | SigLIP graph overlaps with image upload |
| **Graph autotune** | -3.8 ms | Eliminates instantiation variance |
| **Net improvement** | **-26.2 ms** | **70.2 → 44 ms (37.3% faster)** |

---

## 6. Why the Compiler Cannot Match FlashRT

### Structural Limitation: Opaque Boundary

Any TRT Plugin inserted into the Myelin engine creates an "opaque boundary" — Myelin cannot fuse across it. Tested with RoPE Plugin:
- Plugin alone: -18 ms (RoPE kernel 10x faster)
- Myelin fusion regression: +36 ms (surrounding GEMMs lose cooperative scheduling)
- **Net effect: +10 ms slower**

### S=10 Latency Regime

The decoder operates at S=10 (10 action tokens). At this batch size:
- All GEMMs are **launch-bound**, not compute-bound
- Kernel fusion savings (memory traffic) are negligible (< 1 KB intermediate data)
- **Dispatch overhead dominates**: compiler's 16,480 launches × ~2μs = ~33 ms dispatch cost
- FlashRT's 2,340 launches × ~3μs = ~7 ms dispatch cost
- **Dispatch savings alone: ~26 ms** (matches the total improvement)

### Myelin Tactic Search vs cuBLASLt

For large shapes (encoder Se=522), cuBLASLt consistently selects better CUTLASS configurations than Myelin's tactic search. This is because:
1. cuBLASLt has access to NVIDIA's latest tactic database (updated with driver)
2. Myelin's search space is constrained by the engine builder's time budget
3. For very small shapes (S=10 decoder), both are launch-bound and the difference is minimal

### The Fundamental Trade-off

```
Compiler (Myelin):  Global fusion view → fewer kernels → but limited to Myelin's patterns
FlashRT:           No global fusion   → more kernels → but each kernel is purpose-built

At large batch (SigLIP S=512):  Myelin's global fusion wins (1 ms vs 6 ms)
At small batch (Decoder S=10):  FlashRT's fewer launches wins (24 ms vs 36 ms)
At medium batch (Encoder S=522): FlashRT's cuBLASLt + fusion wins (20 ms vs 37 ms)
```
