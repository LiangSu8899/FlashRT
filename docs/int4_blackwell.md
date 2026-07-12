# INT4 (E0M3) on Blackwell / sm_120 — experimental

> **Status: experimental.** This is a runtime primitive plus a standalone
> validation harness. It is not wired into `flash_rt_kernels` or any model
> pipeline yet; enabling it requires the post-build SASS patch step
> described below. The end-to-end model integration lands in a separate PR.

## Summary

sm_120 (RTX 50-series) exposes a single block-scaled 4-bit tensor-core
instruction, `mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`
(SASS `OMMA.SF.16864`), which `ptxas` only accepts with the `e2m1`
(NVFP4) element type. On this architecture the operand **element format**
is not fixed by the opcode — it is carried in two bits of the 128-bit SASS
encoding, **bit 78 for the A operand and bit 79 for the B operand**:

| bit value | decoded format |
|---|---|
| 0 | `e2m1` — NVFP4, non-uniform codebook `±{0, .5, 1, 1.5, 2, 3, 4, 6}` |
| 1 | `e0m3` — uniform INT4, sign-magnitude codebook `−7 … +7` |

Flipping those bits turns the same instruction into a native INT4 (or
mixed INT4 × NVFP4) multiply. The block-scale path (UE4M3 per-16
scale-factor + fp32 global scale) is unchanged, so INT4 reuses the entire
NVFP4 quantization and swizzle machinery.

This was documented publicly by the Ling Team; see **Credits** below. The
kernels here reproduce and productize that finding for FlashRT.

## Why INT4 and not E2M1

E2M1's non-uniform grid has an inherent *shrinkage bias*: the asymmetric
rounding bins bias quantized values toward the origin, and that bias
accumulates. A uniform 4-bit grid (INT4 / E0M3) removes it, so per-block
scaling and rotation (e.g. randomized Hadamard) translate directly into
better quantization quality. See the Ling UFP4 report in **References**.

Measured on real weights (bit-faithful simulation of the shipped two-level
quantizer; identical UE4M3 per-16 SF + fp32 global scale, `amax/(7·448)`
for INT4 vs `amax/(6·448)` for E2M1), INT4 lowers weight quantization error
(1 − cosine) versus E2M1 on **37 / 37** tensors sampled across a diffusion
DiT, a 3D-conv VAE, and a dense transformer:

| tensor family | mean quant-error reduction (best INT4 vs E2M1) |
|---|---|
| diffusion DiT (attn + FFN) | −15 % |
| 3D-conv VAE (residual convs) | −18 % |
| dense transformer (attn + MLP + lm_head) | −15 % |

Uniform-grid + per-16 rotation (Hadamard, size 16) wins on every tensor;
the same rotation applied under E2M1 never helps, matching the shrinkage
-bias explanation.

## Throughput — identical to NVFP4

Because it is the same tensor-core instruction, INT4 runs at the NVFP4
rate. Issue-rate microbenchmark, register-resident MMA (no memory
traffic), **RTX 5090, CUDA 13.0**:

| operand formats | throughput |
|---|---|
| NVFP4 `e2m1 × e2m1` | 2026.6 TFLOPS |
| INT4 `e0m3 × e0m3` | 2026.8 – 2027.1 TFLOPS |
| mixed `e0m3 × e2m1` | 2027.9 TFLOPS |

*Metric: instruction issue rate (2·M·N·K per MMA), median of 5, no HBM
traffic — this measures the tensor-core rate, not an end-to-end GEMM.* The
standalone test below reproduces ≈ 2013 TFLOPS with the same method.

A prebuilt HF kernel with the same microbenchmark is published at
`huggingface.co/kernels/flashrt/int4-blackwell`.

## Kernels

`csrc/kernels/int4_w4a4_mma_sm120.{cu,cuh}`:

- `int4_w4a4_mma_sm120_full_n_bf16out` — M=1 full-N W4A4 GEMV, bf16 out.
  Same fragment layout and cp.async double-buffering as the NVFP4 twin
  `fp4_w4a4_mma_sm120`.
- `int4_quantize_bf16_sm120` — bf16 → INT4 nibbles + swizzled UE4M3 SF.
- `int4_global_scale_bf16_sm120` — `amax / (7·448)` reduction.
- `int4_w4a4_sm120_codebook_canary` — runtime self-check (returns 0 only
  if the running SASS truly decodes E0M3).
- `extern "C"` wrappers (`flashrt_int4_sm120_*`) for non-C++ backends,
  e.g. a GGML / llama.cpp custom op.

## The kernel is compiled E2M1 and RUN as E0M3

`ptxas` rejects any element type other than `e2m1` for this instruction,
so the `.cu` emits the normal NVFP4 form and the **patch step is part of
linking**. `tools/patch_int4_omma_sm120.py` flips bits 78/79 of every
`OMMA.SF` instruction inside device functions whose (mangled) name
contains `int4_`. It operates on a bare `.cubin`, or in place on a host
ELF (`.so` / executable) with an uncompressed `.nv_fatbin`.

Loading an unpatched binary decodes E2M1 on INT4 data — i.e. wrong
results, undetectable by the launcher. Always gate module load on
`int4_w4a4_sm120_codebook_canary() == 0` (fail-fast). Mixed
W-INT4 × A-E2M1 is available with `--operands b` (patch the weight operand
only).

**Static verification is asymmetric — the runtime canary is
authoritative.** These bits are undocumented: an *unpatched* binary
disassembles normally (the sites are locatable and readable), but once the
bits are set `cuobjdump`/`nvdisasm` no longer decode the instruction as
`OMMA.SF` at all, so a *patched* binary cannot be re-located or re-read
statically. Consequently:

- `patch_int4_omma_sm120.py --verify --expect e2m1` is a reliable
  **pre-patch** gate (exit 0 iff every site is still E2M1; nonzero
  otherwise, including "no sites"). Use it in the build to prove the right
  thing was compiled before patching.
- `--verify --expect int4` returns 1 on an unpatched binary and 2
  (inconclusive) on a patched one — it never claims success by guessing.
- The only authoritative proof that a binary decodes E0M3 is
  `int4_w4a4_sm120_codebook_canary() == 0` at load.

Patch mode fails loudly (nonzero exit) if it flips zero instructions, so a
build step cannot silently ship an unpatched or double-run binary.
Re-verify the encoding after any CUDA-toolkit upgrade.

## Reproduce

Requires an sm_120 GPU (RTX 50-series) and CUDA 13.0+. No checkpoints —
the test uses synthetic data.

```bash
nvcc -std=c++17 -O3 -gencode arch=compute_120a,code=sm_120a \
  csrc/kernels/int4_w4a4_mma_sm120.cu \
  csrc/kernels/test_int4_w4a4_standalone.cu -o <out>/test_int4
python tools/patch_int4_omma_sm120.py <out>/test_int4 --verify --expect e2m1  # pre-patch gate (exit 0)
python tools/patch_int4_omma_sm120.py <out>/test_int4                          # flip to INT4
<out>/test_int4                                                                # runtime canary = authoritative
```

Expected output:

```text
[canary] 0  (E0M3 decode OK)
[gemv] N=4096 K=4096  cos(kernel, INT4-ref) = 0.9994  (expect > 0.999)
[bench] issue-rate: ~2013 TFLOPS  (INT4 x INT4)
[result] PASS
```

`cos(kernel, INT4-ref)` compares the kernel against a plain-C++
implementation of the identical INT4 two-level recipe, validating the SF
swizzle addressing and fragment layout (the residual is bf16-output
rounding vs the fp32 reference). Build with `arch=compute_120a` (not
`compute_120`) — plain `compute_120` rejects `mxf4nvf4`.

## llama.cpp / GGUF notes

The natural consumer is a 4-bit GGML backend on Blackwell: hardware
dequant happens inside the tensor core at the NVFP4 rate, replacing a
software Q4-dequant + BF16 GEMV.

- Quantize **from the original weights** to E0M3. Re-encoding an existing
  Q4_0 GGUF double-quantizes (cosine drops measurably) — GGML's per-32
  fp16 scale and our per-16 UE4M3 scale are different grids.
- GGML `Q4_0`: the single `−8` code clamps to our `±7` grid (≈ 3.6 % of
  codes on a sampled tensor, negligible); the per-32 scale maps onto two
  per-16 UE4M3 blocks exactly.
- GGML `Q4_K`: fold the per-block `min` into a per-row bias, then the
  symmetric per-16 scale maps directly onto UE4M3.

## Credits

The sm_120 `OMMA.SF` element-format bits were first documented publicly by
the **Ling Team** (author **@im0qianqian**). This work reproduces and
productizes that finding. Article (Chinese):
<https://zhuanlan.zhihu.com/p/2059376150565089368>

## References

- Ling Team, *Rethinking Shrinkage Bias in LLM FP4 Pretraining: Geometric
  Origin, Systemic Impact, and UFP4 Recipe* (arXiv). Motivates uniform
  4-bit grids over E2M1 and the Hadamard-rotation interaction.
- NVIDIA, *NVFP4* block-scaled FP4 format (E2M1 element + UE4M3 per-16
  scale factor), Blackwell tensor cores.
- Tseng et al., *QuIP#* / Hadamard-incoherence rotations for low-bit
  quantization (background for the per-16 rotation results above).
