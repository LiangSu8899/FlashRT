# FlashRT RTX 4090 (SM89) Deployment Guide

> End-to-end recipe for bringing up the Pi0 / Pi0.5 / GROOT pipelines on
> a fresh RTX 4090 machine, from Docker image to measured cosine +
> latency on real LIBERO frames. Software stack is the **same as the
> RTX 5090 build** — Ada SM89 is a supported target in CMake
> (`GPU_ARCH=89`) and cuBLASLt has FP8 E4M3 tensor cores on SM89.
>
> **Attention kernel is vendored.** Flash-Attention 2 v2.7.4.post1
> source is shipped under `csrc/attention/flash_attn_2_src/` and
> built into `flash_rt/flash_rt_fa2.so` during `cmake && make`.
> No `pip install flash-attn` needed — on 4090 the kernel uses
> `arch=compute_80,code=sm_80` AOT SASS (Ampere ISA is a strict
> subset of Ada), so the compiled `.so` runs natively without
> PTX JIT. No code changes required.

---

## Table of Contents

1. [Hardware + OS prerequisites](#1-hardware--os-prerequisites)
2. [Docker image + container setup](#2-docker-image--container-setup)
3. [Transfer the repo folder + datasets](#3-transfer-the-repo-folder--datasets)
4. [Build `flash_rt_kernels.so` (for SM89)](#4-build-flash_rt_kernelsso-for-sm89)
5. [Sanity checks](#5-sanity-checks)
6. [Cosine regression tests](#6-cosine-regression-tests)
7. [Latency benchmarks](#7-latency-benchmarks)
8. [Expected numbers vs 5090](#8-expected-numbers-vs-5090)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Hardware + OS prerequisites

| Component | 5090 reference (this work) | 4090 expected |
| --------- | -------------------------- | ------------- |
| GPU | RTX 5090 (SM120, Blackwell) | RTX 4090 (SM89, Ada Lovelace) |
| VRAM | 32 GB | 24 GB (sufficient for Pi0/Pi0.5/GROOT) |
| Host OS | Ubuntu 24.04 LTS (kernel 6.8) | Ubuntu 22.04 / 24.04, kernel ≥ 5.15 |
| NVIDIA driver | 580.82 | **≥ 545** (required for CUDA 13 toolkit container; 550.54.15+ for FP8 E4M3 + Hopper ISA) |
| nvidia-container-toolkit | 1.15+ | 1.15+ |

Verify on the 4090 box before anything else:

```bash
nvidia-smi                                    # sees RTX 4090, driver ≥ 545
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
#   → 8.9
docker info | grep -i runtime                 # needs nvidia runtime
```

---

## 2. Docker image + container setup

### 2a. Image

This work uses **`flashrt-rtx-x86`** (custom image built on top of
`nvcr.io/nvidia/pytorch:25.10-py3`). The image bundles:

- CUDA 13.0.2 toolkit + cuBLASLt (ships FP8 E4M3 matmul)
- PyTorch `2.9.0a0+145a3a7bda.nv25.10`
- JAX `0.5.3` + `jax-cuda12-pjrt/plugin 0.5.3`
- `torch-tensorrt`, `torch-xla 2.9.0`
- `ml_dtypes 0.5.3`, `numpy 2.2.6`, `pandas 3.0.1`, `pillow 12.0.0`,
  `safetensors 0.7.0`
- `pybind11 3.0.1`, `cmake 3.31.6`
- `flash_attn 2.7.4.post1+25.10` (optional — left installed only as
  a debug fallback path via `FVK_RTX_FA2=0`; default runs use the
  vendored in-SO FA2 and have zero `flash-attn` wheel dependency)

`TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 9.0 10.0 12.0+PTX` is set at image
level — for 4090 add `8.9` before building.

### 2b. Transfer or rebuild the image on 4090

**Option A (recommended) — save/load the existing image** (fastest if
you already have `flashrt-rtx-x86` on the 5090 box):

```bash
# On 5090 host
docker save flashrt-rtx-x86 | gzip > flashrt-rtx-x86.tar.gz
scp flashrt-rtx-x86.tar.gz user@4090-host:/tmp/

# On 4090 host
docker load < /tmp/flashrt-rtx-x86.tar.gz
```

**Option B — rebuild from NGC base** (if the Dockerfile is available
or you prefer a clean build):

```dockerfile
# Dockerfile skeleton — extend as needed
FROM nvcr.io/nvidia/pytorch:25.10-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git libegl1 && \
    rm -rf /var/lib/apt/lists/*

# Python deps pinned to match 5090 build
RUN pip install --no-cache-dir \
      jax==0.5.3 jax-cuda12-pjrt==0.5.3 jax-cuda12-plugin==0.5.3 \
      ml_dtypes==0.5.3 numpy==2.2.6 pandas==3.0.1 pillow==12.0.0 \
      safetensors==0.7.0 pybind11==3.0.1

# NOTE: `pip install flash-attn` is NOT required any more. FlashRT
# vendors the FA2 fp16 + bf16 fwd kernels at source level under
# csrc/attention/flash_attn_2_src/ and builds them into
# flash_rt/flash_rt_fa2.so during `cmake && make`. The runtime
# imports `from flash_rt import flash_rt_fa2` directly, so there is
# no dependency on the pip `flash-attn` wheel and no need to match
# the wheel's torch × CUDA × driver × glibc compatibility matrix.
#
# Install flash-attn below only if you want the FVK_RTX_FA2=0 fallback
# for A/B debugging against the pip wheel:
# RUN pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# openpi reference model (for cosine tests against PyTorch FP32 ref)
# openpi-client is on PyPI; openpi (src) comes from the openpi repo.
RUN pip install --no-cache-dir openpi-client
# COPY or git clone openpi source separately — see §3.

ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 10.0 12.0+PTX"
ENV CUDA_MODULE_LOADING=LAZY
WORKDIR /workspace
```

Note: add `8.9` to `TORCH_CUDA_ARCH_LIST` for Ada Lovelace tensor-core
codegen. The 5090 image has `12.0+PTX` which would JIT for 4090 but
native AOT compile is preferred.

### 2c. Launch the container

```bash
docker run -d --gpus all --ipc=host --network=host --name pi0-4090 \
  -v /path/on/4090/workspace:/workspace \
  flashrt-rtx-x86 sleep infinity
```

---

## 3. Transfer the repo folder + datasets

### 3a. Copy the repo folder

A full copy of the repo is sufficient. Keep the folder path consistent
between the 5090 box and the 4090 host so any in-container paths match
(some test scripts hardcode paths — see §3c).

```bash
# From the source machine
rsync -avz --exclude='build/' --exclude='__pycache__/' \
      --exclude='*.so' --exclude='.git' \
      /path/to/FlashRT/ \
      user@4090-host:/path/on/4090/FlashRT/
```

Excluded:
- `build/` — CMake cache is host-specific, will be regenerated
- `*.so` — 5090 `sm_120` binary is incompatible with 4090
- `__pycache__/` — regenerates on first run

### 3b. Copy data assets

Three datasets are required:

| Path | Purpose | Size |
| ---- | ------- | ---- |
| `<ckpts>/pi0_base` | Orbax JAX ckpt (for JAX frontend) | 12 GB |
| `<ckpts>/pi0_base_pytorch` | PyTorch safetensors ckpt + `assets/physical-intelligence/libero/norm_stats.json` | 6.6 GB |
| `<openpi-compiler>/RL/data/libero_rollouts/` | Real LIBERO frames (images, state, action) | 1.2 GB |

```bash
rsync -avz --progress <ckpts>/pi0_base \
      user@4090-host:<ckpts>/

rsync -avz --progress <ckpts>/pi0_base_pytorch \
      user@4090-host:<ckpts>/

rsync -avz --progress \
      <openpi-compiler>/RL/data/libero_rollouts \
      user@4090-host:<openpi-compiler>/RL/data/
```

For Pi0.5 / GROOT you will also need:

| Path | Purpose |
| ---- | ------- |
| `<ckpts>/pi05_libero_pytorch_migrated` | Pi0.5 LIBERO safetensors |
| `<ckpts>/pi05_base` | Pi0.5 base Orbax |
| `<ckpts>/GR00T-N1.6-3B` | GROOT N1.6 HuggingFace bundle |

### 3c. Also copy the openpi reference package

`rtx_pi0_cosine_vs_official.py` imports `openpi.models.pi0_config` +
`openpi.models_pytorch.pi0_pytorch` to generate the PyTorch FP32
reference. Make sure `<openpi_repo>/src` is in `PYTHONPATH`:

```bash
# 5090 had: PYTHONPATH=<openpi_repo>/src:
rsync -avz <openpi_repo> user@4090-host:<remote_root>/
```

If `PYTHONPATH` isn't set in the container, `export` it:

```bash
export PYTHONPATH=<openpi_repo>/src:$PYTHONPATH
```

---

## 4. Build `flash_rt_kernels.so` (for SM89)

CMake auto-detects the local GPU's compute capability via
`nvidia-smi`, so on a 4090 it will default to `GPU_ARCH=89`. No flag
needed.

```bash
docker exec -it pi0-4090 bash
cd <repo_root>

mkdir -p build && cd build
cmake ..            # check output: "Auto-detected GPU arch: sm_89"
make -j$(nproc)     # ~3-5 min on 4090
```

Expected CMake output on 4090:
```
-- Auto-detected GPU arch: sm_89
-- SM100 CUTLASS FP8: DISABLED (sm_89)             ← expected, SM89 has no CUTLASS SM100
-- NVFP4/W4A8 support: DISABLED (requires sm_120a)  ← expected, 4090 is Ada not Blackwell
-- FA2 in-SO attention: ENABLED (sm_89)             ← expected, vendored FA2 built for SM80-family
-- FA2 vendor object library: building for sm_80 + sm_120 + PTX fallback
-- FA2 pybind module: flash_rt_fa2 (separate .so)
```

Build time on 4090 (CUDA 13 NGC container): ~10–12 min total
(main kernels ~2 min + FA2 vendor ~8–10 min; FA2's CUTLASS 3.x
templates dominate). Subsequent rebuilds of only the main kernels
take ~2 min because FA2 is a separate CMake target.

The SM89 build uses:
- `fp8_gemm_descale_fp16` → cuBLASLt (works on SM89+)
- FP16/BF16 templated kernels (norm, activation, residual) — all support SM80+
- **In-SO Flash-Attention 2** (fp16 + bf16) via `flash_rt_fa2.so`.
  Runs `arch=compute_80,code=sm_80` SASS natively on SM89 (Ampere
  ISA is a strict subset of Ada) — no PTX JIT, no pip `flash-attn`
  wheel required.
- **Does not include** `cutlass_fp8_sq/t1/wide` (SM100-only) — fine,
  RTX path uses cuBLASLt

Install both `.so` files next to the Python package:

```bash
cp build/flash_rt_kernels.cpython-312-x86_64-linux-gnu.so \
   flash_rt/flash_rt_kernels.cpython-312-x86_64-linux-gnu.so
cp build/flash_rt_fa2.cpython-312-x86_64-linux-gnu.so \
   flash_rt/flash_rt_fa2.cpython-312-x86_64-linux-gnu.so
```

Verify the bindings are loadable:

```bash
python -c "from flash_rt import flash_rt_kernels as fvk; \
           from flash_rt import flash_rt_fa2     as fa2; \
           print('gate_geglu_fp16:',   hasattr(fvk, 'gate_geglu_fp16')); \
           print('qkv_split_fp16:',       hasattr(fvk, 'qkv_split_fp16')); \
           print('fp8_gemm_descale_fp16:', hasattr(fvk, 'fp8_gemm_descale_fp16')); \
           print('has_cutlass_sm100:',    fvk.has_cutlass_sm100()); \
           print('fa2.fwd_fp16:',         callable(fa2.fwd_fp16)); \
           print('fa2.fwd_bf16:',         callable(fa2.fwd_bf16))"
```

Expected:
```
gate_geglu_fp16: True
qkv_split_fp16: True
fp8_gemm_descale_fp16: True
has_cutlass_sm100: False
fa2.fwd_fp16: True
fa2.fwd_bf16: True
```

---

## 5. Sanity checks

### 5a. Hardware auto-dispatch picks the RTX frontend

```bash
python -c "
from flash_rt.hardware import detect_arch, _PIPELINE_MAP
arch = detect_arch()
print('detected arch:', arch)
print('pi0 torch dispatch:', _PIPELINE_MAP[('pi0', 'torch', arch)])
print('pi0 jax dispatch:  ', _PIPELINE_MAP[('pi0', 'jax',   arch)])
"
```

Expected:
```
detected arch: rtx_sm89
pi0 torch dispatch: ('flash_rt.frontends.torch.pi0_rtx', 'Pi0TorchFrontendRtx')
pi0 jax dispatch:   ('flash_rt.frontends.jax.pi0_rtx',   'Pi0JaxFrontendRtx')
```

If this prints `rtx_sm89` and resolves to the same classes as 5090,
the whole codepath is good.

### 5b. Smoke-import every frontend

```bash
python -c "
from flash_rt.frontends.torch.pi0_rtx   import Pi0TorchFrontendRtx
from flash_rt.frontends.jax.pi0_rtx     import Pi0JaxFrontendRtx
from flash_rt.frontends.torch.pi05_rtx  import Pi05TorchFrontendRtx
from flash_rt.frontends.jax.pi05_rtx    import Pi05JaxFrontendRtx
from flash_rt.frontends.torch.groot_rtx import GrootTorchFrontendRtx
print('all imports OK')
"
```

---

## 6. Cosine regression tests

All tests use **real LIBERO frame 50** for FP8 calibration — random
inputs produce unrepresentative scales (the activation amax distribution
of random data does not match the trained model's, so FP8 scales clip
real frames). Expected cos targets are the same as 5090 since the FP8
kernels are algorithmically identical.

For each of the four RTX paths below, the validation pattern is the
same: load the model via `flash_rt.load_model(...)`, run `predict()`
with a matched-noise observation, and compare the output against your
PyTorch FP32 reference run on the same inputs.

### 6a. Pi0.5 torch RTX (shipped baseline)
Target: per-action cos ≥ 0.999 vs FP32 reference.

### 6b. Pi0 torch RTX
Target: cos ≥ 0.997 vs FP32 reference. Typical value on 5090:
0.9982. Expect the same on 4090 (±0.0005 noise).

The first cosine run typically caches the PyTorch FP32 and FP16
references at `/tmp/pi0_libero_f50_{fp32,fp16}.npy` (~12 GB GPU
usage); subsequent runs reuse the cache.

### 6c. Pi0 JAX RTX (Orbax)
Targets: jax cos vs FP32 ≥ 0.997, jax vs torch ≥ 0.998.
Typical: jax_vs_fp32 = 0.9984, jax_vs_torch = 0.9990.

### 6d. GROOT torch RTX
Targets: pass per-embodiment cosine thresholds (see
`flash_rt.models.groot.embodiments`). Requires the HuggingFace
Isaac-GR00T checkpoint locally available.

If `transformers` complains about `VideoInput`, pin a compatible
transformers version (this is a pre-existing upstream issue, not a
regression):

```bash
pip install "transformers<4.56" --upgrade
```

---

## 7. Latency benchmarks

### 7a. Smoke + latency per frontend

The standard pattern: load → first predict (warm) → 10 warmup
replays → 50 timed replays via `cuda.Event` around
`model._pipe._enc_ae_graph.replay()` (see README §Reproducing for
the snippet). Repeat for both torch and jax frontends.

### 7b. Sweep torch/jax × 1/2/3 views (the README-style table)

Loop the 7a pattern over `framework ∈ {torch, jax}` and
`num_views ∈ {1, 2, 3}` (six configs). Use 100 warmup + 200 timed
replays per cell. Lock GPU clocks per §7c below before measuring.

Before long benchmarks, lock the GPU to its default clock so numbers
don't vary with thermals:

```bash
# On host (outside container), requires root:
sudo nvidia-smi -i 0 -pm 1                # persistence mode
sudo nvidia-smi -i 0 -lgc 2505,2520       # 4090 boost clock range
# After measuring:
sudo nvidia-smi -i 0 -rgc                  # reset gpu clocks
```

---

## 8. Expected numbers vs 5090

The **cosine values should match 5090 within noise** (±0.0005) since
FP8 kernels and pipeline are algorithmically identical. **Latency on
4090 is expected to be higher** because Ada FP8 tensor-core throughput
is ~0.5× Blackwell and memory bandwidth is 1 TB/s vs 1.8 TB/s.

| Metric | 5090 (measured) | 4090 (estimated) |
| ------ | --------------- | ---------------- |
| Pi0 full FP8 vs FP32 ref (cos) | 0.9982 | 0.9980–0.9985 (±noise) |
| Pi0 full FP16 vs FP32 ref (cos) | 0.9997 | 0.9997 (same) |
| Pi0 p50 latency @ 1v | 18.4 ms | **est. 30–38 ms** |
| Pi0 p50 latency @ 2v | 21.2 ms | **est. 34–44 ms** |
| Pi0 p50 latency @ 3v | 24.5 ms | **est. 40–52 ms** |

The range reflects uncertainty about cuBLASLt FP8 kernel selection on
Ada — first run with an empty cuBLASLt heuristic cache can autotune to
a slow algo. The 100-warmup loop in `rtx_pi0_bench_views.py` is enough
to stabilise, but you may see p50 drift down ~5% after ~500 total
infers.

**Record actual numbers on the 4090 and update the README table
row for `Pi0 RTX 4090 (SM89)` once validated.**

---

## 9. Troubleshooting

### `fp8_gemm_descale_fp16: False` after build
cuBLASLt FP8 GEMM section got compiled out. Check CMake log for
`cuBLASLt`-related errors. The 25.10 PyTorch container includes
cuBLASLt 13.x which supports FP8 on SM89+. Older CUDA 11 containers
have cuBLASLt 11 without FP8 support — upgrade the base image.

### `ImportError: libcudart.so.13` on import
Host driver < 545 can't run CUDA 13 containers. Either upgrade the
driver or rebuild the image against `nvcr.io/nvidia/pytorch:24.07-py3`
(CUDA 12.6, compatible with driver 535+).

### `ImportError: cannot import name 'flash_rt_fa2' from 'flash_rt'`
You skipped the FA2 `.so` install step. The main kernel build
produces TWO `.so` files on RTX targets — copy both:
```bash
cp build/flash_rt_kernels*.so flash_rt/
cp build/flash_rt_fa2*.so     flash_rt/
```
If `build/flash_rt_fa2*.so` does not exist, check that CMake
printed `FA2 in-SO attention: ENABLED (sm_89)` during `cmake ..`
— if it said DISABLED, the detected GPU arch is wrong (`cmake ..
-DGPU_ARCH=89` forces SM89).

### FA2 build takes too long
Default cold build on a 5090 is ~4.5 min (CUTLASS 3.x template-heavy
.cu files dominate). Subsequent rebuilds of only the main kernels
take ~2 min because FA2 is a separate CMake target — it relinks but
does not recompile unless vendored source changes.

If you're iterating on a single 5090/4090 and only running one model
family, use the slim-build flags to cut cold FA2 cost to ~1.5 min:

```bash
cmake .. \
  -DFA2_ARCH_NATIVE_ONLY=ON \       # skip sm_80 + PTX fallback (-59%)
  -DFA2_HDIMS="96;256" \            # drop hdim=128 (-21%)
  -DFA2_DTYPES="fp16"               # drop bf16, Pi0-only (-33%)
# combined: 267 s -> 87 s (-67%); .so: 135 MB -> 17.8 MB (-87%)
```

Defaults emit a cross-arch-compatible .so (sm_80 + sm_120 AOT +
compute_120 PTX) that works on any RTX card from a single build.
Only use the slim flags if you know you don't need that.

Alternatively, skip FA2 entirely and fall back to the legacy
`pip flash-attn` wheel (for A/B debugging), install the wheel and
set `FVK_RTX_FA2=0`:
```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
FVK_RTX_FA2=0 python <your-cosine-test>.py
```

See [README § Slim-build flags](../README.md#slim-build-flags-developer-iteration-speed)
for the full flag table and measured savings.

### Cosine catastrophically low (~0.1)
You almost certainly have `*.so` from the 5090 build (sm_120) loaded
on the 4090. Remove, rebuild, re-install:
```bash
rm -f flash_rt/flash_rt_kernels*.so flash_rt/flash_rt_fa2*.so \
      build/flash_rt_kernels*.so     build/flash_rt_fa2*.so
cd build && rm CMakeCache.txt && cmake .. && make -j && cd ..
cp build/flash_rt_kernels.*.so build/flash_rt_fa2.*.so flash_rt/
```

### Cosine is 0.91, per-action gradient 0.72 → 0.95
You are running the test with `torch.randn` inputs. Fix: make sure
you're using the libero-data version of the cosine tests (the ones
currently shipped — verify the top docstring mentions "real LIBERO
frame"). Random inputs produce out-of-distribution activation
statistics that break FP8 calibration.

### `norm_stats.json not found`
The JAX frontend falls back to a sibling `<name>_pytorch` directory
when the Orbax checkpoint lacks `assets/physical-intelligence/libero/`.
Make sure both `pi0_base/` and `pi0_base_pytorch/` are present on the
4090 box.

### High variance in latency (±2 ms between runs)
GPU clocks are boosting up/down. Lock them (§7b). Also check for:
- thermal throttling (4090 TDP 450 W under full load)
- MIG mode active (check `nvidia-smi -q | grep MIG`)
- Other processes sharing the GPU

---

## Appendix — quick-fire commands (paste-and-go)

```bash
# Assumes the container is running and data is at <your_data_root>/...
docker exec -it pi0-4090 bash

cd <repo_root>
export PYTHONPATH=<openpi_repo>/src:$PYTHONPATH

# 1. Build (produces flash_rt_kernels.so + flash_rt_fa2.so on RTX)
cd build && cmake .. && make -j$(nproc) && cd ..
cp build/flash_rt_kernels.cpython-312-*.so flash_rt/
cp build/flash_rt_fa2.cpython-312-*.so     flash_rt/

# 2. Verify bindings
python -c "from flash_rt import flash_rt_kernels as fvk; \
           from flash_rt import flash_rt_fa2     as fa2; \
  print('gate_geglu_fp16:', hasattr(fvk, 'gate_geglu_fp16')); \
  print('has_cutlass_sm100:',  fvk.has_cutlass_sm100()); \
  print('fa2.fwd_fp16/bf16:',  callable(fa2.fwd_fp16) and callable(fa2.fwd_bf16))"

# 3. Cosine regression for the rtx models (Pi0.5 / Pi0 torch / Pi0 jax)
#    Pattern: load → predict() with matched-noise observation → cosine
#    vs your saved PyTorch FP32 reference for that model.
#    Targets per §6 above.

# 4. Latency sweep — see §7b (torch/jax × 1/2/3 views CUDA-graph replay)

# 5. Inspect results, update README table with 4090-measured numbers.
```
