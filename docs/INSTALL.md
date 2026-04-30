# Installing FlashVLA

One-page install guide. Picks up where the README leaves off and
covers the details the README keeps short to stay readable.

For the full build overview (what .so files are produced, which arch
enables which kernels), read the "Build" section of the top-level
[README](../README.md) first.

---

## 1. Two supported paths

| Path | When to use | Entry point |
|---|---|---|
| **Docker** | You want a known-good CUDA/PyTorch toolchain and don't care about host Python | [README §Option A](../README.md) |
| **Native Linux** | You already run CUDA workloads on the host and want the library in your existing venv | [README §Option B](../README.md) |

Both paths end at the same verification step — `import flash_vla;
flash_vla.__version__` returns the installed version, and
`flash_vla.flash_vla_kernels` is importable.

---

## 2. Prerequisites (native path)

| Component | Minimum | Notes |
|---|---|---|
| GPU | SM80+ | A100 / RTX 30-series / 40-series / Thor / 5090. Pre-SM80 (V100, 20-series) is unsupported — FA2 vendored code requires Ampere. |
| NVIDIA driver | 525+ (CUDA 12.4) / 545+ (CUDA 13) | 5090 needs 550+ |
| CUDA Toolkit | 12.4+ on Thor/Ada/Hopper, 12.8+ on Blackwell | CUDA 13 is the NGC-image default |
| Python | 3.10 / 3.11 / 3.12 | One venv; the interpreter that runs `cmake` MUST match the interpreter that later imports `flash_vla` |
| GCC / G++ | 11+ (C++17) | |
| CMake | 3.24+ | |

## 3. Python environment

**Always use a fresh venv or conda env.** The build step resolves
`pybind11` via `python3 -m pybind11 --cmakedir`, and the `.so` files
ship with an ABI tag tied to the interpreter they were compiled
against. Mixing a system Python at build time with a conda Python at
import time is the #1 native-install failure mode.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

## 4. CUTLASS dependency

FlashVLA's main FP8/FP4 GEMM path is built against **CUTLASS 4.x**, not
bundled in the repo to keep clone size small. Clone it before
running `cmake`:

```bash
git clone --depth 1 --branch v4.4.2 \
    https://github.com/NVIDIA/cutlass.git third_party/cutlass
```

CMake now fails with a clear message if this step is missing (see
`CMakeLists.txt` near the top of the "Paths" section).

> **Note**: FA2 uses a vendored CUTLASS 3.x under
> `csrc/attention/flash_attn_2_src/`. That one IS checked in — only
> the CUTLASS 4.x for the main kernels needs a manual clone.

## 5. Editable install is required

```bash
pip install -e ".[torch]"       # or "[jax]" / "[all]"
```

`-e` is not optional. The CMake build drops compiled `.so` files into
the `flash_vla/` source tree; only editable install makes that
directory importable without an extra copy step. A plain
`pip install .` would snapshot `flash_vla/` BEFORE the kernels are
built, and `import flash_vla` later would fail with a missing
`flash_vla_kernels` error.

## 6. Build

```bash
mkdir build && cd build
cmake ..                             # auto-detects GPU arch via nvidia-smi
# Or override: cmake .. -DGPU_ARCH=110   (110=Thor, 120=5090, 89=4090, 80=A100)
make -j$(nproc)
cp flash_vla*.so ../flash_vla/
# libfmha_fp16_strided.so is built only on Thor / Hopper (SM100+);
# on RTX this glob is empty and the cp below is a no-op.
cp libfmha*.so ../flash_vla/ 2>/dev/null || true
cd ..
```

Per-arch produced shared libraries:

| Target  | `flash_vla_kernels.so` | `flash_vla_fp4.so` | `flash_vla_fa2.so` | `libfmha_fp16_strided.so` |
|---------|:----------------------:|:------------------:|:------------------:|:-------------------------:|
| Thor (SM110) | ✅ | ✅ | — | ✅ (SigLIP fast path) |
| Hopper (SM100) | ✅ | ✅ | — | ✅ |
| RTX 5090 (SM120) | ✅ | ✅ | ✅ (in-SO FA2) | — |
| RTX 4090 (SM89) | ✅ | — | ✅ (in-SO FA2) | — |

### 6.1 Building on CUDA < 12.8

The default vendor build of Flash-Attention 2 emits a ``compute_120``
PTX fallback alongside the per-arch SASS so a single ``.so`` covers
all listed gencodes — including Blackwell (SM120) targets that need
CUDA 12.8+. On older toolchains (e.g. an L40S running a CUDA-12.4
image) ``nvcc`` rejects the ``compute_120`` PTX target with a
``Value 'compute_120' is not defined`` error and the build aborts.

If you only need a binary for the GPU detected on the build host
(typical for cloud / self-hosted users that aren't shipping the
``.so`` to a different arch), set ``FA2_ARCH_NATIVE_ONLY=ON`` to
skip the cross-arch SASS + PTX fallback. The build emits SASS for
the current arch only, runs ~66 % faster, and works on any CUDA
toolchain that supports that arch:

```bash
cmake -B build -S . -DFA2_ARCH_NATIVE_ONLY=ON
cmake --build build -j$(nproc)
```

## 7. Verify

```bash
python -c "
import flash_vla, torch, numpy
print('flash_vla:', flash_vla.__version__)
print('torch    :', torch.__version__, torch.cuda.get_device_capability())
print('numpy    :', numpy.__version__)
from flash_vla import flash_vla_kernels
print('kernels CUTLASS SM100:', flash_vla_kernels.has_cutlass_sm100())
"
```

Expected (Thor example):
```
flash_vla: 0.1.0
torch    : 2.9.0+cu124 (11, 0)
numpy    : 1.26.x
kernels CUTLASS SM100: True
```

If `import flash_vla` fails with "no module named flash_vla_kernels",
either (a) `make` didn't produce the `.so`, or (b) you forgot
`cp *.so ../flash_vla/`, or (c) you installed non-editable and the
import is hitting a stale site-packages copy. Check in order.

## 8. JAX frontend (optional)

The JAX path uses a specific Orbax / jaxlib / PJRT plugin combo. Pins
below are what we test against — don't upgrade one without the others:

```bash
pip install jax==0.5.3 jax-cuda12-pjrt==0.5.3 jax-cuda12-plugin==0.5.3 ml_dtypes==0.5.3 orbax-checkpoint flax
```

Upgrade path (tracked, not yet done):

- jax 0.6+ needs the `jax-cuda12-plugin` name to stay aligned (no
  rename expected but verify); check the PJRT plugin registers
  cleanly with `python -c "import jax; jax.devices()"`.
- Orbax 0.6+ changed the default metadata layout for `StandardRestore`;
  our `load_from_cache` path in `flash_vla/frontends/jax/` expects
  the 0.5.x layout.

## 9. `transformers` version constraint

`transformers<4.56` is pinned because the Pi0.5 PaliGemma tokenizer
was broken by internal refactors in 4.56+. This affects ONLY the
Pi0.5 torch frontend; Pi0 / GROOT / Pi0-FAST are unaffected. Plan
is to upgrade the pin once we port the tokenizer call-site.

## 10. Checkpoints

FlashVLA does not bundle model weights. Bring your own Pi0 / Pi0.5 /
GROOT checkpoint in whichever format your trainer produced:

- `safetensors` (HuggingFace / PyTorch format) — used by the torch
  frontends
- Orbax (JAX native) — used by the JAX frontends

See [USAGE.md](../USAGE.md) §Loading a model for the per-frontend
`load_model` call.

## 11. Troubleshooting quick reference

| Symptom | Likely cause |
|---|---|
| `CMake Error ... CUTLASS headers not found` | Step 4 skipped |
| `No module named 'flash_vla_kernels'` | Step 6's `cp *.so` step skipped, OR non-editable install |
| `PJRT plugin ... not found` at JAX import | JAX / jax-cuda12-plugin version mismatch (Step 8) |
| `cuBLAS error code=13` when loading second model | Ran two model loads in one process; subprocess-isolate per model |
| cos regression right after calibrate | `act_scale * weight_scale` alpha computed in f64 somewhere; see `docs/calibration.md` §2.3 |
