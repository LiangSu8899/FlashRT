# FlashRT on AMD (ROCm / CDNA4)

Status: pi05 first target. The AMD backend is a self-contained tree — the
CUDA tree and root CMake are untouched, and the two never build together.

## Layout

| Path | Role |
|---|---|
| `csrc/amd/` | HIP kernels + hipBLASLt GemmRunner + pybind module `flash_rt_amd_kernels` (standalone CMake project) |
| `flash_rt/amd/core/` | `HipBuffer` / `HipGraph` — ctypes twins of the CUDA runtime seam over `libamdhip64` |
| `flash_rt/amd/models/pi05/pipeline.py` | Mirror of `pipeline_rtx.py` for CDNA4 (BF16 + FP8 branches) |
| `flash_rt/amd/frontends/torch/pi05.py` | `Pi05TorchFrontendAmd` — weights, quantization, prompt, capture driving |
| `flash_rt/amd/hardware/cdna4/attn_backend.py` | Attention backend (pointer-stable buffers; interim sdpa math) |

The pybind ABI is identical to the CUDA module: every kernel entry takes
`uintptr_t` device pointers plus a `uintptr_t` stream. Pipeline code is
portable text between the two trees.

## Requirements

- ROCm 7.x with hipBLASLt (`gfx950` validated; `GPU_ARCH` overridable)
- PyTorch ROCm build (allocation + interim attention; `torch.version.hip`)
- pybind11 (auto-installed to a local target dir by the build wrapper if
  the interpreter lacks it)

## Build

```bash
cmake -B build-amd -S csrc/amd [-DGPU_ARCH=gfx950]
cmake --build build-amd -j 8
# or, in cmake-less environments (falls back to a one-shot hipcc):
bash scripts/amd/build_amd.sh gfx950
```

Output: `flash_rt/amd/flash_rt_amd_kernels*.so`.

## Run (pi05)

```python
from flash_rt.amd.frontends.torch.pi05 import Pi05TorchFrontendAmd

fe = Pi05TorchFrontendAmd(checkpoint_dir, num_views=2, use_fp8=False)
fe.set_prompt(prompt_text, state=state)          # captures the HIP graph
result = fe.infer({"image": img, "wrist_image": wrist})
actions = result["actions"]                       # (chunk, 7) unnormalized
```

`use_fp8=True` (default) enables OCP FP8 E4M3 weights + activations for the
large GEMMs after calibration, mirroring the RTX recipe. The PaliGemma
tokenizer model is resolved as documented in
`flash_rt/utils/paligemma_tokenizer.py` (`FLASH_RT_PALIGEMMA_TOKENIZER`
override supported).

## HIP-vs-CUDA notes

- Wavefront is 64: reductions use `__shfl_down` wave64 helpers
  (`csrc/amd/kernels/common_hip.h`); there are no `*_sync` shuffle variants.
- The 3-arg graph instantiate is `hipGraphInstantiateWithFlags`.
- Memcpy-kind and capture-mode enums match CUDA numerically (validated on
  hardware by the runtime-seam smoke).
- FP8 storage is OCP `e4m3` (`__hip_fp8_e4m3`, `HIP_R_8F_E4M3`) — never the
  CDNA3 `fnuz` variant.
- hipBLASLt matmul is column-major; the GemmRunner uses the operand-swap
  form (`D_col = B_col @ A_col`), the same trick the CUDA FP8 paths use.
