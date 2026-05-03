# FlashRT on RTX 5090 (Blackwell, SM120)

## Prerequisites

- RTX 5090 or other Blackwell GPU
- CUDA Toolkit 13.0+
- Python 3.10+, PyTorch 2.x

## Step 1: Install Dependencies

For the full install path (Docker / native, CMake build) see
[`docs/INSTALL.md`](../../docs/INSTALL.md). Minimum:

```bash
pip install torch pybind11 pyyaml numpy safetensors
# NOTE: pip install flash-attn is NOT required — FlashRT vendors
# Flash-Attention 2 source and builds it into flash_rt_fa2.so during
# cmake. See README §Build for details.
```

## Step 2: Download CUTLASS and Build

```bash
cd FlashRT

# CUTLASS (header-only)
git clone --depth 1 --branch v4.4.2 \
  https://github.com/NVIDIA/cutlass.git third_party/cutlass

# Build flash_rt_kernels module
mkdir build && cd build
cmake ..              # auto-detects SM120
make -j$(nproc)
make install          # installs .so → flash_rt/
cd ..
```

Verify:

```bash
python -c "import flash_rt; print(flash_rt.__version__)"
```

## Step 3: Download Checkpoint

```bash
# Pi0.5 LIBERO checkpoint (requires OpenPI access)
python -c "from openpi.models import download; download('pi05_libero_pytorch')"
```

## Step 4: Run Evaluation

```bash
python examples/blackwell/eval_libero.py \
  --checkpoint /path/to/pi05_libero_pytorch \
  --task_suite libero_spatial
```

## Performance

| Views | Latency | Frequency |
|-------|---------|-----------|
| 1 view | 14.79 ms | 67.6 Hz |
| 2 views | 17.88 ms | 55.9 Hz |
| 3 views | 20.56 ms | 48.6 Hz |

Cosine vs PyTorch: 0.999638

## Troubleshooting

**`ModuleNotFoundError: flash_rt_fa2`**: build step skipped or `cp *.so ../flash_rt/` not run after `make`. See [`docs/INSTALL.md`](../../docs/INSTALL.md) §6.

**`CMake Error: pybind11 not found`**: Run `pip install pybind11`.

**`ENABLE_NVFP4: DISABLED`**: Expected if GPU is not SM120. NVFP4 is optional (FP8 is the primary path).
