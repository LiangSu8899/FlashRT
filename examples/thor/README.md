# FlashRT on Jetson AGX Thor (SM110)

End-to-end Pi0.5 evaluation on Jetson AGX Thor. For the full install
guide (Docker / native, dependencies, CMake build of the kernel
library) see [`docs/INSTALL.md`](../../docs/INSTALL.md). This page
covers only the Thor-specific run path.

## Prerequisites

- Jetson AGX Thor (SM110) with JetPack / L4T
- CUDA 13.0+ toolkit (matches the NGC PyTorch container default)
- FlashRT installed and verified per [`docs/INSTALL.md`](../../docs/INSTALL.md)
  (you should have `flash_rt/flash_rt_kernels*.so` in place and
  `python -c "import flash_rt; print(flash_rt.__version__)"` works)

## Run E2E LIBERO evaluation

```bash
python examples/thor/eval_libero.py \
    --checkpoint /path/to/pi05_libero_pytorch \
    --task_suite libero_spatial
```

Expected (default `--num_views 2`):

```
============================================================
FlashRT Thor — Pi0.5 LIBERO Spatial
============================================================
[1/3] Loading model + weights         (~10 s)
[2/3] Calibrate FP8 + capture graph   (~3 s, then cached)
[3/3] Running 50 episodes...
============================================================
P50 latency:    ~44 ms (23 Hz)
LIBERO Spatial: 491/500 (98.2%)
============================================================
```

The first invocation calibrates FP8 activation scales and saves them
to `~/.flash_rt/calibration/`. Subsequent runs against the same
checkpoint + prompt length skip calibration automatically (~0.1 s).

## NVFP4 (optional)

Pi0.5 also supports NVFP4 encoder FFN on Thor, with the same E2E
latency floor and identical task accuracy. Enable with:

```bash
python examples/thor/eval_libero.py \
    --checkpoint /path/to/pi05_libero_pytorch \
    --use_fp4
```

See the README §NVFP4 section for the full latency/accuracy table
across 1/2/3 views.

## Troubleshooting

| Symptom | Likely fix |
|---|---|
| `No module named 'flash_rt_kernels'` | Build step skipped or non-editable install — see [`docs/INSTALL.md`](../../docs/INSTALL.md) §6 |
| First run slow (~30 s before benchmark) | Normal — FP8 calibration on first prompt length. Cached after. |
| `cuBLAS error code=13` when loading second model | Don't load multiple VLA checkpoints in one process; subprocess-isolate (Thor memory limit). |
| LIBERO score below 95% | Re-check the checkpoint format and `--task_suite` flag; report repro details if persistent. |

For deeper precision debugging, see [`docs/calibration.md`](../../docs/calibration.md) §4.
