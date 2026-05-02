# FlashRT — Docker

The fastest path to a working FlashRT install on a Linux x86_64 host
(RTX 5090 / 4090 / 3090 / Ampere). One image, one command, no CUTLASS
clone, no `flash-attn` wheel-hunting, no manual `cp *.so` step.

> **Thor (SM110)** is not covered by this image — Jetson is ARM64 and
> uses a different NVIDIA base (`nvcr.io/nvidia/l4t-jetpack`). Thor
> users follow the native install in [`docs/INSTALL.md`](../docs/INSTALL.md).

---

## 1. Pull the prebuilt image (recommended)

Once a release is tagged, the image is published to GitHub Container
Registry by CI:

```bash
docker pull ghcr.io/liangsu8899/flashrt:<tag>
docker run --rm --gpus all -it ghcr.io/liangsu8899/flashrt:<tag>
```

`<tag>` matches a release version (e.g. `0.2.0`) or `latest`.
Available tags: <https://github.com/LiangSu8899/FlashRT/pkgs/container/flashrt>.

### Modal / RunPod / Vast / cloud

```python
import modal

image = modal.Image.from_registry(
    "ghcr.io/liangsu8899/flashrt:0.2.0"
).pip_install("your-app-deps")

app = modal.App("flashrt-app", image=image)

@app.function(gpu="L40S")  # or H100, A100, etc.
def infer():
    import flash_rt
    model = flash_rt.load_model(checkpoint="/path/to/ckpt", framework="torch")
    ...
```

The image already has CUDA 13.0, PyTorch 2.9 with SM120 support, cuBLAS,
and the FlashRT kernels prebuilt — Modal cold-start is **dominated by
the pull (~30s on a warm CDN)** instead of a 10-minute kernel compile.

---

## 2. Build locally

If you want to pin a specific commit, target a different GPU, or
modify the kernels, build the image yourself:

```bash
# Default — auto-detects GPU arch via nvidia-smi (requires --gpus on build).
docker build -t flashrt:dev -f docker/Dockerfile .

# Pin to a specific arch (recommended for image distribution):
docker build -t flashrt:5090 \
    --build-arg GPU_ARCH=120 \
    -f docker/Dockerfile .

# Slim FA2 codegen for shipped models only (Pi0/Pi0.5/GROOT use 96 + 256):
docker build -t flashrt:slim \
    --build-arg GPU_ARCH=120 \
    --build-arg FA2_HDIMS="96;256" \
    -f docker/Dockerfile .
```

### Build args

| Arg | Default | When to set |
|---|---|---|
| `BASE_IMAGE` | `nvcr.io/nvidia/pytorch:25.10-py3` | Pin to an older NGC if your host CUDA driver is old. |
| `GPU_ARCH` | _(auto-detect)_ | Set when shipping the image to a different GPU than the build host. `120`=5090, `89`=4090, `86`=3090, `80`=A100. |
| `CUTLASS_REF` | `v4.4.2` | Bump if the upstream tag is yanked or you want to test a newer CUTLASS. |
| `FA2_HDIMS` | _(all of 96;128;256)_ | Drop unused head_dims to slim the image. Shipped models only need `96;256`. |

### Build time

Cold build (no NGC image cached): ~25 min, dominated by the FA2
template instantiation pass (~10 min) and the NGC pull (~10 min).
Warm build (NGC cached): ~12 min. With `FA2_ARCH_NATIVE_ONLY` and a
single-arch slim, the kernel compile drops to ~4 min.

---

## 3. Run

```bash
# Default: drops you in a Python REPL with `flash_rt` already imported.
docker run --rm --gpus all -it flashrt:dev

# Run the quickstart against a checkpoint mounted from the host:
docker run --rm --gpus all \
    -v /path/to/pi05_ckpt:/ckpt:ro \
    flashrt:dev \
    python3 examples/quickstart.py --checkpoint /ckpt --benchmark 20
```

---

## 4. What's inside

- Base: `nvcr.io/nvidia/pytorch:25.10-py3`
  (CUDA 13.0, PyTorch 2.9, cuBLASLt, nvcc, Python 3.12)
- CUTLASS 4.4.2 vendored at `/opt/cutlass`
- FlashRT source at `/workspace/FlashRT`, editable-installed
- All five kernel `.so` files prebuilt under `flash_rt/`:
  `flash_rt_kernels`, `flash_rt_fa2`, `flash_rt_fp4` (NVFP4-capable archs),
  `flash_rt_jax_ffi`, and `libfmha_fp16_strided` (Thor/Hopper only)
- An import smoke check runs at image-build time, so a broken image
  fails the `docker build` instead of the user's first pull

The image deliberately does **not** include the upstream `flash-attn`
pip wheel — the default RTX path uses the vendored `flash_rt_fa2.so`
and works without it. If you need legacy upstream attention or run
GROOT, install it yourself:

```bash
docker run --rm --gpus all flashrt:dev \
    pip install flash-attn  # or grab a prebuilt wheel from the releases page
```

