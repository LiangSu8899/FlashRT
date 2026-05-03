# FlashRT — Docker

The fastest path to a working FlashRT install. One image, one
command, no CUTLASS clone, no `flash-attn` wheel-hunting, no manual
`cp *.so` step.

Two Dockerfiles ship with the repo:

| Hardware | Dockerfile | NGC base |
|----------|------------|----------|
| RTX 5090 / 4090 / 3090 / Ampere (x86_64)    | [`Dockerfile`](Dockerfile)         | `nvcr.io/nvidia/pytorch:25.10-py3` |
| Jetson AGX Thor (SM110, aarch64)            | [`Dockerfile.thor`](Dockerfile.thor) | `nvcr.io/nvidia/pytorch:25.09-py3` (arm64 manifest) |

Thor uses a hand-tuned cuBLAS-decomposed attention path
(`csrc/attention/fmha_dispatch.cu`) instead of the vendored
Flash-Attention 2, so its image deliberately does NOT produce
`flash_rt_fa2.so`. Everything else builds the same way. Skip to
[§5](#5-thor-jetson-agx-thor-sm110-aarch64) for the Thor flow.

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

---

## 5. Thor (Jetson AGX Thor, SM110, aarch64)

The Thor image uses a separate Dockerfile, [`Dockerfile.thor`](Dockerfile.thor),
because Thor pulls a different NGC manifest (`linux/arm64`) and skips
the FA2 build (Thor has its own attention path). Build on a Thor
host so `nvidia-smi` auto-detects `sm_110a`:

```bash
# On the Thor host
docker build -t flashrt:thor -f docker/Dockerfile.thor .

# Run (note --runtime=nvidia for Jetson — see below for why)
docker run --rm --gpus all -it --runtime=nvidia flashrt:thor
```

### Why `--runtime=nvidia` on Jetson

Unlike a discrete-GPU host (where `--gpus all` alone is enough — the
libnvidia-container shim auto-discovers `/dev/nvidia*` and the
matching driver libs), Jetson's iGPU stack is bound to host kernel
drivers and is exposed to containers through a **CSV-driven
mount mechanism** owned by `nvidia-container-runtime`:

```
/etc/nvidia-container-runtime/host-files-for-container.d/
├── devices.csv     # /dev/nvgpu, /dev/nvhost-*, /dev/nvmap, …
└── drivers.csv     # /usr/lib/aarch64-linux-gnu/tegra/libcuda.so.*, …
```

Passing `--runtime=nvidia` is what activates that runtime, which in
turn parses the two CSV files at container start and bind-mounts
every listed device node and driver library from the Tegra host
into the container. Without the flag the standard runc starts the
container without those mounts; the result is no `/dev/nvgpu`, no
`libcuda.so`, and `torch.cuda.is_available()` returns `False` even
though `nvidia-smi` works on the host.

`--gpus all` is left in the example for parity with the x86 docs and
because the libnvidia-container CLI hook ignores it gracefully on
Jetson, but the load-bearing flag here is `--runtime=nvidia`.

### What's different vs the x86 image

- **Base**: `nvcr.io/nvidia/pytorch:25.09-py3` (one minor older than the
  x86 image — 25.09 has the validated arm64 / Thor manifest, 25.10
  arm64 has not been smoke-tested on SM110 yet).
- **Build targets**: 4 `.so` files instead of 5
  (`flash_rt_kernels`, `flash_rt_fp4`, `libfmha_fp16_strided`,
  `flash_rt_jax_ffi`).
- **No `flash_rt_fa2.so`**: Thor's `csrc/attention/fmha_dispatch.cu`
  loads `libfmha_fp16_strided.so` at runtime via dlopen — no FA2
  template instantiation, ~10 min faster cold build than x86.
- **`flash_rt_fp4.so` on Thor**: built for sm_110a (NVFP4 instructions
  are SM120-only at the SASS level, but the kernel object compiles
  fine on Thor and the runtime dispatcher gates calls accordingly —
  see `docs/kernel_catalog.md` § FP4 path).

### Build args

Same as the x86 image (`GPU_ARCH`, `CUTLASS_REF`), minus `FA2_HDIMS`
which is a no-op on Thor.

### Smoke check

The image-build smoke deliberately asserts `libfmha_fp16_strided.so`
is present and does NOT import `flash_rt_fa2`, so a future regression
that reintroduces FA2 onto Thor by accident gets caught at build
time.

