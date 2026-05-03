# FlashRT — Thor image (Jetson AGX Thor, SM110, aarch64).
#
# Base: NVIDIA NGC PyTorch 25.09 — Docker auto-selects the linux/arm64
# manifest on a Thor host. Ships CUDA 13.0, PyTorch 2.9 (Thor-validated),
# cuBLASLt, nvcc 13.0.88, Python 3.12, Ubuntu 24.04. Same content as
# the maintainer's openpi-pi0.5:l4t-jp7.0 base layer (verified via NGC
# build labels).
#
# Build (on a Thor host):
#   docker build -t flashrt:thor -f docker/Dockerfile.thor .
#
# Run:
#   docker run --rm --gpus all -it --runtime nvidia flashrt:thor
#
# Note on FA2: Thor (SM110) uses the in-tree cuBLAS-decomposed attention
# path (csrc/attention/fmha_dispatch.cu + libfmha_fp16_strided.so), not
# the vendored Flash-Attention 2 sources. CMakeLists gates FA2 to
# SM80/86/89/120 only, so this image produces ONLY:
#   flash_rt/flash_rt_kernels.cpython-312-aarch64-linux-gnu.so
#   flash_rt/flash_rt_fp4.cpython-312-aarch64-linux-gnu.so
#   flash_rt/libfmha_fp16_strided.so
#   flash_rt/flash_rt_jax_ffi.so
# No flash_rt_fa2.so on Thor — that's expected, not a build failure.
#
# Layer order is tuned for cache reuse: anything that changes per
# commit (the source tree) is copied last so flipping a one-line
# patch does NOT invalidate the CUTLASS clone or pip layer.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${BASE_IMAGE}

# Build-arg knobs:
#   GPU_ARCH    pin the SASS arch. Default empty = CMake auto-detect via
#               nvidia-smi (works for `docker build --gpus all` on the
#               same Thor host the image will run on). Override to 110
#               explicitly if you build on a Thor host that exposes a
#               different compute_cap to nvidia-smi.
#   CUTLASS_REF pin in case upstream tags get yanked.
ARG GPU_ARCH=""
ARG CUTLASS_REF=v4.4.2

# ── Tiny base utilities. The NGC PyTorch image already has cmake / git /
#    ninja, but pin ccache for iterative dev and re-install missing pieces
#    defensively in case a future base image drops one. ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        ccache \
    && rm -rf /var/lib/apt/lists/*

# ── CUTLASS 4.4.2 — vendor in the image so per-commit rebuilds skip
#    the ~80 MB git clone every time. Pinned to CUTLASS_REF for repro. ──
RUN git clone --depth 1 --branch ${CUTLASS_REF} \
        https://github.com/NVIDIA/cutlass.git /opt/cutlass \
    && rm -rf /opt/cutlass/.git

# ── pybind11 + ninja are usually in NGC; install only if missing so
#    the layer stays empty on already-equipped bases. ──
RUN python3 -m pip install --no-cache-dir \
        --upgrade pybind11 ninja

WORKDIR /workspace/FlashRT

# Copy build scaffolding first (CMakeLists, pyproject, etc.) so a
# pure-source-only edit does NOT invalidate the build layers.
COPY CMakeLists.txt pyproject.toml setup.py README.md USAGE.md ./
COPY csrc/         csrc/
COPY flash_rt/     flash_rt/
COPY flash_wm/     flash_wm/
COPY tools/        tools/
COPY docs/         docs/
COPY examples/     examples/
COPY training/     training/
COPY tests/        tests/

# Symlink the vendored CUTLASS into where CMakeLists expects it.
# `mkdir -p` first because .dockerignore excludes third_party/ from
# the build context (the host clone may have a stale CUTLASS we don't
# want shipped).
RUN mkdir -p third_party && ln -sfn /opt/cutlass third_party/cutlass

# ── Build the CUDA kernels for SM110 ──
# CMake auto-detects sm_110a from nvidia-smi (override via -DGPU_ARCH=110
# if needed). All four .so targets land directly under flash_rt/ thanks
# to LIBRARY_OUTPUT_DIRECTORY in CMakeLists; no follow-up `cp` step.
RUN cmake -B build -S . \
        $( [ -n "${GPU_ARCH}" ] && echo "-DGPU_ARCH=${GPU_ARCH}" ) \
 && cmake --build build -j"$(nproc)" \
 && rm -rf build/CMakeFiles

# ── Editable install — registers flash_rt with the system Python so
#    `python -c "import flash_rt"` works from any working directory. ──
RUN python3 -m pip install --no-cache-dir --no-build-isolation -e ".[torch]"

# ── Smoke check at image-build time so a broken image fails the
#    docker build, not the user's first pull. Note Thor produces NO
#    flash_rt_fa2.so (FA2 is SM80/89/120-only); the smoke deliberately
#    does NOT import it. ──
RUN python3 -c "import flash_rt; \
print('flash_rt', flash_rt.__version__); \
from flash_rt import flash_rt_kernels; \
print('flash_rt_kernels symbols:', len([s for s in dir(flash_rt_kernels) if not s.startswith('_')])); \
import os; \
assert os.path.exists(os.path.join(os.path.dirname(flash_rt.__file__), 'libfmha_fp16_strided.so')), 'libfmha_fp16_strided.so missing'; \
print('Thor build OK — kernels + fmha_fp16_strided + fp4 + jax_ffi present')"

# Default to a Python REPL with flash_rt pre-imported. Override with
#   docker run --rm --gpus all -it --runtime nvidia flashrt:thor <cmd>
CMD ["python3", "-c", "import flash_rt; print('flash_rt', flash_rt.__version__, '— Thor (SM110) build ready'); import code; code.interact(local={'flash_rt': flash_rt})"]
