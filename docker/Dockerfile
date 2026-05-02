# FlashRT — official x86_64 image (RTX 5090 / 4090 / Ampere).
#
# Base: NVIDIA NGC PyTorch 25.10 — ships CUDA 13.0, PyTorch 2.9 with
# SM120 support, cuBLASLt, nvcc 13.0.88, Python 3.12. Same combo
# pinned in this repo's CI.
#
# Build:
#   docker build -t flashrt:dev -f docker/Dockerfile .
#
# Run:
#   docker run --rm --gpus all -it flashrt:dev
#
# Modal / RunPod / cloud:
#   image = modal.Image.from_registry("ghcr.io/liangsu8899/flashrt:<tag>")
#
# Layer order is tuned for cache reuse: anything that changes per
# commit (the source tree) is copied last so flipping a one-line
# patch does NOT invalidate the CUTLASS clone or pip layer.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

# Build-arg knobs:
#   GPU_ARCH   pin the SASS arch (120=5090, 89=4090, 80=A100, 86=3090).
#              Default omits the flag; CMake auto-detects via nvidia-smi
#              at build time, which works for `docker build --gpus all`
#              + same-host deployment but not for cross-arch image
#              distribution. Override explicitly if you ship to a
#              different GPU than the build host.
#   GIT_REF    optional CUTLASS pin if upstream tags get yanked.
#   FA2_HDIMS  forwarded to CMake to slim FA2 codegen for shipped models.
ARG GPU_ARCH=""
ARG CUTLASS_REF=v4.4.2
ARG FA2_HDIMS=""

# ── Tiny base utilities (NGC image already has cmake/ninja/git, but
#    pin in case a future base image drops one of them). ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        ccache \
    && rm -rf /var/lib/apt/lists/*

# ── CUTLASS 4.4.2 — vendor in the image so per-commit rebuilds skip
#    the 80 MB git clone every time. Pinned to CUTLASS_REF for repro. ──
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

# ── Build the CUDA kernels ──
# All five .so targets land directly under flash_rt/ thanks to
# LIBRARY_OUTPUT_DIRECTORY in CMakeLists; no follow-up `cp` /
# `make install` step is needed.
RUN cmake -B build -S . \
        $( [ -n "${GPU_ARCH}" ] && echo "-DGPU_ARCH=${GPU_ARCH}" ) \
        $( [ -n "${FA2_HDIMS}" ] && echo "-DFA2_HDIMS=${FA2_HDIMS}" ) \
 && cmake --build build -j"$(nproc)" \
 && rm -rf build/CMakeFiles

# ── Editable install — registers flash_rt with the system Python so
#    `python -c "import flash_rt"` works from any working directory. ──
RUN python3 -m pip install --no-cache-dir --no-build-isolation -e ".[torch]"

# ── Smoke check at image-build time so a broken image fails the
#    docker build, not the user's first pull. ──
RUN python3 -c "import flash_rt; \
print('flash_rt', flash_rt.__version__); \
from flash_rt import flash_rt_kernels, flash_rt_fa2; \
assert callable(flash_rt_fa2.fwd_bf16); \
print('kernels + fa2 OK')"

# Default to a Python REPL with flash_rt pre-imported. Override with
#   docker run --rm --gpus all -it flashrt:dev <your-command>
CMD ["python3", "-c", "import flash_rt; print('flash_rt', flash_rt.__version__, '— ready'); import code; code.interact(local={'flash_rt': flash_rt})"]
