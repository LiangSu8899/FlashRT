#!/usr/bin/env bash
# Build flash_rt_amd_kernels in a ROCm environment (GPU visible or not).
#   bash scripts/amd/build_amd.sh [gfx950]
#
# Prefers CMake; falls back to a direct hipcc one-shot when no usable
# cmake is present. Output: flash_rt/amd/flash_rt_amd_kernels*.so
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ARCH="${1:-gfx950}"
PYTHON_BIN="${PYTHON:-python3}"
JOBS="${SLURM_CPUS_PER_TASK:-8}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export PATH=${ROCM_PATH}/bin:${PATH}

# pybind11 headers: use the interpreter's copy, else a one-time --target
# install into FLASHRT_AMD_PYDEPS (never into a read-only/shared venv).
if ! "${PYTHON_BIN}" -m pybind11 --includes >/dev/null 2>&1; then
  PYDEPS="${FLASHRT_AMD_PYDEPS:-${ROOT}/.pydeps}"
  if [[ ! -d "${PYDEPS}/pybind11" ]]; then
    "${PYTHON_BIN}" -m pip install --target "${PYDEPS}" pybind11
  fi
  export PYTHONPATH="${PYDEPS}${PYTHONPATH:+:${PYTHONPATH}}"
fi
PYBIND_INCLUDES="$("${PYTHON_BIN}" -m pybind11 --includes)"
EXT_SUFFIX="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')"

mkdir -p "${ROOT}/flash_rt/amd"
OUT="${ROOT}/flash_rt/amd/flash_rt_amd_kernels${EXT_SUFFIX}"

if command -v cmake >/dev/null 2>&1 \
   && cmake -B "${ROOT}/build-amd" -S "${ROOT}/csrc/amd" \
        -DGPU_ARCH="${GPU_ARCH}" -DPython_EXECUTABLE="${PYTHON_BIN}" 2>&1; then
  cmake --build "${ROOT}/build-amd" -j "${JOBS}"
  echo "built via cmake: $(ls "${ROOT}"/flash_rt/amd/flash_rt_amd_kernels*.so)"
else
  echo "cmake unavailable or failed — falling back to direct hipcc"
  hipcc -O3 -std=c++17 -fPIC -shared \
    --offload-arch="${GPU_ARCH}" \
    -ffp-contract=fast \
    -DFLASHRT_AMD_GPU_ARCH="\"${GPU_ARCH}\"" \
    ${PYBIND_INCLUDES} \
    -I"${ROOT}/csrc/amd" \
    -x hip "${ROOT}/csrc/amd/bindings.cpp" \
    "${ROOT}/csrc/amd/kernels/norm.hip" \
    -o "${OUT}"
  echo "built via hipcc: ${OUT}"
fi

"${PYTHON_BIN}" - <<PY
import sys; sys.path.insert(0, "${ROOT}")
from flash_rt.amd import flash_rt_amd_kernels as k
print("import ok:", dict(k.build_info()))
PY
