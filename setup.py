"""FlashRT — pip install support.

Usage:
    # Development mode (editable, recommended for development):
    pip install -e .

    # Standard install:
    pip install .

    # With optional dependencies:
    pip install -e ".[torch]"       # PyTorch frontend
    pip install -e ".[jax]"         # JAX frontend
    pip install -e ".[server]"      # FastAPI server
    pip install -e ".[all]"         # Everything

Note: CUDA kernels must be built separately. CMake drops the .so
files directly into ``flash_vla/`` at build time — no follow-up
``make install`` / ``ninja install`` / manual ``cp`` step is needed:

    cmake -B build -S .
    cmake --build build -j

After this, ``flash_vla/flash_vla_kernels*.so`` (and on RTX,
``flash_vla_fa2*.so``; on Thor/Hopper, ``flash_vla_fp4*.so``) exist
and ``import flash_vla`` works in editable installs.

Optional pip dependency: the legacy upstream attention path
(``FVK_RTX_FA2=0`` or sites excluded via ``FVK_RTX_FA2_SITES``) and
the GROOT backend require the ``flash-attn`` wheel. The default RTX
Pi0 / Pi0.5 path uses the vendored ``flash_vla_fa2`` and does NOT
need it — environments without a prebuilt flash-attn wheel (Modal,
older CUDA images) can still install and run.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
