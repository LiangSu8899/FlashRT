"""Locate ``flash_rt_jax_ffi.so`` and register its handlers with JAX.

The ``.so`` is installed by the CMake build into the package root
(``flash_rt/flash_rt_jax_ffi.so``). Two XLA FFI handlers are
exposed from C++ (see ``csrc/training/jax_ffi/``):

* ``flashvla_fp8_gemm_bf16_out``    — bf16 = fp8 @ fp8 (cuBLASLt FP8 GEMM)
* ``flashvla_quantize_fp8_static``  — bf16 → fp8 byte tensor, fixed scale

This module wraps both as JAX FFI targets registered under names
``flashvla::fp8_gemm_bf16_out`` and ``flashvla::quantize_fp8_static``
on the CUDA platform. ``ensure_registered()`` is idempotent and
cheap on subsequent calls — register-once-per-process semantics.
"""

from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path

# Lazy-imported to keep `import training.jax.fp8` cheap when JAX isn't installed.
_jax_ffi_module = None


def _jax_ffi():
    global _jax_ffi_module
    if _jax_ffi_module is None:
        import jax.ffi as ffi
        _jax_ffi_module = ffi
    return _jax_ffi_module


def _candidate_so_paths() -> list[Path]:
    """Return possible install locations of ``flash_rt_jax_ffi.so``."""
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    # 1. Adjacent to the FlashRT package: <repo>/flash_rt/flash_rt_jax_ffi.so
    repo_root = here.parents[3]                       # training/jax/fp8 → repo
    candidates.append(repo_root / "flash_rt" / "flash_rt_jax_ffi.so")
    # 2. CMake build dir during dev: <repo>/build/flash_rt_jax_ffi.so
    candidates.append(repo_root / "build" / "flash_rt_jax_ffi.so")
    # 3. Optional override
    override = os.environ.get("FLASHVLA_JAX_FFI_SO")
    if override:
        candidates.insert(0, Path(override))
    return candidates


_LOCK = threading.Lock()
_REGISTERED = False
_SO_HANDLE = None  # Keep alive: dlclose would tear down handlers JAX still holds.


# Mapping of {ctypes-symbol-name : JAX FFI target name}.
# Both target names use the ``flashvla::`` namespace so they cannot
# collide with other libraries' XLA FFI registrations.
_HANDLERS = {
    "flashvla_fp8_gemm_bf16_out":   "flashvla::fp8_gemm_bf16_out",
    "flashvla_quantize_fp8_static": "flashvla::quantize_fp8_static",
}


def get_so_path() -> Path:
    """Return the resolved path to ``flash_rt_jax_ffi.so`` or raise."""
    for p in _candidate_so_paths():
        if p.is_file():
            return p
    raise FileNotFoundError(
        "flash_rt_jax_ffi.so not found. Candidates checked: "
        + ", ".join(str(p) for p in _candidate_so_paths())
        + ". Build the JAX FFI target via `cmake --build build "
          "--target flash_rt_jax_ffi`, or set FLASHVLA_JAX_FFI_SO."
    )


def ensure_registered() -> None:
    """Register ``flashvla::*`` FFI targets with JAX (idempotent)."""
    global _REGISTERED, _SO_HANDLE
    if _REGISTERED:
        return
    with _LOCK:
        if _REGISTERED:
            return
        ffi = _jax_ffi()
        so_path = get_so_path()
        # ctypes.CDLL with RTLD_GLOBAL so XLA can resolve the handler
        # symbols at FFI call time without re-loading the .so.
        _SO_HANDLE = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
        for sym, target in _HANDLERS.items():
            fn_ptr = ctypes.cast(getattr(_SO_HANDLE, sym), ctypes.CFUNCTYPE(None))
            ffi.register_ffi_target(target, ffi.pycapsule(fn_ptr), platform="CUDA")
        _REGISTERED = True


def is_registered() -> bool:
    return _REGISTERED


def registered_targets() -> tuple[str, ...]:
    return tuple(_HANDLERS.values())
