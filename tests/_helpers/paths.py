"""Centralised checkpoint-path resolution for FlashRT tests.

Tests in this directory used to hardcode placeholder paths
(``"<your_pi05_torch_ckpt>"``), which made every script fail-fast for
public users. This helper unifies the resolution rule:

    env var (FLASH_RT_<NAME>) > legacy bare env var (<NAME>) > built-in default

Defaults point at the maintainer's Thor layout under ``/workspace/...``
so internal CI keeps running with no env exports. Public users export
the env vars listed below before invoking any test script.

The helper is intentionally tiny and dependency-free so it can be
imported by both pytest collection sites and bare ``python3 tests/foo.py``
runs (no ``conftest.py`` magic).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Defaults — Thor maintainer layout. Overridable per-key via env var.
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, str] = {
    # PyTorch (safetensors) checkpoints
    "PI05_CKPT":        "/workspace/pytorch_checkpoints/pi05_libero_converted",
    "PI0_CKPT":         "/workspace/pytorch_checkpoints/pi0_base_converted",
    "PI0FAST_CKPT":     "/workspace/pytorch_checkpoints/pi0_fast_base_converted",
    "TORCH_CKPTS":      "/workspace/pytorch_checkpoints",

    # JAX (Orbax) checkpoints
    "PI05_JAX_CKPT":    "/workspace/checkpoints/pi05_libero_jax",
    "PI0_JAX_CKPT":     "/workspace/checkpoints/pi0_base",
    "PI0FAST_JAX_CKPT": "/workspace/checkpoints/pi0_fast_base",
    "JAX_CKPTS":        "/workspace/checkpoints",

    # GROOT N1.6
    "GROOT_CKPT":       "/workspace/openpi-main/openpi-main/groot_ckpt",
    "GROOT_REF":        "/workspace/openpi-main/openpi-main/groot_ref",

    # GROOT N1.7 (separate container; see INTERNAL_TESTING.md §4.10)
    "GROOT_N17_CKPT":   "",  # filled at runtime from HF cache snapshot
    "GROOT_N17_FX":     "",
    "GROOT_N17_AUX":    "",
    "GROOT_N17_AUX_LIST": "",

    # Datasets / scratch
    "LIBERO_ROOT":      "/workspace/libero_10_image",
}


def _envlookup(name: str) -> Optional[str]:
    """Two-level env lookup: namespaced var wins, bare var as legacy fallback.

    ``FLASH_RT_PI05_CKPT`` is the documented form. Several test scripts
    historically used the bare ``PI05_CKPT`` name; we keep it working
    so existing CI invocations don't have to re-export.
    """
    v = os.environ.get(f"FLASH_RT_{name}")
    if v:
        return v
    v = os.environ.get(name)
    return v if v else None


def resolve(name: str, *, optional: bool = False, must_exist: bool = True) -> Optional[str]:
    """Return the resolved path for a given checkpoint key.

    Parameters
    ----------
    name : key from ``_DEFAULTS`` (e.g. ``"PI05_CKPT"``).
    optional : if True, return ``None`` instead of exiting when the path
               is missing. The caller is responsible for printing a SKIP
               line and continuing.
    must_exist : if False, return the resolved string without checking
                 the filesystem. Use this for prefixes like ``JAX_CKPTS``
                 that get joined with a sub-dir before access.
    """
    if name not in _DEFAULTS:
        raise KeyError(f"unknown checkpoint key: {name!r}; "
                       f"add a default to tests/_helpers/paths.py")
    val = _envlookup(name) or _DEFAULTS[name]
    if not val:
        if optional:
            return None
        _die(name, val, reason="no default and no env override")
    if must_exist and not Path(val).exists():
        if optional:
            return None
        _die(name, val, reason="path does not exist on disk")
    return val


def resolve_file(name: str, suffix: str, *, optional: bool = False) -> Optional[str]:
    """Resolve a base path then join a fixed sub-path. Useful for
    ``GROOT_REF / groot_ref_e2e_full.pt`` style references where the
    user controls the directory but not the filename inside it.
    """
    base = resolve(name, optional=optional, must_exist=False)
    if base is None:
        return None
    full = Path(base) / suffix
    if not full.exists():
        if optional:
            return None
        _die(name, str(full), reason=f"file missing under base ({suffix})")
    return str(full)


def _die(name: str, value: str, *, reason: str) -> None:
    print(
        f"  [ERROR] FlashRT test path '{name}' unresolved ({reason}).\n"
        f"          looked at: {value or '<empty>'}\n"
        f"          override:  export FLASH_RT_{name}=/path/to/your/ckpt\n"
        f"          (legacy bare {name}= also accepted)\n"
        f"          See tests/_helpers/paths.py for all keys + defaults.",
        file=sys.stderr,
    )
    sys.exit(2)
