"""FlashRT — Weight Loader.

Loads model checkpoints from multiple formats into a unified
dict[str, numpy.ndarray] with engine key names.

Supports:
  - PyTorch safetensors (per-layer keys, bfloat16)
  - JAX Orbax/OCDBT (stacked layers, float32)

Output: dict mapping engine keys → numpy arrays, all in float32.
The weight transformer (transformer.py) handles all subsequent
transformations (QKV merge, interleave, FP8 quantize).
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np


logger = logging.getLogger(__name__)


def detect_format(path: str) -> str:
    """Auto-detect checkpoint format.

    Returns: "safetensors", "orbax", or "unknown"
    """
    p = Path(path)
    if p.is_file() and p.suffix == '.safetensors':
        return "safetensors"
    if p.is_dir():
        if (p / "model.safetensors").exists():
            return "safetensors"
        if (p / "params").is_dir():
            return "orbax"
        # Check inside params/ for OCDBT markers
        if (p / "_METADATA").exists() or (p / "manifest.ocdbt").exists():
            return "orbax"
    return "unknown"


def load_weights(path: str, format: str = None) -> Dict[str, np.ndarray]:
    """Load checkpoint into unified engine-key dict.

    Args:
        path: Path to checkpoint file or directory.
        format: Force format. Auto-detect if None.

    Returns:
        Dict mapping engine key names to numpy arrays (float32).
    """
    if format is None:
        format = detect_format(path)
    logger.info(f"Loading checkpoint: {path} (format={format})")

    if format == "safetensors":
        return _load_safetensors(path)
    elif format == "orbax":
        return _load_orbax(path)
    else:
        raise ValueError(f"Unknown checkpoint format at {path}. "
                        f"Expected safetensors or Orbax directory.")


def _load_safetensors(path: str) -> Dict[str, np.ndarray]:
    """Load safetensors → dict with original HF key names + numpy arrays."""
    from safetensors import safe_open

    p = Path(path)
    if p.is_dir():
        p = p / "model.safetensors"

    weights = {}
    # Try numpy first, fall back to torch for bfloat16
    try:
        with safe_open(str(p), framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    except TypeError:
        # numpy can't handle bfloat16, use torch
        import torch
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                weights[key] = t.float().numpy()

    logger.info(f"Loaded {len(weights)} tensors from safetensors")
    return weights


def _load_orbax(path: str) -> Dict[str, np.ndarray]:
    """Load JAX Orbax checkpoint → dict with JAX key names + numpy arrays.

    Handles multi-device checkpoints (saved on 8 GPUs) by restoring
    as numpy arrays with no sharding.
    """
    import sys

    p = Path(path)
    params_path = p / "params" if (p / "params").is_dir() else p

    # Use openpi's restore_params if available (handles sharding correctly).
    # If openpi is not importable we fall through to the direct orbax
    # fallback below — verified byte-identical on tested checkpoints, so
    # both paths produce the same engine-weights dict.
    try:
        # Try importing openpi's loader
        import importlib.util
        openpi_paths = [
            "/workspace/src",
        ]
        for op in openpi_paths:
            if os.path.exists(op) and op not in sys.path:
                sys.path.insert(0, op)

        from openpi.models.model import restore_params
        import flax.traverse_util as tu

        raw = restore_params(str(params_path), restore_type=np.ndarray)
        flat = tu.flatten_dict(raw, sep=".")
        weights = {k: np.array(v, dtype=np.float32) if v.dtype != np.float32 else v
                   for k, v in flat.items()}
        logger.info(f"Loaded {len(weights)} tensors from Orbax (via openpi)")
        return weights

    except ImportError:
        # Fallback: direct orbax loading
        import orbax.checkpoint as ocp
        import jax

        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        with ocp.PyTreeCheckpointer() as ckptr:
            metadata = ckptr.metadata(str(params_path))
            item = {"params": metadata["params"]}
            params = ckptr.restore(
                str(params_path),
                ocp.args.PyTreeRestore(
                    item=item,
                    restore_args=jax.tree.map(
                        lambda _: ocp.ArrayRestoreArgs(
                            sharding=sharding, restore_type=np.ndarray
                        ), item
                    ),
                ),
            )["params"]

        import flax.traverse_util as tu
        flat = tu.flatten_dict(params, sep=".")
        # Remove 'value' suffix if present (NNX convention)
        if all(k.endswith(".value") for k in flat):
            flat = {k[:-6]: v for k, v in flat.items()}

        weights = {k: np.array(v, dtype=np.float32) if v.dtype != np.float32 else v
                   for k, v in flat.items()}
        logger.info(f"Loaded {len(weights)} tensors from Orbax (direct)")
        return weights
