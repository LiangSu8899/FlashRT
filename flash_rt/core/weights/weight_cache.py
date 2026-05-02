"""FlashVLA — FP8 Weight Cache.

Caches FP8-quantized engine weights to disk after first load.
Subsequent loads skip Orbax read + transform + FP8 quantize (~42s → ~5s).

Cache format: JSON header (length-prefixed) + contiguous raw binary blobs.
Cache location: ~/.flash_rt/weights/{ckpt_hash}_nv{num_views}.bin

Designed for JAX (Orbax) frontend where loading is expensive.
Torch (safetensors) loads in ~3s and doesn't need this.
"""

import json
import logging
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

WEIGHT_CACHE_DIR = Path.home() / ".flash_rt" / "weights"
CACHE_VERSION = 1
MAGIC = b"FVW1"  # FlashVLA Weights v1


def _cache_path(ckpt_hash: str, num_views: int) -> Path:
    return WEIGHT_CACHE_DIR / f"{ckpt_hash}_nv{num_views}.bin"


def save_weight_cache(checkpoint_path: str, num_views: int,
                      entries: List[Dict], blobs: List[bytes]) -> Path:
    """Save weight cache to disk.

    Args:
        checkpoint_path: for hashing
        num_views: buffer sizes depend on this
        entries: list of {"name": str, "nbytes": int, "dtype": str, "shape": list}
        blobs: list of raw bytes, one per entry
    """
    from flash_rt.core.quant.calibrator import _checkpoint_hash

    ckpt_hash = _checkpoint_hash(checkpoint_path)
    WEIGHT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Compute offsets
    offset = 0
    for entry, blob in zip(entries, blobs):
        entry["offset"] = offset
        entry["nbytes"] = len(blob)
        offset += len(blob)

    header = {
        "version": CACHE_VERSION,
        "ckpt_hash": ckpt_hash,
        "num_views": num_views,
        "num_entries": len(entries),
        "total_bytes": offset,
        "entries": entries,
    }
    header_json = json.dumps(header).encode("utf-8")

    path = _cache_path(ckpt_hash, num_views)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
        for blob in blobs:
            f.write(blob)

    logger.info("Weight cache saved: %s (%.1f MB, %d entries)",
                path, offset / 1e6, len(entries))
    return path


def load_weight_cache(checkpoint_path: str, num_views: int
                      ) -> Optional[Tuple[Dict, bytes]]:
    """Load weight cache from disk.

    Returns (header_dict, body_bytes) or None on miss.
    """
    from flash_rt.core.quant.calibrator import _checkpoint_hash

    try:
        ckpt_hash = _checkpoint_hash(checkpoint_path)
    except FileNotFoundError:
        return None

    path = _cache_path(ckpt_hash, num_views)
    if not path.exists():
        logger.info("Weight cache miss: %s", path)
        return None

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            logger.warning("Weight cache bad magic, ignoring: %s", path)
            return None

        header_len = struct.unpack("<I", f.read(4))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))

        if header.get("version") != CACHE_VERSION:
            logger.warning("Weight cache version mismatch, ignoring")
            return None
        if header.get("num_views") != num_views:
            logger.warning("Weight cache num_views mismatch (%s != %s)",
                           header.get("num_views"), num_views)
            return None

        body = f.read()

    if len(body) != header["total_bytes"]:
        logger.warning("Weight cache truncated (%d != %d)",
                       len(body), header["total_bytes"])
        return None

    logger.info("Weight cache loaded: %s (%.1f MB, %d entries)",
                path, len(body) / 1e6, header["num_entries"])
    return header, body


def clear_weight_cache(checkpoint_path: str = None):
    """Clear weight cache files."""
    if not WEIGHT_CACHE_DIR.exists():
        return

    if checkpoint_path is None:
        count = 0
        for f in WEIGHT_CACHE_DIR.glob("*.bin"):
            f.unlink()
            count += 1
        if count:
            logger.info("Cleared %d weight cache files", count)
    else:
        from flash_rt.core.quant.calibrator import _checkpoint_hash
        try:
            ckpt_hash = _checkpoint_hash(checkpoint_path)
        except FileNotFoundError:
            return
        count = 0
        for f in WEIGHT_CACHE_DIR.glob(f"{ckpt_hash}_*.bin"):
            f.unlink()
            count += 1
        if count:
            logger.info("Cleared %d weight cache files for %s", count, ckpt_hash)
