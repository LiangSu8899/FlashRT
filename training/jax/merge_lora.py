"""Merge a JAX LoRA delta into a pi0.5 base — JAX native (Orbax → Orbax).

Closes the JAX-side train→serve loop. After the upstream
``train_jax_lora_recap.py`` saves a full Orbax checkpoint
containing both the frozen base weights and the trained LoRA
adapters, this tool produces a *standalone* Orbax checkpoint with
the LoRA delta folded into the base — drop-in for openpi-style
inference, including
:class:`flash_rt.frontends.jax.pi05_rtx.Pi05JaxFrontendRtx`.

Math, per LoRA-bearing layer ``L``::

    W_base[*outer, K, N]   ← read from Orbax
    A[*outer, K, rank]     ← lora_a
    B[*outer, rank, N]     ← lora_b
    delta = scaling * (A @ B)         # batched matmul over leading dims
    W_merged = W_base + delta

Two openpi naming patterns are handled by the same code path
(:func:`training.jax.checkpoint._resolve_base_path`):

* Einsum (``parent/lora_a``)              — base at ``parent/w``
* FeedForward (``parent/X_lora_a``)       — base at ``parent/X``

There is **no PyTorch in the merge path** — the JAX line stays
JAX-native end to end. Users running the FlashRT torch frontend
have a separate, parallel torch path
(``training.rl.merge_lora``); the two never need to meet.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import flax.traverse_util as tu
import numpy as np

from training.jax.checkpoint import (
    _LORA_A_SUFFIX,
    _LORA_B_SUFFIX,
    discover_lora_pairs,
    load_lora_state,
)

logger = logging.getLogger(__name__)


# ── Orbax I/O ──────────────────────────────────────────────────────


def _restore_orbax_flat(orbax_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load an Orbax checkpoint into a (flat-dict, nested-dict) pair.

    Both views point at the same arrays — the flat dict is for
    discovery/edits; the nested dict is what Orbax's ``save`` wants.
    """
    import orbax.checkpoint as ocp

    ckptr = ocp.PyTreeCheckpointer()
    nested = ckptr.restore(str(orbax_dir))
    flat = tu.flatten_dict(nested, sep="/")
    flat_np = {k: np.asarray(v) for k, v in flat.items()}
    return flat_np, nested


def _save_orbax_from_flat(
    flat: dict[str, np.ndarray],
    output_dir: Path,
    *,
    overwrite: bool,
    wrap_in_params: bool = True,
) -> Path:
    """Write a flat dict to a fresh Orbax directory.

    When ``wrap_in_params`` is True (the default), the saved tree gets
    a ``{"params": <model>}`` outer dict. openpi's ``restore_params``
    (the loader path
    :func:`flash_rt.core.weights.loader._load_orbax` uses for
    inference) expects this wrapper — without it the loader raises
    ``KeyError('params')`` at metadata read.

    Upstream ``train_jax_lora_recap.py`` saves WITHOUT the wrapper
    (the trained nnx state's top-level keys are ``PaliGemma``,
    ``action_in_proj``, ...), while the pristine ``pi05_base/params``
    distributed by openpi DOES have the wrapper. Always wrapping
    the merged output normalises to the openpi convention so
    every downstream JAX consumer (inference frontend included)
    loads without surprise.
    """
    import orbax.checkpoint as ocp

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass overwrite=True")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Strip a leading ``params/`` if the input already had one — we re-add
    # below so the saved layout is canonical regardless of the source.
    keys_with_prefix = sum(1 for k in flat if k.startswith("params/"))
    if keys_with_prefix > 0 and keys_with_prefix == len(flat):
        flat = {k[len("params/"):]: v for k, v in flat.items()}

    nested_inner = tu.unflatten_dict(
        {tuple(k.split("/")): v for k, v in flat.items()}
    )
    nested = {"params": nested_inner} if wrap_in_params else nested_inner
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(str(output_dir), nested)
    return output_dir


# ── Math ───────────────────────────────────────────────────────────


def _compute_delta(
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    scaling: float,
) -> np.ndarray:
    """``delta = scaling * (lora_a @ lora_b)`` with leading-axis broadcast.

    ``np.matmul`` on rank ≥ 2 inputs treats the last two axes as the
    matmul dims and broadcasts the rest — exactly the contraction we
    want for every openpi LoRA pair (``Einsum`` and ``FeedForward``,
    multi-axis or not). The result has the same shape as the base
    weight ``W``.

    Computed in fp32 for precision, then cast back to the LoRA dtype
    (typically bf16 on disk).
    """
    if lora_a.ndim < 2 or lora_b.ndim < 2:
        raise ValueError(
            f"lora_a/lora_b must have rank ≥ 2; got shapes "
            f"{lora_a.shape} and {lora_b.shape}"
        )
    if lora_a.shape[-1] != lora_b.shape[-2]:
        raise ValueError(
            f"rank axis mismatch: lora_a last={lora_a.shape[-1]} "
            f"vs lora_b second-last={lora_b.shape[-2]}"
        )
    a32 = lora_a.astype(np.float32)
    b32 = lora_b.astype(np.float32)
    delta32 = (a32 @ b32) * float(scaling)
    return delta32.astype(lora_a.dtype, copy=False)


# ── Public API ─────────────────────────────────────────────────────


def merge_lora_into_base(
    trained_orbax_dir: str | Path,
    output_orbax_dir: str | Path,
    *,
    scaling: float | dict[str, float] = 1.0,
    overwrite: bool = False,
) -> Path:
    """Fold LoRA into the base in-place, single-input form.

    Use this when you have a checkpoint produced by the upstream
    ``train_jax_lora_recap.py:save_checkpoint`` (which writes the
    full nnx state — base ``w`` next to ``lora_a`` / ``lora_b``).

    Args:
        trained_orbax_dir: Orbax directory carrying both the frozen
            base weights and the trained LoRA adapters.
        output_orbax_dir: Destination Orbax directory.
        scaling: LoRA scaling factor — ``float`` applies uniformly,
            ``dict[layer_path, float]`` applies per layer. pi05's
            openpi default is ``alpha == rank`` → 1.0.
        overwrite: If True, allow overwriting ``output_orbax_dir``.

    Returns:
        ``output_orbax_dir`` as a :class:`Path`.
    """
    trained_orbax_dir = Path(trained_orbax_dir)
    output_orbax_dir = Path(output_orbax_dir)

    flat, _ = _restore_orbax_flat(trained_orbax_dir)
    pairs = discover_lora_pairs(flat)
    if not pairs:
        raise ValueError(
            f"no LoRA pairs found in {trained_orbax_dir}; nothing to merge"
        )

    # Resolve scaling per layer.
    if isinstance(scaling, dict):
        scaling_lookup = {k: float(v) for k, v in scaling.items()}
    else:
        scaling_lookup = {layer: float(scaling) for layer, _, _, _ in pairs}

    n_merged = 0
    for layer_path, base_path, a_path, b_path in pairs:
        if base_path not in flat:
            raise KeyError(
                f"base weight missing for LoRA layer {layer_path!r}: "
                f"expected '{base_path}' in Orbax tree."
            )
        s = scaling_lookup.get(layer_path, 1.0)
        delta = _compute_delta(flat[a_path], flat[b_path], s)
        if delta.shape != flat[base_path].shape:
            raise RuntimeError(
                f"shape mismatch on {layer_path}: "
                f"delta {delta.shape} vs base {flat[base_path].shape}"
            )
        # Cast base to fp32, add delta in fp32, then back to base dtype.
        base_dtype = flat[base_path].dtype
        merged = (
            flat[base_path].astype(np.float32) + delta.astype(np.float32)
        ).astype(base_dtype, copy=False)
        flat[base_path] = merged
        flat.pop(a_path, None)
        flat.pop(b_path, None)
        n_merged += 1

    logger.info(
        "merged %d LoRA layers from %s → %s",
        n_merged, trained_orbax_dir, output_orbax_dir,
    )
    return _save_orbax_from_flat(flat, output_orbax_dir, overwrite=overwrite)


def merge_lora_artifact_into_base(
    base_orbax_dir: str | Path,
    lora_dir: str | Path,
    output_orbax_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Two-input form: untouched base + LoRA artifact → merged base.

    Use this when the LoRA delta is shipped separately (e.g. a small
    ``lora_weights.npz`` + ``lora_metadata.json`` produced by
    :func:`training.jax.checkpoint.save_lora_state`) and you want to
    fold it into a clean upstream base such as ``pi05_base/params``.

    Args:
        base_orbax_dir: Orbax directory of the **clean** pi0.5 base.
            Read-only — never modified.
        lora_dir: Directory produced by
            :func:`training.jax.checkpoint.save_lora_state` (must
            carry ``lora_weights.npz`` + ``lora_metadata.json``).
        output_orbax_dir: Destination Orbax directory.
        overwrite: If True, allow overwriting ``output_orbax_dir``.

    Returns:
        ``output_orbax_dir`` as a :class:`Path`.
    """
    base_orbax_dir = Path(base_orbax_dir)
    lora_dir = Path(lora_dir)
    output_orbax_dir = Path(output_orbax_dir)

    flat, _ = _restore_orbax_flat(base_orbax_dir)
    lora_flat, metadata = load_lora_state(lora_dir)

    n_merged = 0
    for layer_path in metadata.layer_paths:
        base_path = metadata.base_path_per_layer[layer_path]
        scaling = metadata.scaling_per_layer.get(layer_path, 1.0)
        # The LoRA paths in the metadata may use either the Einsum
        # convention (parent/lora_a) or the FeedForward one
        # (parent/X_lora_a). We re-discover from the npz to stay
        # uniform with merge_lora_into_base.
        a_candidates = [
            f"{layer_path}/{_LORA_A_SUFFIX}",
            # FeedForward case: layer_path itself is the base path; the
            # lora_a key sits next to it as <parent>/<X>_lora_a where
            # <parent>/<X> == base_path.
            base_path + "_" + _LORA_A_SUFFIX if "/" in base_path else "",
        ]
        b_candidates = [
            f"{layer_path}/{_LORA_B_SUFFIX}",
            base_path + "_" + _LORA_B_SUFFIX if "/" in base_path else "",
        ]
        a_path = next((p for p in a_candidates if p and p in lora_flat), None)
        b_path = next((p for p in b_candidates if p and p in lora_flat), None)
        if a_path is None or b_path is None:
            raise KeyError(
                f"could not locate lora_a/lora_b for layer {layer_path!r}; "
                f"npz has {sorted(lora_flat)[:6]}..."
            )
        if base_path not in flat:
            raise KeyError(
                f"base weight missing for LoRA layer {layer_path!r}: "
                f"expected '{base_path}' in base Orbax."
            )
        delta = _compute_delta(lora_flat[a_path], lora_flat[b_path], scaling)
        if delta.shape != flat[base_path].shape:
            raise RuntimeError(
                f"shape mismatch on {layer_path}: "
                f"delta {delta.shape} vs base {flat[base_path].shape}"
            )
        base_dtype = flat[base_path].dtype
        flat[base_path] = (
            flat[base_path].astype(np.float32) + delta.astype(np.float32)
        ).astype(base_dtype, copy=False)
        n_merged += 1

    logger.info(
        "merged %d LoRA layers from %s + %s → %s",
        n_merged, base_orbax_dir, lora_dir, output_orbax_dir,
    )
    return _save_orbax_from_flat(flat, output_orbax_dir, overwrite=overwrite)
