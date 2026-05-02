"""Merge a saved LoRA adapter into a pi0.5 base safetensors.

Closes the train→serve loop: after :func:`save_lora_state` writes
``lora.safetensors`` + ``lora_metadata.json``, this tool produces a
*standalone* pi0.5 ``model.safetensors`` that any downstream
inference consumer can load — including FlashRT's
``Pi05TorchFrontendRtx`` (which expects a directory mirroring the
upstream ``pi05_libero_pytorch`` layout).

Math, per LoRA-injected ``nn.Linear`` layer ``L``::

    W_base[N, K]    ←  load from base safetensors
    A[r, K], B[N, r], scaling  ←  load from lora.safetensors + metadata
    delta[N, K]    =  scaling * (B @ A)
    W_merged[N, K] =  W_base + delta

Non-LoRA tensors are passed through unchanged. The output directory
mirrors the input layout (``config.json``, ``policy_postprocessor.json``,
``policy_preprocessor.json``, ``assets/``) so the saved tree is a
drop-in replacement for the base directory.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


def merge_lora_into_base(
    base_dir: str | Path,
    lora_dir: str | Path,
    output_dir: str | Path,
    *,
    safetensors_filename: str = "model.safetensors",
    overwrite: bool = False,
) -> Path:
    """Merge ``lora_dir/lora.safetensors`` into ``base_dir/{safetensors_filename}``.

    Args:
        base_dir: Directory of the pi0.5 base ckpt (must contain
            ``model.safetensors``, ``config.json``, etc.). Read-only —
            the input is NEVER modified.
        lora_dir: Directory produced by
            :func:`training.rl.checkpoint.save_lora_state` (contains
            ``lora.safetensors`` + ``lora_metadata.json``).
        output_dir: Destination directory. Must not already exist
            unless ``overwrite=True``.
        safetensors_filename: Override the base ckpt filename
            (default ``model.safetensors`` — matches openpi).
        overwrite: If True, allow overwriting ``output_dir``.

    Returns:
        ``output_dir`` as a :class:`Path`.
    """
    base_dir = Path(base_dir)
    lora_dir = Path(lora_dir)
    output_dir = Path(output_dir)

    base_safetensors = base_dir / safetensors_filename
    if not base_safetensors.exists():
        raise FileNotFoundError(f"missing {base_safetensors}")
    lora_safetensors = lora_dir / "lora.safetensors"
    lora_metadata_path = lora_dir / "lora_metadata.json"
    if not (lora_safetensors.exists() and lora_metadata_path.exists()):
        raise FileNotFoundError(
            f"lora_dir must contain lora.safetensors + lora_metadata.json: {lora_dir}"
        )

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass overwrite=True")
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()

    # Copy the entire base tree first (config.json, policy_*.json,
    # assets/, etc.) — preserves any auxiliary files the inference
    # consumer might expect alongside the safetensors.
    shutil.copytree(base_dir, output_dir)

    base_state = load_file(str(base_safetensors))
    lora_weights = load_file(str(lora_safetensors))
    metadata = json.loads(lora_metadata_path.read_text())

    layer_names = list(metadata["layer_names"])
    scaling_per_layer = {k: float(v) for k, v in metadata["scaling_per_layer"].items()}

    n_merged = 0
    n_missing_base = 0
    for layer in layer_names:
        weight_key = f"{layer}.weight"
        if weight_key not in base_state:
            n_missing_base += 1
            logger.warning(
                "LoRA layer %r has no matching base weight %r; skipping.",
                layer,
                weight_key,
            )
            continue
        a = lora_weights[f"{layer}.lora_A"]
        b = lora_weights[f"{layer}.lora_B"]
        scaling = scaling_per_layer[layer]

        # delta_NK = scaling * (B @ A); B shape (N, r), A shape (r, K).
        # Compute in fp32 for numerical safety, then cast back to the
        # base weight's dtype before adding.
        delta = (scaling * (b.float() @ a.float())).to(
            base_state[weight_key].dtype
        )
        # Both shapes must match (N, K) where N=out_features,
        # K=in_features in nn.Linear convention.
        if delta.shape != base_state[weight_key].shape:
            raise RuntimeError(
                f"shape mismatch on {layer}: delta {tuple(delta.shape)} "
                f"vs base {tuple(base_state[weight_key].shape)}"
            )
        base_state[weight_key] = base_state[weight_key] + delta
        n_merged += 1

    logger.info(
        "merged %d LoRA layers; %d had no matching base weight (skipped).",
        n_merged,
        n_missing_base,
    )

    # Atomic-ish write: temp file then rename.
    out_safetensors = output_dir / safetensors_filename
    tmp = out_safetensors.with_suffix(out_safetensors.suffix + ".tmp")
    save_file(base_state, str(tmp))
    tmp.replace(out_safetensors)

    return output_dir
