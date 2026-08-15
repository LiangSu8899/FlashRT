"""Apache-2.0 Transformers-backed Chameleon VQ-VAE helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


_VQ_PREFIXES = ("model.vqmodel.", "vqmodel.")


def _checkpoint_index(checkpoint_dir: Path) -> tuple[dict[str, str], Path]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid weight_map in {index_path}")
        return weight_map, index_path

    single = checkpoint_dir / "model.safetensors"
    if single.is_file():
        from safetensors import safe_open

        with safe_open(single, framework="pt", device="cpu") as handle:
            return {key: single.name for key in handle.keys()}, single
    raise FileNotFoundError(
        f"no model.safetensors or model.safetensors.index.json under "
        f"{checkpoint_dir}")


def _vq_weight_map(checkpoint_dir: Path) -> tuple[dict[str, str], str, Path]:
    weight_map, manifest = _checkpoint_index(checkpoint_dir)
    for prefix in _VQ_PREFIXES:
        selected = {
            key[len(prefix):]: shard
            for key, shard in weight_map.items()
            if key.startswith(prefix)
        }
        if selected:
            return selected, prefix, manifest
    raise FileNotFoundError(
        f"no Chameleon VQ-VAE weights ({', '.join(_VQ_PREFIXES)}) under "
        f"{checkpoint_dir}")


def load_chameleon_vqvae(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Load only the HF Chameleon VQ-VAE from checkpoint safetensors."""
    from safetensors import safe_open
    from transformers import ChameleonConfig, ChameleonVQVAE
    from transformers.models.chameleon.modeling_chameleon import (
        ChameleonImageVocabularyMapping,
    )

    root = Path(checkpoint_dir).expanduser().resolve()
    selected, prefix, _ = _vq_weight_map(root)
    state_dict: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[str]] = {}
    for key, shard in selected.items():
        by_shard.setdefault(shard, []).append(key)
    for shard, keys in by_shard.items():
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                state_dict[key] = handle.get_tensor(prefix + key)

    config = ChameleonConfig.from_pretrained(str(root))
    model = ChameleonVQVAE(config.vq_config)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Chameleon VQ-VAE checkpoint mismatch: "
            f"missing={incompatible.missing_keys[:4]}, "
            f"unexpected={incompatible.unexpected_keys[:4]}")
    model = model.eval().to(device=device, dtype=dtype)
    mapping = ChameleonImageVocabularyMapping(config.vocabulary_map)
    return model, mapping


def preprocess_vqvae_image(
    image: Image.Image,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Convert a PIL image to the Chameleon VQ-VAE ``[-1, 1]`` tensor."""
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(background, rgba).convert("RGB")
    values = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype).contiguous()


def encode_vqvae_tokens(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return image-token indices across Transformers 4.43+ output APIs."""
    output = model.encode(pixel_values)
    if hasattr(output, "image_tokens"):
        return output.image_tokens
    return output[2]


def vqvae_checkpoint_digest(checkpoint_dir: str | Path) -> str:
    """Return a stable digest of the VQ-VAE config and referenced shards."""
    root = Path(checkpoint_dir).expanduser().resolve()
    selected, _, manifest = _vq_weight_map(root)
    paths = [root / "config.json", manifest]
    paths.extend(root / name for name in sorted(set(selected.values())))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:16]
