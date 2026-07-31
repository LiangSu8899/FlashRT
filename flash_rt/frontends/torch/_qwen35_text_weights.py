"""Put a ``qwen3_5_text`` checkpoint on the device in the form it arrived in.

Nothing is dequantized and nothing is repacked. The packed 4-bit weights go to
the device as the file holds them and the kernels decode them there, so the
resident footprint is the checkpoint's own and a load is a copy rather than a
conversion.

What this produces is deliberately not tensors. Every address the decode step
needs is taken once, here, and handed on as an integer; the tensors themselves
are kept alive in ``anchors`` and never looked at again. A hot path that holds
tensors ends up asking Torch for shapes, slices and views -- on a 35B sibling
that reached 4274 dispatched operators per token, which at the edge target's
cost per dispatch was most of the step. The way not to pay that is not to have
the objects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

from flash_rt.frontends.torch._qwen35_text_spec import (
    PACKED_SUFFIX,
    SCALE_SUFFIX,
    TextDims,
    compressed_sites,
)


@dataclass
class TextWeights:
    """Device addresses for one loaded checkpoint.

    ``layers[i]`` maps a short name to an address or a size. ``anchors`` owns
    the tensors those addresses point into; dropping it frees the model.
    """

    dims: TextDims
    group_size: int
    top: dict[str, int] = field(default_factory=dict)
    layers: list[dict[str, int]] = field(default_factory=list)
    anchors: list[torch.Tensor] = field(default_factory=list)
    resident_bytes: int = 0

    def close(self) -> None:
        self.anchors.clear()
        self.layers.clear()
        self.top.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _Reader:
    """Opens each shard once and hands out tensors by name."""

    def __init__(self, path: str):
        from safetensors import safe_open

        index_path = os.path.join(path, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            import json

            with open(index_path, "r", encoding="utf-8") as f:
                self._map = json.load(f)["weight_map"]
        else:
            self._map = None
        self._path = path
        self._open = safe_open
        self._shards: dict[str, Any] = {}

    def _shard(self, name: str):
        shard = "model.safetensors" if self._map is None else self._map[name]
        reader = self._shards.get(shard)
        if reader is None:
            reader = self._open(os.path.join(self._path, shard),
                                framework="pt", device="cpu")
            self._shards[shard] = reader
        return reader

    def get(self, name: str) -> torch.Tensor:
        return self._shard(name).get_tensor(name)

    def has(self, name: str) -> bool:
        if self._map is not None:
            return name in self._map
        return name in self._shard(name).keys()


def _place(weights: TextWeights, tensor: torch.Tensor, device: str,
           dtype: torch.dtype | None = None) -> int:
    """Move one tensor to the device, keep it alive, return its address."""
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    resident = tensor.to(device).contiguous()
    weights.anchors.append(resident)
    weights.resident_bytes += resident.numel() * resident.element_size()
    return int(resident.data_ptr())


def _load_compressed(weights: TextWeights, reader: _Reader, site: str,
                     into: dict[str, int], prefix: str, shape: tuple[int, int],
                     device: str) -> None:
    """One packed weight: its two tensors, its shape, nothing else."""
    rows, columns = shape
    packed = reader.get(site + PACKED_SUFFIX)
    scale = reader.get(site + SCALE_SUFFIX)
    if tuple(packed.shape) != (rows, columns // 8):
        raise ValueError(
            f"{site}: packed is {tuple(packed.shape)}, but the geometry makes "
            f"it {(rows, columns // 8)}")
    if tuple(scale.shape) != (rows, columns // weights.group_size):
        raise ValueError(
            f"{site}: scale is {tuple(scale.shape)}, but group "
            f"{weights.group_size} makes it "
            f"{(rows, columns // weights.group_size)}")
    into[prefix + "_packed"] = _place(weights, packed, device, torch.int32)
    into[prefix + "_scale"] = _place(weights, scale, device, torch.bfloat16)
    into[prefix + "_n"] = rows
    into[prefix + "_k"] = columns


def load_text_weights(path: str, contract: dict[str, Any],
                      device: str = "cuda:0") -> TextWeights:
    """Load the text backbone, packed weights kept packed.

    ``contract`` is what ``validate_checkpoint`` returned, so the geometry has
    already been agreed with the file rather than being re-derived here.
    """
    dims: TextDims = contract["dims"]
    weights = TextWeights(dims=dims, group_size=contract["quant"].group_size)
    reader = _Reader(contract["checkpoint_path"])
    sites = compressed_sites(dims)

    embed_name = "model.language_model.embed_tokens.weight"
    weights.top["embed"] = _place(
        weights, reader.get(embed_name), device, torch.bfloat16)
    weights.top["final_norm"] = _place(
        weights, reader.get("model.language_model.norm.weight"), device,
        torch.bfloat16)
    weights.top["vocab_size"] = dims.vocab_size
    weights.top["hidden"] = dims.hidden

    # Tied embeddings mean the output projection is the embedding table read
    # the other way; there is no second copy to load and none to allocate.
    weights.top["lm_head_tied"] = int(dims.tie_word_embeddings)
    if dims.tie_word_embeddings:
        weights.top["lm_head"] = weights.top["embed"]
    elif reader.has("lm_head" + PACKED_SUFFIX):
        _load_compressed(weights, reader, "lm_head", weights.top, "lm_head",
                         (dims.vocab_size, dims.hidden), device)
    else:
        weights.top["lm_head"] = _place(
            weights, reader.get("lm_head.weight"), device, torch.bfloat16)

    for layer in range(dims.num_layers):
        prefix = f"model.language_model.layers.{layer}."
        entry: dict[str, int] = {}
        linear = dims.layer_types[layer] == "linear_attention"
        entry["linear_attention"] = int(linear)

        entry["input_norm"] = _place(
            weights, reader.get(prefix + "input_layernorm.weight"), device,
            torch.bfloat16)
        entry["post_norm"] = _place(
            weights, reader.get(prefix + "post_attention_layernorm.weight"),
            device, torch.bfloat16)

        for short, site in (("gate", "mlp.gate_proj"),
                            ("up", "mlp.up_proj"),
                            ("down", "mlp.down_proj")):
            _load_compressed(weights, reader, prefix + site, entry, short,
                             sites[prefix + site], device)

        if linear:
            for short, site in (("in_qkv", "linear_attn.in_proj_qkv"),
                                ("in_z", "linear_attn.in_proj_z"),
                                ("out", "linear_attn.out_proj")):
                _load_compressed(weights, reader, prefix + site, entry, short,
                                 sites[prefix + site], device)
            for short, name in (("a_log", "linear_attn.A_log"),
                                ("dt_bias", "linear_attn.dt_bias"),
                                ("in_a", "linear_attn.in_proj_a.weight"),
                                ("in_b", "linear_attn.in_proj_b.weight"),
                                ("conv", "linear_attn.conv1d.weight"),
                                ("gdn_norm", "linear_attn.norm.weight")):
                entry[short] = _place(
                    weights, reader.get(prefix + name), device,
                    torch.bfloat16)
        else:
            for short, site in (("q", "self_attn.q_proj"),
                                ("k", "self_attn.k_proj"),
                                ("v", "self_attn.v_proj"),
                                ("o", "self_attn.o_proj")):
                _load_compressed(weights, reader, prefix + site, entry, short,
                                 sites[prefix + site], device)
            for short, name in (("q_norm", "self_attn.q_norm.weight"),
                                ("k_norm", "self_attn.k_norm.weight")):
                entry[short] = _place(
                    weights, reader.get(prefix + name), device,
                    torch.bfloat16)

        weights.layers.append(entry)

    return weights
