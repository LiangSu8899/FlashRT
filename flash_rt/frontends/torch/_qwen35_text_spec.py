"""Geometry and quantization contract for a ``qwen3_5_text`` checkpoint.

Everything here is derived from ``config.json`` rather than written down,
because the family ships at several sizes and the shapes that matter are not
all obvious from the parameter count. ``q_proj`` is twice the query width when
``attn_output_gate`` is set, the gated-DeltaNet projection is
``2 * key + value`` wide, and the rotary embedding covers a fraction of the
head. A constant for one size is a wrong answer for the next.

The quantization contract is checked rather than assumed. The packed 4-bit
kernels decode a specific layout -- offset-binary nibbles, one scale per group
along K, no reordering -- and a checkpoint that violates any of it would still
load and still produce plausible numbers, so each property is refused by name.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

MODEL_TYPE = "qwen3_5_text"
# A compressed site is named for its module; the tensors hang off it.
PACKED_SUFFIX = ".weight_packed"
SCALE_SUFFIX = ".weight_scale"
# Group sizes producers publish, and which the packed kernels dispatch on.
SUPPORTED_GROUP_SIZES = (32, 64, 128)


class CheckpointContractError(ValueError):
    """The checkpoint is not what this runtime decodes."""


@dataclass(frozen=True)
class TextDims:
    """Derived geometry of one checkpoint."""

    hidden: int
    num_layers: int
    vocab_size: int
    intermediate: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    layer_types: tuple[str, ...]

    # full attention
    q_heads: int
    kv_heads: int
    head_dim: int
    attn_output_gate: bool
    rope_theta: float
    partial_rotary_factor: float

    # gated DeltaNet
    lin_key_heads: int
    lin_value_heads: int
    lin_key_head_dim: int
    lin_value_head_dim: int
    lin_conv_kernel: int

    @property
    def q_width(self) -> int:
        """Rows of ``q_proj``: the gate rides in the same projection."""
        width = self.q_heads * self.head_dim
        return 2 * width if self.attn_output_gate else width

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def attn_width(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def lin_key_width(self) -> int:
        return self.lin_key_heads * self.lin_key_head_dim

    @property
    def lin_value_width(self) -> int:
        return self.lin_value_heads * self.lin_value_head_dim

    @property
    def lin_qkv_width(self) -> int:
        """Query, key and value share one projection: 2 * key + value."""
        return 2 * self.lin_key_width + self.lin_value_width

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.layer_types)
                     if kind == "full_attention")

    @property
    def linear_attention_layers(self) -> tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.layer_types)
                     if kind == "linear_attention")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TextDims":
        text = config.get("text_config", config)
        model_type = text.get("model_type", config.get("model_type"))
        if model_type != MODEL_TYPE:
            raise CheckpointContractError(
                f"expected model_type={MODEL_TYPE!r}, got {model_type!r}")

        layer_types = tuple(text.get("layer_types") or ())
        num_layers = int(text["num_hidden_layers"])
        if len(layer_types) != num_layers:
            raise CheckpointContractError(
                f"layer_types lists {len(layer_types)} entries for "
                f"{num_layers} layers")
        unknown = sorted(set(layer_types) - {"full_attention",
                                             "linear_attention"})
        if unknown:
            raise CheckpointContractError(
                f"unknown layer types {unknown}; this runtime implements "
                "full_attention and linear_attention")

        rope = text.get("rope_parameters") or {}
        return cls(
            hidden=int(text["hidden_size"]),
            num_layers=num_layers,
            vocab_size=int(text["vocab_size"]),
            intermediate=int(text["intermediate_size"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
            layer_types=layer_types,
            q_heads=int(text["num_attention_heads"]),
            kv_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            attn_output_gate=bool(text.get("attn_output_gate", False)),
            rope_theta=float(rope.get("rope_theta", 10000.0)),
            partial_rotary_factor=float(
                rope.get("partial_rotary_factor", 1.0)),
            lin_key_heads=int(text["linear_num_key_heads"]),
            lin_value_heads=int(text["linear_num_value_heads"]),
            lin_key_head_dim=int(text["linear_key_head_dim"]),
            lin_value_head_dim=int(text["linear_value_head_dim"]),
            lin_conv_kernel=int(text["linear_conv_kernel_dim"]),
        )


@dataclass(frozen=True)
class QuantSpec:
    """What the weight compression is, once it has been agreed to."""

    group_size: int
    num_bits: int = 4

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QuantSpec":
        quant = config.get("quantization_config")
        if not isinstance(quant, dict):
            raise CheckpointContractError(
                "no quantization_config; this runtime loads a compressed "
                "checkpoint and does not quantize one itself")
        if quant.get("format") != "pack-quantized":
            raise CheckpointContractError(
                f"weight format is {quant.get('format')!r}; the packed 4-bit "
                "kernels read 'pack-quantized'")

        groups = quant.get("config_groups") or {}
        if len(groups) != 1:
            raise CheckpointContractError(
                f"{len(groups)} quantization groups; this runtime decodes one "
                "scheme for every linear weight")
        weights = next(iter(groups.values())).get("weights") or {}

        # Each of these is a property the kernel's decode depends on, and each
        # would otherwise surface as plausible-looking output.
        problems = []
        if weights.get("num_bits") != 4:
            problems.append(f"num_bits={weights.get('num_bits')} (want 4)")
        if weights.get("type") != "int":
            problems.append(f"type={weights.get('type')!r} (want 'int')")
        if not weights.get("symmetric", False):
            problems.append("symmetric=False; the decode has no zero point")
        if weights.get("strategy") != "group":
            problems.append(
                f"strategy={weights.get('strategy')!r} (want 'group')")
        if weights.get("actorder"):
            problems.append(
                f"actorder={weights.get('actorder')!r}; the decode applies no "
                "column permutation")
        if weights.get("block_structure"):
            problems.append(
                f"block_structure={weights.get('block_structure')!r}")
        group_size = weights.get("group_size")
        if group_size not in SUPPORTED_GROUP_SIZES:
            problems.append(
                f"group_size={group_size} (want one of "
                f"{list(SUPPORTED_GROUP_SIZES)})")
        activations = next(iter(groups.values())).get("input_activations")
        if activations is not None and not activations.get("dynamic", True):
            problems.append(
                "input_activations are quantized; these kernels are "
                "weight-only")
        if problems:
            raise CheckpointContractError(
                "the weight quantization is not what the packed 4-bit kernels "
                "decode: " + "; ".join(problems))
        return cls(group_size=int(group_size))


def _linear_sites(dims: TextDims, layer: int) -> dict[str, tuple[int, int]]:
    """Compressed weights of one layer, as logical (out, in) shapes."""
    prefix = f"model.language_model.layers.{layer}."
    sites = {
        prefix + "mlp.gate_proj": (dims.intermediate, dims.hidden),
        prefix + "mlp.up_proj": (dims.intermediate, dims.hidden),
        prefix + "mlp.down_proj": (dims.hidden, dims.intermediate),
    }
    if dims.layer_types[layer] == "full_attention":
        sites.update({
            prefix + "self_attn.q_proj": (dims.q_width, dims.hidden),
            prefix + "self_attn.k_proj": (dims.kv_width, dims.hidden),
            prefix + "self_attn.v_proj": (dims.kv_width, dims.hidden),
            prefix + "self_attn.o_proj": (dims.hidden, dims.attn_width),
        })
    else:
        sites.update({
            prefix + "linear_attn.in_proj_qkv": (
                dims.lin_qkv_width, dims.hidden),
            prefix + "linear_attn.in_proj_z": (
                dims.lin_value_width, dims.hidden),
            prefix + "linear_attn.out_proj": (
                dims.hidden, dims.lin_value_width),
        })
    return sites


def compressed_sites(dims: TextDims) -> dict[str, tuple[int, int]]:
    """Every packed weight in the text backbone, with its logical shape."""
    sites: dict[str, tuple[int, int]] = {}
    for layer in range(dims.num_layers):
        sites.update(_linear_sites(dims, layer))
    return sites


def plain_sites(dims: TextDims) -> dict[str, tuple[int, ...]]:
    """Tensors the producer leaves uncompressed, with their shapes.

    The decay parameters of the recurrence are among them, and deliberately:
    they set how the state forgets rather than contributing to a product, so
    quantizing them changes the dynamics instead of adding noise.
    """
    sites: dict[str, tuple[int, ...]] = {
        "model.language_model.embed_tokens.weight": (
            dims.vocab_size, dims.hidden),
        "model.language_model.norm.weight": (dims.hidden,),
    }
    for layer in range(dims.num_layers):
        prefix = f"model.language_model.layers.{layer}."
        sites[prefix + "input_layernorm.weight"] = (dims.hidden,)
        sites[prefix + "post_attention_layernorm.weight"] = (dims.hidden,)
        if dims.layer_types[layer] == "full_attention":
            sites[prefix + "self_attn.q_norm.weight"] = (dims.head_dim,)
            sites[prefix + "self_attn.k_norm.weight"] = (dims.head_dim,)
        else:
            sites[prefix + "linear_attn.A_log"] = (dims.lin_value_heads,)
            sites[prefix + "linear_attn.dt_bias"] = (dims.lin_value_heads,)
            sites[prefix + "linear_attn.in_proj_a.weight"] = (
                dims.lin_value_heads, dims.hidden)
            sites[prefix + "linear_attn.in_proj_b.weight"] = (
                dims.lin_value_heads, dims.hidden)
            sites[prefix + "linear_attn.conv1d.weight"] = (
                dims.lin_qkv_width, 1, dims.lin_conv_kernel)
            sites[prefix + "linear_attn.norm.weight"] = (
                dims.lin_value_head_dim,)
    return sites


def validate_checkpoint(path: str) -> dict[str, Any]:
    """Refuse a checkpoint this runtime would decode wrongly.

    Returns the geometry and quantization spec on success, so a caller that
    validates has no reason to parse the config a second time.
    """
    path = os.path.abspath(os.fspath(path))
    config_path = os.path.join(path, "config.json")
    index_path = os.path.join(path, "model.safetensors.index.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"checkpoint is missing {config_path!r}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    dims = TextDims.from_config(config)
    quant = QuantSpec.from_config(config)

    for name, (rows, columns) in compressed_sites(dims).items():
        if columns % quant.group_size:
            raise CheckpointContractError(
                f"{name} has K={columns}, which group_size="
                f"{quant.group_size} does not divide")

    # A single-shard checkpoint has no index; both layouts are ordinary.
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map") or {}
        available = set(weight_map)
    else:
        shard = os.path.join(path, "model.safetensors")
        if not os.path.isfile(shard):
            raise FileNotFoundError(
                f"checkpoint has neither {index_path!r} nor {shard!r}")
        from safetensors import safe_open

        with safe_open(shard, framework="pt", device="cpu") as reader:
            available = set(reader.keys())

    required = {name + PACKED_SUFFIX for name in compressed_sites(dims)}
    required |= {name + SCALE_SUFFIX for name in compressed_sites(dims)}
    required |= set(plain_sites(dims))
    if not dims.tie_word_embeddings:
        required.add("lm_head" + PACKED_SUFFIX)
    missing = sorted(required.difference(available))
    if missing:
        preview = ", ".join(missing[:8])
        if len(missing) > 8:
            preview += f", ... ({len(missing)} missing)"
        raise CheckpointContractError(
            "checkpoint is missing text backbone tensors: " + preview)

    return {
        "checkpoint_path": path,
        "dims": dims,
        "quant": quant,
        "tensor_count": len(available),
    }
