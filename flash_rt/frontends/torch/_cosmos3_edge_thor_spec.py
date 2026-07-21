"""Cosmos3-Edge Thor weight specification.

The public Edge checkpoint is a diffusers-style directory. This file is the
single source of truth for mapping those tensor names onto the FlashRT Thor
denoise engine. It is intentionally explicit: in the public Edge checkpoint as
loaded by Cosmos Framework, ``to_*`` maps to the understanding/causal tower and
``add_*`` maps to the generation/full tower. The latter also owns
``norm_added_{q,k}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Cosmos3EdgeThorSpec:
    num_layers: int = 28
    hidden_size: int = 2048
    ffn_size: int = 9216
    num_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    patch_latent_dim: int = 192
    latent_channels: int = 48
    action_dim: int = 64
    num_action_domains: int = 32
    vocab_size: int = 131072
    rms_eps: float = 1e-5

    @property
    def q_width(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_kv_heads * self.head_dim


SPEC = Cosmos3EdgeThorSpec()


GLOBAL_SHAPES: dict[str, tuple[int, ...]] = {
    "action_modality_embed": (SPEC.hidden_size,),
    "action_proj_in.bias.weight": (SPEC.num_action_domains, SPEC.hidden_size),
    "action_proj_in.fc.weight": (
        SPEC.num_action_domains,
        SPEC.action_dim * SPEC.hidden_size,
    ),
    "action_proj_out.bias.weight": (SPEC.num_action_domains, SPEC.action_dim),
    "action_proj_out.fc.weight": (
        SPEC.num_action_domains,
        SPEC.action_dim * SPEC.hidden_size,
    ),
    "embed_tokens.weight": (SPEC.vocab_size, SPEC.hidden_size),
    "lm_head.weight": (SPEC.vocab_size, SPEC.hidden_size),
    "norm.weight": (SPEC.hidden_size,),
    "norm_moe_gen.weight": (SPEC.hidden_size,),
    "proj_in.bias": (SPEC.hidden_size,),
    "proj_in.weight": (SPEC.hidden_size, SPEC.patch_latent_dim),
    "proj_out.bias": (SPEC.patch_latent_dim,),
    "proj_out.weight": (SPEC.patch_latent_dim, SPEC.hidden_size),
    "time_embedder.linear_1.bias": (SPEC.hidden_size,),
    "time_embedder.linear_1.weight": (SPEC.hidden_size, 256),
    "time_embedder.linear_2.bias": (SPEC.hidden_size,),
    "time_embedder.linear_2.weight": (SPEC.hidden_size, SPEC.hidden_size),
}


LAYER_SHAPES: dict[str, tuple[int, ...]] = {
    # Und tower norms/MLP.
    "input_layernorm.weight": (SPEC.hidden_size,),
    "post_attention_layernorm.weight": (SPEC.hidden_size,),
    "mlp.up_proj.weight": (SPEC.ffn_size, SPEC.hidden_size),
    "mlp.down_proj.weight": (SPEC.hidden_size, SPEC.ffn_size),
    # Gen tower norms/MLP.
    "input_layernorm_moe_gen.weight": (SPEC.hidden_size,),
    "post_attention_layernorm_moe_gen.weight": (SPEC.hidden_size,),
    "mlp_moe_gen.up_proj.weight": (SPEC.ffn_size, SPEC.hidden_size),
    "mlp_moe_gen.down_proj.weight": (SPEC.hidden_size, SPEC.ffn_size),
    # Und tower attention: diffusers ``to_*``.
    "self_attn.to_q.weight": (SPEC.q_width, SPEC.hidden_size),
    "self_attn.to_k.weight": (SPEC.kv_width, SPEC.hidden_size),
    "self_attn.to_v.weight": (SPEC.kv_width, SPEC.hidden_size),
    "self_attn.to_out.weight": (SPEC.hidden_size, SPEC.hidden_size),
    # Gen tower attention: diffusers ``add_*``.
    "self_attn.add_q_proj.weight": (SPEC.q_width, SPEC.hidden_size),
    "self_attn.add_k_proj.weight": (SPEC.kv_width, SPEC.hidden_size),
    "self_attn.add_v_proj.weight": (SPEC.kv_width, SPEC.hidden_size),
    "self_attn.to_add_out.weight": (SPEC.hidden_size, SPEC.hidden_size),
    "self_attn.norm_added_q.weight": (SPEC.head_dim,),
    "self_attn.norm_added_k.weight": (SPEC.head_dim,),
    # Edge-only: gen tower consumes und K through this extra normalization.
    "self_attn.k_norm_und_for_gen.weight": (SPEC.head_dim,),
}


GEN_ATTENTION_KEYS = (
    "self_attn.add_q_proj.weight",
    "self_attn.add_k_proj.weight",
    "self_attn.add_v_proj.weight",
    "self_attn.to_add_out.weight",
)

UND_ATTENTION_KEYS = (
    "self_attn.to_q.weight",
    "self_attn.to_k.weight",
    "self_attn.to_v.weight",
    "self_attn.to_out.weight",
)

GEN_MLP_KEYS = (
    "mlp_moe_gen.up_proj.weight",
    "mlp_moe_gen.down_proj.weight",
)

UND_MLP_KEYS = (
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def layer_key(layer: int, suffix: str) -> str:
    if layer < 0 or layer >= SPEC.num_layers:
        raise ValueError(f"layer index out of range: {layer}")
    return f"layers.{layer}.{suffix}"


def iter_expected_shapes() -> Iterable[tuple[str, tuple[int, ...]]]:
    yield from GLOBAL_SHAPES.items()
    for layer in range(SPEC.num_layers):
        for suffix, shape in LAYER_SHAPES.items():
            yield layer_key(layer, suffix), shape


def transformer_index_path(checkpoint: str | Path) -> Path:
    root = Path(checkpoint)
    return root / "transformer" / "diffusion_pytorch_model.safetensors.index.json"


def load_transformer_weight_map(checkpoint: str | Path) -> dict[str, str]:
    path = transformer_index_path(checkpoint)
    data = json.loads(path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{path} does not contain a weight_map.")
    return {str(k): str(v) for k, v in weight_map.items()}


def validate_transformer_index(checkpoint: str | Path) -> None:
    weight_map = load_transformer_weight_map(checkpoint)
    expected = dict(iter_expected_shapes())
    missing = sorted(set(expected) - set(weight_map))
    extra = sorted(set(weight_map) - set(expected))
    if missing or extra:
        raise ValueError(
            "Cosmos3-Edge transformer index mismatch: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )


def validate_transformer_shapes(checkpoint: str | Path) -> None:
    from safetensors import safe_open

    root = Path(checkpoint)
    weight_map = load_transformer_weight_map(root)
    for key, expected_shape in iter_expected_shapes():
        shard = root / "transformer" / weight_map[key]
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            actual_shape = tuple(f.get_slice(key).get_shape())
        if actual_shape != expected_shape:
            raise ValueError(
                f"{key} shape mismatch: expected {expected_shape}, got {actual_shape}"
            )
