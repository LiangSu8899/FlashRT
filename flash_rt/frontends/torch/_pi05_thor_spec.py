"""Declarative weight spec for Pi05TorchFrontendThor.

SigLIP + Paligemma encoder blocks come from ``_thor_spec_common``; the
Pi0.5-specific pieces are the Gemma-expert decoder (no norm fuse, AE
cuBLASLt path) and the per-layer AdaRMS modulation Dense layers.

Op order inside each Item is byte-for-byte equivalent to the original
loader; see ``docs/v2/stage7_weight_loader.md`` §7.
"""

from __future__ import annotations

from flash_vla.executors.weight_loader import Item, LayerBlock, ModelWeightSpec
from flash_vla.executors.torch_weights import (
    FlatCat,
    FusedGateUp,
    FusedQKV,
    Quant,
    TensorList,
    ToFp16,
    tT,
)
from flash_vla.frontends.torch._thor_spec_common import (
    paligemma_encoder_block,
    paligemma_siglip_block,
)


def _decoder_block() -> LayerBlock:
    """Gemma-expert decoder / action expert (18 layers, cuBLASLt path).

    Unlike Pi0, Pi0.5 does **not** fuse the per-layer norm weight into
    QKV/GateUp — modulation happens via AdaRMSNorm Dense layers instead.
    Weights go through ``.t().contiguous()`` + FP8 quant, then FlatCat
    into ``_dec_{qkv,o,gu,d}_flat``.
    """
    dp = "paligemma_with_expert.gemma_expert.model.layers.{i}"
    items = [
        Item("qkv_w",
             FusedQKV(q=f"{dp}.self_attn.q_proj.weight",
                      k=f"{dp}.self_attn.k_proj.weight",
                      v=f"{dp}.self_attn.v_proj.weight",
                      interleave_q_heads=8,
                      interleave_k_heads=1),
             [tT(), Quant()],
             FlatCat("_dec_qkv_flat"), scale_into="_ae_w_scales"),
        Item("o_w", f"{dp}.self_attn.o_proj.weight",
             [ToFp16(), tT(), Quant()],
             FlatCat("_dec_o_flat"), scale_into="_ae_w_scales"),
        Item("gu_w",
             FusedGateUp(gate=f"{dp}.mlp.gate_proj.weight",
                         up=f"{dp}.mlp.up_proj.weight"),
             [tT(), Quant()],
             FlatCat("_dec_gu_flat"), scale_into="_ae_w_scales"),
        Item("d_w", f"{dp}.mlp.down_proj.weight",
             [ToFp16(), tT(), Quant()],
             FlatCat("_dec_d_flat"), scale_into="_ae_w_scales"),
    ]
    return LayerBlock(prefix_fmt="", num_layers=18, items=items, name="decoder")


def _decoder_mods_block() -> LayerBlock:
    """Per-layer AdaRMS modulation Dense layers (Pi0.5-only)."""
    dp = "paligemma_with_expert.gemma_expert.model.layers.{i}"
    items = [
        Item("attn_mod_w", f"{dp}.input_layernorm.dense.weight",
             [ToFp16()], TensorList("_attn_mod_w")),
        Item("attn_mod_b", f"{dp}.input_layernorm.dense.bias",
             [ToFp16()], TensorList("_attn_mod_b")),
        Item("ffn_mod_w",  f"{dp}.post_attention_layernorm.dense.weight",
             [ToFp16()], TensorList("_ffn_mod_w")),
        Item("ffn_mod_b",  f"{dp}.post_attention_layernorm.dense.bias",
             [ToFp16()], TensorList("_ffn_mod_b")),
    ]
    return LayerBlock(prefix_fmt="", num_layers=18, items=items, name="decoder_mods")


def build_spec() -> ModelWeightSpec:
    return ModelWeightSpec(
        framework="torch",
        blocks=[
            paligemma_siglip_block(),
            paligemma_encoder_block(),
            _decoder_block(),
            _decoder_mods_block(),
        ],
    )


__all__ = ["build_spec"]
