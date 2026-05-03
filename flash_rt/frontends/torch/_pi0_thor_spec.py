"""Declarative weight spec for Pi0TorchFrontendThor (stage 7.4).

Pi0 shares the SigLIP + Paligemma encoder blocks with Pi0.5 (imported
from ``_thor_spec_common``). The decoder differs: Pi0 fuses the standard
RMSNorm weight into QKV (input_layernorm) and Gate/Up
(post_attention_layernorm), matching the encoder pattern — Pi0.5 does
not (uses AdaRMSNorm Dense instead). Pi0 also has no per-layer
modulation Dense block.

Op order byte-for-byte equivalent to the original pi0_thor loader
(lines 365-394 pre-refactor). Non-block singletons (action proj,
state_proj, action_time_mlp, final_norm) stay inline in the frontend.
"""

from __future__ import annotations

from flash_rt.executors.weight_loader import Item, LayerBlock, ModelWeightSpec
from flash_rt.executors.torch_weights import (
    FlatCat,
    FusedGateUp,
    FusedQKV,
    Quant,
    ToFp16,
    tT,
)
from flash_rt.frontends.torch._thor_spec_common import (
    paligemma_encoder_block,
    paligemma_siglip_block,
)


def _decoder_block() -> LayerBlock:
    """Pi0 decoder (18 layers, cuBLASLt path, with norm fuse).

    Matches pi0_thor.py:365-394 exactly:
      qkv : FusedQKV(interleave 8/1, norm_fuse=input_layernorm)
            → .t().contiguous() → Quant
      o   : ToFp16 → .t().contiguous() → Quant
      gu  : FusedGateUp(norm_fuse=post_attention_layernorm)
            → .t().contiguous() → Quant
      d   : ToFp16 → .t().contiguous() → Quant
    """
    dp = "paligemma_with_expert.gemma_expert.model.layers.{i}"
    items = [
        Item("qkv_w",
             FusedQKV(q=f"{dp}.self_attn.q_proj.weight",
                      k=f"{dp}.self_attn.k_proj.weight",
                      v=f"{dp}.self_attn.v_proj.weight",
                      norm_fuse=f"{dp}.input_layernorm.weight",
                      interleave_q_heads=8,
                      interleave_k_heads=1),
             [tT(), Quant()],
             FlatCat("_dec_qkv_flat"), scale_into="_ae_w_scales"),
        Item("o_w", f"{dp}.self_attn.o_proj.weight",
             [ToFp16(), tT(), Quant()],
             FlatCat("_dec_o_flat"), scale_into="_ae_w_scales"),
        Item("gu_w",
             FusedGateUp(gate=f"{dp}.mlp.gate_proj.weight",
                         up=f"{dp}.mlp.up_proj.weight",
                         norm_fuse=f"{dp}.post_attention_layernorm.weight"),
             [tT(), Quant()],
             FlatCat("_dec_gu_flat"), scale_into="_ae_w_scales"),
        Item("d_w", f"{dp}.mlp.down_proj.weight",
             [ToFp16(), tT(), Quant()],
             FlatCat("_dec_d_flat"), scale_into="_ae_w_scales"),
    ]
    return LayerBlock(prefix_fmt="", num_layers=18, items=items, name="decoder")


def build_spec() -> ModelWeightSpec:
    return ModelWeightSpec(
        framework="torch",
        blocks=[
            paligemma_siglip_block(),
            paligemma_encoder_block(),
            _decoder_block(),
        ],
    )


__all__ = ["build_spec"]
