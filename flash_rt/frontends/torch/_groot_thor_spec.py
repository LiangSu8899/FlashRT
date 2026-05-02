"""Declarative weight spec for GrootTorchFrontendThor (stage 7.8).

**Partial migration**: this stage moves only the SigLIP2 vision encoder
block (27 layers × 12 items) to declarative spec. The Qwen3 block
produces parallel fp16/fp8 outputs into a ``_qwen3_w`` dict and the
DiT block has self/cross-attn branching with dual outputs — both
patterns are awkward fits for the current sink abstraction and are
worth a separate evaluation stage before touching GROOT's
bit-identical 0.998607 baseline. Embodiment weights are singletons
indexed by ``embodiment_id`` rather than per-layer, so they do not
match the LayerBlock shape at all.

SigLIP2 layout is identical in schema to Pi0.5/Pi0 torch SigLIP —
27 layers × (LN1, LN2, QKV, O, FC1, FC2) — the only differences are:
  * Key prefix uses the ``backbone.model.vision_model.vision_model``
    path (GROOT's double vision_model).
  * Biases go through ``.to(fp16)`` before ``Cat`` (the legacy loader
    does the cat on raw dtypes then ``.to(fp16)``; ``Cat(dtype=fp16)``
    casts each part to fp16 before ``torch.cat``, which is
    bit-equivalent for fp16 output).

Op order inside each Item is byte-for-byte equivalent to
``groot_thor.py::_load_siglip2_weights`` lines 221-255 pre-refactor.
"""

from __future__ import annotations

from flash_rt.executors.weight_loader import Item, LayerBlock, ModelWeightSpec
from flash_rt.executors.torch_weights import (
    Cat,
    Quant,
    T,
    TensorList,
    ToFp16,
)


# Matches VIS_PREFIX constant in groot_thor.py.
_VP = "backbone.model.vision_model.vision_model.encoder.layers.{i}"


def _siglip2_block() -> LayerBlock:
    items = [
        Item("ln_attn_w", f"{_VP}.layer_norm1.weight", [ToFp16()],
             TensorList("_sig_ln_attn_w")),
        Item("ln_attn_b", f"{_VP}.layer_norm1.bias",   [ToFp16()],
             TensorList("_sig_ln_attn_b")),

        # QKV: cat(q,k,v) -> .T.contiguous() -> FP8
        Item("qkv_w",
             Cat([f"{_VP}.self_attn.q_proj.weight",
                  f"{_VP}.self_attn.k_proj.weight",
                  f"{_VP}.self_attn.v_proj.weight"], dim=0),
             [T(), Quant()],
             TensorList("_sig_qkv_w"), scale_into="_sig_alpha"),
        Item("qkv_b",
             Cat([f"{_VP}.self_attn.q_proj.bias",
                  f"{_VP}.self_attn.k_proj.bias",
                  f"{_VP}.self_attn.v_proj.bias"], dim=0),
             [],
             TensorList("_sig_qkv_b")),

        # O
        Item("o_w", f"{_VP}.self_attn.out_proj.weight",
             [ToFp16(), T(), Quant()],
             TensorList("_sig_o_w"), scale_into="_sig_alpha"),
        Item("o_b", f"{_VP}.self_attn.out_proj.bias",
             [ToFp16()], TensorList("_sig_o_b")),

        # FFN LN (order: legacy emits LN1 first, QKV, O, LN2, up, down —
        # keep this ordering exactly so scale_into collects [q, o, u, d]
        # per layer to match legacy sig_alpha).
        Item("ln_ffn_w", f"{_VP}.layer_norm2.weight", [ToFp16()],
             TensorList("_sig_ln_ffn_w")),
        Item("ln_ffn_b", f"{_VP}.layer_norm2.bias",   [ToFp16()],
             TensorList("_sig_ln_ffn_b")),

        # FFN up (fc1)
        Item("up_w", f"{_VP}.mlp.fc1.weight",
             [ToFp16(), T(), Quant()],
             TensorList("_sig_up_w"), scale_into="_sig_alpha"),
        Item("up_b", f"{_VP}.mlp.fc1.bias",
             [ToFp16()], TensorList("_sig_up_b")),

        # FFN down (fc2)
        Item("down_w", f"{_VP}.mlp.fc2.weight",
             [ToFp16(), T(), Quant()],
             TensorList("_sig_down_w"), scale_into="_sig_alpha"),
        Item("down_b", f"{_VP}.mlp.fc2.bias",
             [ToFp16()], TensorList("_sig_down_b")),
    ]
    return LayerBlock(prefix_fmt="", num_layers=27, items=items, name="siglip2")


def build_siglip2_spec() -> ModelWeightSpec:
    """Spec covering only the SigLIP2 block.

    Callers supply the in-memory ``state_dict`` via ``DictSource`` (the
    GROOT loader already reads all safetensors into a single dict, so
    the loader reads from that rather than opening the file again).
    """
    return ModelWeightSpec(framework="torch", blocks=[_siglip2_block()])


__all__ = ["build_siglip2_spec"]
