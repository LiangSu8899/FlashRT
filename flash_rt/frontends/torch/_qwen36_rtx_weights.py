"""FlashRT -- Qwen3.6 (RTX) weight extractor.

Walks an HF ``Qwen3_5ForCausalLM`` instance and pulls every tensor
the upcoming own-forward needs, returning a flat dict of int device
pointers. The HF model stays loaded as a weight container; nothing
about its forward() is invoked.

Per-layer roles (see :func:`extract_weights` for the full schema):

  Common (every layer):
    * ``input_norm_eff_w``        — bf16 (5120,) pre-applied (1+w)
    * ``post_attn_norm_eff_w``    — bf16 (5120,) pre-applied (1+w)
    * ``mlp_gate_w/_s``           — fp8 (17408, 5120) + fp32 scale
    * ``mlp_up_w/_s``             — fp8 (17408, 5120) + fp32 scale
    * ``mlp_down_w/_s``           — fp8 (5120, 17408) + fp32 scale

  Linear-attn layers only (48 layers):
    * ``in_proj_qkv_w/_s``        — fp8 (10240, 5120) + scale (q+k+v fused)
    * ``in_proj_z_w/_s``          — fp8 (6144, 5120)  + scale (output-gate z)
    * ``in_proj_a_w``             — bf16 (48, 5120)   recurrent gate raw
    * ``in_proj_b_w``             — bf16 (48, 5120)   beta
    * ``out_proj_w/_s``           — fp8 (5120, 6144)  + fp32 scale
    * ``conv1d_w``                — bf16 (10240, 4)   depthwise (squeezed)
    * ``conv1d_b``                — bf16 (10240,)     or 0 if no bias
    * ``head_norm_w``             — bf16 (128,) plain weight (no (1+w))
    * ``A_log``                   — bf16 (48,)
    * ``dt_bias``                 — bf16 (48,)

  Full-attn layers only (16 layers):
    * ``q_proj_w/_s``             — fp8 (12288, 5120) (Q + output-gate fused)
    * ``k_proj_w/_s``             — fp8 (1024, 5120)  + fp32 scale
    * ``v_proj_w/_s``             — fp8 (1024, 5120)  + fp32 scale
    * ``o_proj_w/_s``             — fp8 (5120, 6144)  + fp32 scale
    * ``q_norm_eff_w``            — bf16 (256,)       (1+w) precomputed
    * ``k_norm_eff_w``            — bf16 (256,)       (1+w) precomputed

  Top level:
    * ``embed_w``                 — bf16 (248320, 5120)
    * ``final_norm_eff_w``        — bf16 (5120,)      (1+w) precomputed
    * ``lm_head_w``               — bf16 (248320, 5120)
    * ``layer_types``             — list[str] of length 64
    * ``vocab_size``, ``hidden``  — int

Extractor design:

  * Returns ``WeightHandles`` with two attributes:
      - ``ptrs``    — dict (or list of dicts for layers) of int ptrs
      - ``anchors`` — list[Tensor] keeping every referenced tensor
                      alive so its data_ptr remains valid across
                      forward calls.
  * No new allocations on existing FP8 weight tensors -- we keep the
    HF tensor pointers as-is. Only the (1+w) RMSNorm precompute and
    the bf16 weight_scale_inv -> fp32 widening allocate new tensors,
    each O(few KB)..O(few MB) and one-time at load.
  * The ``q_proj`` "fused" output dim of 12288 is left fused at the
    weight side -- the pipeline decides at forward time how to slice
    Q (first 6144 cols) vs output-gate (last 6144). No reshape here.
  * Linear-attn ``conv1d.weight`` is shape (10240, 1, 4); we squeeze
    the size-1 channel axis to (10240, 4) so the fvk
    ``causal_conv1d_qwen36_update`` kernel can read it directly. The
    underlying memory is shared with HF's tensor (no copy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class WeightHandles:
    """Container for extracted weight pointers.

    ``ptrs`` carries the int device pointers + scalar metadata the
    own-forward consumes. ``anchors`` keeps every referenced tensor
    pinned so the pointers stay live for the lifetime of the
    frontend. Anchor tensors are inserted in a deterministic order
    (top-level first, then per-layer) so a future hash check can
    detect accidental tensor identity changes between calls.
    """

    ptrs: dict = field(default_factory=dict)
    anchors: list = field(default_factory=list)


def _eff_rmsnorm_weight(w: torch.Tensor) -> torch.Tensor:
    """(1 + w).bf16().contiguous() -- Qwen3_5RMSNorm quirk."""
    return (1.0 + w.float()).to(torch.bfloat16).contiguous()


def _fp32_scale(s: torch.Tensor) -> torch.Tensor:
    """Cast bf16 weight_scale_inv to fp32 contiguous."""
    return s.to(torch.float32).contiguous()


def _ensure_anchored(handles: WeightHandles, t: torch.Tensor) -> int:
    """Append tensor to anchors and return its data_ptr."""
    handles.anchors.append(t)
    return int(t.data_ptr())


def _extract_fp8_linear(out_ptrs: dict, anchors: list, prefix: str,
                        module: torch.nn.Module) -> None:
    """Pull (weight, weight_scale_inv -> fp32) into out_ptrs[prefix + '_w/_s'].

    Module is expected to be a transformers FP8Linear with .weight
    (e4m3fn) and .weight_scale_inv (bf16 or fp32). The fp32 scale
    cast is a one-time O(N/128 * K/128) tensor; the FP8 weight is
    referenced as-is. Tensors are appended to ``anchors`` so the
    pointers stay live.
    """
    w = module.weight
    anchors.append(w)
    out_ptrs[prefix + '_w'] = int(w.data_ptr())
    s_fp32 = _fp32_scale(module.weight_scale_inv)
    anchors.append(s_fp32)
    out_ptrs[prefix + '_s'] = int(s_fp32.data_ptr())


def _extract_bf16_weight(handles: WeightHandles, name: str,
                         t: torch.Tensor) -> None:
    handles.ptrs[name] = _ensure_anchored(handles, t.contiguous())


def _extract_layer(handles: WeightHandles, layer_idx: int,
                   layer: torch.nn.Module, layer_type: str,
                   per_layer: list[dict]) -> None:
    """Populate per_layer[layer_idx] with this layer's pointers."""
    out: dict = {'type': layer_type}

    # Common: pre/post layernorms (Qwen3_5RMSNorm with (1+w) quirk).
    in_eff = _eff_rmsnorm_weight(layer.input_layernorm.weight)
    post_eff = _eff_rmsnorm_weight(layer.post_attention_layernorm.weight)
    out['input_norm_eff_w'] = _ensure_anchored(handles, in_eff)
    out['post_attn_norm_eff_w'] = _ensure_anchored(handles, post_eff)

    # Common MLP (dense SwiGLU, FP8).
    _scope: list[tuple[str, torch.nn.Module]] = [
        ('mlp_gate', layer.mlp.gate_proj),
        ('mlp_up', layer.mlp.up_proj),
        ('mlp_down', layer.mlp.down_proj),
    ]
    for name, mod in _scope:
        _extract_fp8_linear(out, handles.anchors, name, mod)

    if layer_type == 'linear_attention':
        la = layer.linear_attn
        for name, mod in (
            ('in_proj_qkv', la.in_proj_qkv),
            ('in_proj_z', la.in_proj_z),
            ('out_proj', la.out_proj),
        ):
            _extract_fp8_linear(out, handles.anchors, name, mod)

        # bf16 a/b projections (no scale -- weight is bf16 already)
        out['in_proj_a_w'] = _ensure_anchored(
            handles, la.in_proj_a.weight.contiguous())
        out['in_proj_b_w'] = _ensure_anchored(
            handles, la.in_proj_b.weight.contiguous())

        # conv1d: (10240, 1, 4) -> squeeze to (10240, 4)
        conv_w = la.conv1d.weight.squeeze(1).contiguous()
        out['conv1d_w'] = _ensure_anchored(handles, conv_w)
        if la.conv1d.bias is not None:
            out['conv1d_b'] = _ensure_anchored(
                handles, la.conv1d.bias.contiguous())
        else:
            out['conv1d_b'] = 0

        # head-level RMSNormGated (plain weight, no (1+))
        out['head_norm_w'] = _ensure_anchored(
            handles, la.norm.weight.contiguous())

        out['A_log'] = _ensure_anchored(
            handles, la.A_log.detach().contiguous())
        out['dt_bias'] = _ensure_anchored(
            handles, la.dt_bias.detach().contiguous())
        # Phase 4.1 precomputes for in-place g computation:
        #   g_f32 = -A_log.float().exp() * softplus(a.float() + dt_bias)
        # The constant `-A_log.float().exp()` is per-layer (fp32, shape
        # (48,)) and never changes after weight load -- hoist it once.
        # `dt_bias.float()` is also constant.
        neg_a_log_exp = (-la.A_log.float().exp()).contiguous()
        dt_bias_fp32 = la.dt_bias.float().contiguous()
        out['neg_A_log_exp_fp32'] = _ensure_anchored(handles, neg_a_log_exp)
        out['dt_bias_fp32'] = _ensure_anchored(handles, dt_bias_fp32)
        # Phase 4.4 step 2: also expose as direct tensor refs so the
        # forward hot path can use them in `out=` ops without per-call
        # `.float()` allocations. Kept under separate keys so the
        # ptr-only schema invariants (assert_extraction_invariants) stay
        # stable for the existing ptr fields.
        out['neg_A_log_exp_fp32_t'] = neg_a_log_exp
        out['dt_bias_fp32_t'] = dt_bias_fp32

    elif layer_type == 'full_attention':
        sa = layer.self_attn
        for name, mod in (
            ('q_proj', sa.q_proj),
            ('k_proj', sa.k_proj),
            ('v_proj', sa.v_proj),
            ('o_proj', sa.o_proj),
        ):
            _extract_fp8_linear(out, handles.anchors, name, mod)

        # head-dim RMSNorms (Qwen3_5RMSNorm -- (1+w) precompute)
        q_eff = _eff_rmsnorm_weight(sa.q_norm.weight)
        k_eff = _eff_rmsnorm_weight(sa.k_norm.weight)
        out['q_norm_eff_w'] = _ensure_anchored(handles, q_eff)
        out['k_norm_eff_w'] = _ensure_anchored(handles, k_eff)
    else:
        raise ValueError(
            f'unknown layer_type {layer_type!r} at idx {layer_idx}'
        )

    per_layer[layer_idx] = out


def extract_weights(hf_model: Any) -> WeightHandles:
    """Walk ``hf_model`` and produce a :class:`WeightHandles` for Qwen3.6-27B FP8.

    Args:
      hf_model: HF ``Qwen3_5ForCausalLM`` (or ``...ForConditionalGeneration``
        whose .model + .lm_head match the same module schema).

    Returns:
      :class:`WeightHandles` -- ``ptrs`` is the dict the pipeline
      consumes; ``anchors`` is a list of tensors that MUST be retained
      for the lifetime of any forward call that reads from these
      pointers.
    """
    cfg = hf_model.config
    num_layers = cfg.num_hidden_layers
    layer_types = list(cfg.layer_types)
    if len(layer_types) != num_layers:
        raise RuntimeError(
            f'config.layer_types length {len(layer_types)} != '
            f'num_hidden_layers {num_layers}'
        )

    handles = WeightHandles()
    per_layer: list[dict] = [None] * num_layers  # type: ignore[list-item]

    # Top-level tensors.
    embed_w = hf_model.model.embed_tokens.weight
    handles.ptrs['embed_w'] = _ensure_anchored(handles, embed_w)

    final_norm_eff = _eff_rmsnorm_weight(hf_model.model.norm.weight)
    handles.ptrs['final_norm_eff_w'] = _ensure_anchored(
        handles, final_norm_eff
    )

    lm_head_w = hf_model.lm_head.weight
    handles.ptrs['lm_head_w'] = _ensure_anchored(handles, lm_head_w)
    handles.ptrs['lm_head_tied'] = bool(
        lm_head_w.data_ptr() == embed_w.data_ptr()
    )

    handles.ptrs['vocab_size'] = int(cfg.vocab_size)
    handles.ptrs['hidden'] = int(cfg.hidden_size)
    handles.ptrs['num_layers'] = int(num_layers)
    handles.ptrs['layer_types'] = layer_types

    # Per-layer.
    for L, layer in enumerate(hf_model.model.layers):
        _extract_layer(handles, L, layer, layer_types[L], per_layer)

    handles.ptrs['layers'] = per_layer

    return handles


def extract_mtp_weights(mtp: dict, handles: WeightHandles) -> dict:
    """Add MTP head weights into ``handles``, return MTP-specific dict.

    The MTP head (Qwen3.6-27B-FP8 ckpt's ``mtp.safetensors``) is NOT
    loaded by HF's transformers ``Qwen3_5ForCausalLM`` because the
    modeling code has no MTP support. We load the safetensors file
    directly and pass the dict in here. Each tensor is anchored on
    ``handles`` so the int data_ptrs we stash stay live.

    Architecture (DeepSeek-V3 MTP, single layer):

        x = fc(cat[norm_h(prev_hidden), norm_e(embed(prev_token))])
        x = full_attn_layer(x, cos, sin, cur_pos)  # 1 layer, own KV cache
        x = final_norm(x)
        logits = lm_head(x)        # shared with main lm_head

    All projections are FP8 block-128 with bf16 ``weight_scale_inv``,
    matching the main model. ``fc`` is BF16 (no scale), shape
    (5120, 10240). RMSNorms follow Qwen3_5RMSNorm's (1+w) convention.

    Args:
        mtp: dict[str, torch.Tensor] -- the loaded mtp.safetensors
            with the leading ``mtp.`` prefix stripped (e.g. keys are
            ``layers.0.self_attn.q_proj.weight``, ``norm.weight``, ...).
            Tensors must already be on cuda. weight_scale_inv tensors
            must already be cast to fp32.
        handles: existing WeightHandles to extend.

    Returns:
        dict with the same key surface as a main full-attn layer dict
        plus three MTP-specific norms (pre_fc_norm_hidden_eff_w,
        pre_fc_norm_embedding_eff_w, final_norm_eff_w) and
        ``fc_w`` (BF16 GEMM weight ptr, no scale).
    """
    out: dict = {'type': 'mtp'}

    # input/post norms (Qwen3_5RMSNorm (1+w))
    in_eff = _eff_rmsnorm_weight(mtp['layers.0.input_layernorm.weight'])
    post_eff = _eff_rmsnorm_weight(
        mtp['layers.0.post_attention_layernorm.weight'])
    out['input_norm_eff_w'] = _ensure_anchored(handles, in_eff)
    out['post_attn_norm_eff_w'] = _ensure_anchored(handles, post_eff)

    # FP8 projections (q/k/v/o + mlp gate/up/down). Mirror main model.
    fp8_pairs = (
        ('q_proj',   'layers.0.self_attn.q_proj'),
        ('k_proj',   'layers.0.self_attn.k_proj'),
        ('v_proj',   'layers.0.self_attn.v_proj'),
        ('o_proj',   'layers.0.self_attn.o_proj'),
        ('mlp_gate', 'layers.0.mlp.gate_proj'),
        ('mlp_up',   'layers.0.mlp.up_proj'),
        ('mlp_down', 'layers.0.mlp.down_proj'),
    )
    for prefix, base in fp8_pairs:
        w = mtp[base + '.weight']
        s = mtp[base + '.weight_scale_inv']
        handles.anchors.append(w)
        handles.anchors.append(s)
        out[prefix + '_w'] = int(w.data_ptr())
        out[prefix + '_s'] = int(s.data_ptr())

    # head-dim RMSNorms ((1+w))
    q_eff = _eff_rmsnorm_weight(mtp['layers.0.self_attn.q_norm.weight'])
    k_eff = _eff_rmsnorm_weight(mtp['layers.0.self_attn.k_norm.weight'])
    out['q_norm_eff_w'] = _ensure_anchored(handles, q_eff)
    out['k_norm_eff_w'] = _ensure_anchored(handles, k_eff)

    # MTP-specific extras: 2 pre-fc norms + 1 final norm before lm_head
    pre_h_eff = _eff_rmsnorm_weight(mtp['pre_fc_norm_hidden.weight'])
    pre_e_eff = _eff_rmsnorm_weight(mtp['pre_fc_norm_embedding.weight'])
    final_eff = _eff_rmsnorm_weight(mtp['norm.weight'])
    out['pre_fc_norm_hidden_eff_w'] = _ensure_anchored(handles, pre_h_eff)
    out['pre_fc_norm_embedding_eff_w'] = _ensure_anchored(handles, pre_e_eff)
    out['final_norm_eff_w'] = _ensure_anchored(handles, final_eff)

    # fc: BF16 GEMM, K=10240 (cat of 2 x 5120) -> N=5120
    fc_w = mtp['fc.weight'].contiguous()
    handles.anchors.append(fc_w)
    out['fc_w'] = int(fc_w.data_ptr())

    return out


def assert_extraction_invariants(handles: WeightHandles) -> None:
    """Sanity-check that every layer dict has the right keys.

    Designed to be run once at frontend load (not in the forward hot
    path). Catches schema regressions early -- a missing key here is
    an immediate AssertionError, not a mysterious pointer crash later.
    """
    p = handles.ptrs
    assert isinstance(p.get('layers'), list)
    layers = p['layers']
    assert len(layers) == p['num_layers']
    types = p['layer_types']
    n_full = sum(1 for t in types if t == 'full_attention')
    n_lin = sum(1 for t in types if t == 'linear_attention')
    assert n_full == 16, f'expected 16 full-attn, got {n_full}'
    assert n_lin == 48, f'expected 48 linear-attn, got {n_lin}'

    common_keys = {
        'input_norm_eff_w', 'post_attn_norm_eff_w',
        'mlp_gate_w', 'mlp_gate_s',
        'mlp_up_w', 'mlp_up_s',
        'mlp_down_w', 'mlp_down_s',
    }
    lin_keys = common_keys | {
        'in_proj_qkv_w', 'in_proj_qkv_s',
        'in_proj_z_w', 'in_proj_z_s',
        'in_proj_a_w', 'in_proj_b_w',
        'out_proj_w', 'out_proj_s',
        'conv1d_w', 'conv1d_b',
        'head_norm_w', 'A_log', 'dt_bias',
    }
    full_keys = common_keys | {
        'q_proj_w', 'q_proj_s',
        'k_proj_w', 'k_proj_s',
        'v_proj_w', 'v_proj_s',
        'o_proj_w', 'o_proj_s',
        'q_norm_eff_w', 'k_norm_eff_w',
    }

    for L, ld in enumerate(layers):
        assert ld is not None, f'layer {L} not populated'
        t = ld['type']
        if t == 'linear_attention':
            missing = lin_keys - set(ld.keys())
            assert not missing, f'lin layer {L} missing {missing}'
        elif t == 'full_attention':
            missing = full_keys - set(ld.keys())
            assert not missing, f'full layer {L} missing {missing}'
        else:
            raise AssertionError(f'layer {L} has unknown type {t!r}')

    # Top-level keys.
    for k in ('embed_w', 'final_norm_eff_w', 'lm_head_w',
              'lm_head_tied', 'vocab_size', 'hidden'):
        assert k in p, f'top-level missing {k!r}'
