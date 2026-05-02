"""Offline NVFP4 weight preparation for BAGEL gen expert (world-model path).

Loads original fp16 weights directly from ema.safetensors and quantizes to
NVFP4 packed + CUTLASS tile-interleaved SFB via flash_rt.flash_rt_fp4.
This avoids the FP8→fp16 dequant path (double-lossy) that BagelFP8Engine uses.

Scope: gen expert only (q/k/v/o/gate/up/down × 28 layers).
       und expert stays FP8 per B0 decision (M=2 path, FP4 unlikely to win).

Usage:
    from bagel_fp4_weights import prepare_gen_fp4_weights
    fp4_w = prepare_gen_fp4_weights('<bagel_weights>')
    # fp4_w[layer_idx]['gate'] == {'packed': uint8[N,K/2], 'sfb': uint8[..], 'N', 'K'}
"""
from __future__ import annotations

import os, sys, torch
from safetensors import safe_open

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS)))   # repo root

from flash_rt.executors.fp4_utils import quant_weight_nvfp4

fp16 = torch.float16

N_LAYERS = 28
GEN_WEIGHT_KEYS = {
    'q':    'self_attn.q_proj_moe_gen.weight',
    'k':    'self_attn.k_proj_moe_gen.weight',
    'v':    'self_attn.v_proj_moe_gen.weight',
    'o':    'self_attn.o_proj_moe_gen.weight',
    'gate': 'mlp_moe_gen.gate_proj.weight',
    'up':   'mlp_moe_gen.up_proj.weight',
    'down': 'mlp_moe_gen.down_proj.weight',
}


def _safetensors_path(weights_root: str) -> str:
    p = os.path.join(weights_root, 'ema.safetensors')
    assert os.path.exists(p), f"missing {p}"
    return p


def load_gen_fp16_weight(sf, layer: int, gname: str) -> torch.Tensor:
    """Load a single gen-expert weight as fp16 CUDA contiguous."""
    key = f"language_model.model.layers.{layer}.{GEN_WEIGHT_KEYS[gname]}"
    w = sf.get_tensor(key)
    return w.to(fp16).cuda().contiguous()


def prepare_gen_fp4_weights(weights_root: str,
                             layers=None,
                             gnames=None,
                             verbose: bool = True) -> dict:
    """Build fp4 weight dict for all gen-expert projections.

    Args:
        weights_root: directory containing ema.safetensors.
        layers: iterable of layer indices. None → all 28.
        gnames: iterable of gemm names in {q,k,v,o,gate,up,down}. None → all 7.

    Returns:
        dict[layer_idx] = dict[gname] = {'packed', 'sfb', 'N', 'K'}
        (as returned by flash_rt.executors.fp4_utils.quant_weight_nvfp4).
    """
    layers = range(N_LAYERS) if layers is None else list(layers)
    gnames = list(GEN_WEIGHT_KEYS.keys()) if gnames is None else list(gnames)

    path = _safetensors_path(weights_root)
    out = {l: {} for l in layers}
    with safe_open(path, framework='pt', device='cpu') as sf:
        for l in layers:
            for g in gnames:
                w = load_gen_fp16_weight(sf, l, g)
                out[l][g] = quant_weight_nvfp4(w)
                # Free the fp16 source; only the FP4 buffers are kept.
                del w
            if verbose and (l + 1) % 7 == 0:
                print(f"  prepare_gen_fp4_weights: {l + 1}/{len(layers)}")
    torch.cuda.synchronize()
    return out


def prepare_gen_fp4_weights_awq(weights_root: str, awq_scales: dict,
                                  layers=None, verbose: bool = True) -> dict:
    """Like prepare_gen_fp4_weights but pre-multiplies the fp16 weight's K
    axis by the AWQ scale s[k] before NVFP4 quantization.

    Math invariant: W' @ (x * inv_s)^T = (W * s) @ (x / s)^T = W @ x^T.
    Runtime must apply inv_s to the activation before the FP4 GEMM.

    awq_scales: output of compute_awq_scales — {layer: {qkv,o,gu,dn: {s, inv_s}}}.
    Each gemm name (q/k/v/o/gate/up/down) is routed to its corresponding group
    amax bucket via GROUP_TO_WEIGHTS (imported lazily to avoid cycle).
    """
    from bagel_fp4_calibrate import GROUP_TO_WEIGHTS

    # Invert the mapping: weight gname → group name
    weight_to_group = {}
    for group, members in GROUP_TO_WEIGHTS.items():
        for w in members:
            weight_to_group[w] = group

    layers = range(N_LAYERS) if layers is None else list(layers)
    path = _safetensors_path(weights_root)
    out = {l: {} for l in layers}
    with safe_open(path, framework='pt', device='cpu') as sf:
        for l in layers:
            groups_l = awq_scales[l]
            for g in GEN_WEIGHT_KEYS:
                w_fp16 = load_gen_fp16_weight(sf, l, g)  # [N, K] cuda
                s = groups_l[weight_to_group[g]]['s']   # fp16 [K]
                assert s.numel() == w_fp16.shape[1], \
                    f"AWQ scale K={s.numel()} != weight K={w_fp16.shape[1]} (L{l} {g})"
                w_scaled = (w_fp16.float() * s.float().unsqueeze(0)).to(fp16).contiguous()
                out[l][g] = quant_weight_nvfp4(w_scaled)
                del w_fp16, w_scaled
            if verbose and (l + 1) % 7 == 0:
                print(f"  prepare_gen_fp4_weights_awq: {l + 1}/{len(layers)}")
    torch.cuda.synchronize()
    return out


def prepare_gen_fp4_ffn(weights_root: str, awq_scales: dict,
                         ln2_w_per_layer: list,
                         fp4_layers,
                         verbose: bool = True) -> dict:
    """Offline prep for B5 single-layer FP4 FFN swap.

    For each layer in ``fp4_layers``:
      * ``gate_fp4`` = quant_nvfp4(W_gate * s_gu_K)   [FFN, D]
      * ``up_fp4``   = quant_nvfp4(W_up   * s_gu_K)   [FFN, D]
      * ``down_fp4`` = quant_nvfp4(W_down)            [D, FFN]  (no AWQ on dn;
            B4 showed Δcos < 0.001 vs AWQ — not worth the plumbing)
      * ``ln_baked`` = (ln2_w * inv_s_gu)             bf16 [D]
            Fed to fvk.rms_norm as the norm weight; its output then equals
            `rms(x) * ln2_w * inv_s_gu`, and the baked W_gate*s_gu cancels
            inv_s_gu by the AWQ identity. No per_channel_mul at runtime.

    Args:
        weights_root: dir containing ema.safetensors.
        awq_scales: output of bagel_fp4_calibrate.compute_awq_scales.
        ln2_w_per_layer: list of length 28 of bf16 [D] tensors
            (engine.gen_norms[l]['pn']).
        fp4_layers: iterable of layer indices to quantize.

    Returns:
        {layer_idx: {'gate': {packed,sfb,N,K}, 'up': ..., 'down': ...,
                     'gateup': {packed,sfb,N=2*FFN,K=D},  # merged Wab (1b)
                     'ln_baked': bf16 tensor [D]}}
    """
    bf16 = torch.bfloat16
    fp4_layers = list(fp4_layers)
    path = _safetensors_path(weights_root)
    out = {l: {} for l in fp4_layers}
    with safe_open(path, framework='pt', device='cpu') as sf:
        for l in fp4_layers:
            s_gu   = awq_scales[l]['gu']['s'].float()       # fp16 [D]
            inv_gu = awq_scales[l]['gu']['inv_s'].float()   # fp16 [D]

            # Down AWQ scales — bake into up (N axis) and down (K axis) so
            # the hot loop needs no runtime per_channel_mul. Identity:
            #   (silu(g)*up*inv_s_dn) @ (W_down*s_dn)^T = (silu(g)*up) @ W_down^T
            s_dn_   = awq_scales[l]['dn']['s'].float()        # fp16 [FFN]
            inv_dn_ = awq_scales[l]['dn']['inv_s'].float()    # fp16 [FFN]

            # Gate: bake s_gu into K axis only
            w_gate = load_gen_fp16_weight(sf, l, 'gate')      # [FFN, D]
            w_gate_scaled = (w_gate.float()
                             * s_gu.cuda().unsqueeze(0)).to(fp16).contiguous()
            out[l]['gate'] = quant_weight_nvfp4(w_gate_scaled)

            # Up: bake s_gu (K axis) AND inv_s_dn (N axis)
            w_up = load_gen_fp16_weight(sf, l, 'up')          # [FFN, D]
            w_up_scaled = (w_up.float()
                           * s_gu.cuda().unsqueeze(0)                # [FFN, D]
                           * inv_dn_.cuda().unsqueeze(1)              # [FFN, 1]
                          ).to(fp16).contiguous()
            out[l]['up']   = quant_weight_nvfp4(w_up_scaled)

            # Class 1b/1c Wab: merged gate+up weight [2*FFN, D] fp16 → one
            # NVFP4 packed/sfb pair. Runtime: one GEMM with N=2*FFN writes
            # [Sq, 2*FFN] fp16 with gate cols [0..FFN), up cols [FFN..2FFN),
            # exactly the layout gate_silu_mul_fp4_sfa_v2_fp16 consumes.
            w_gateup_scaled = torch.cat(
                [w_gate_scaled, w_up_scaled], dim=0).contiguous()  # [2*FFN, D]
            out[l]['gateup'] = quant_weight_nvfp4(w_gateup_scaled)
            del w_gate, w_gate_scaled, w_up, w_up_scaled, w_gateup_scaled

            # Down: bake s_dn into K axis
            w_down = load_gen_fp16_weight(sf, l, 'down')      # [D, FFN]
            w_down_scaled = (w_down.float()
                             * s_dn_.cuda().unsqueeze(0)).to(fp16).contiguous()
            out[l]['down'] = quant_weight_nvfp4(w_down_scaled)
            del w_down, w_down_scaled

            # Bake ln2_w * inv_s_gu for fvk.rms_norm's weight
            ln2_w = ln2_w_per_layer[l].to(bf16).cuda()      # [D]
            ln_baked = (ln2_w.float() * inv_gu.cuda()).to(bf16).contiguous()
            out[l]['ln_baked'] = ln_baked

            if verbose:
                print(f"  prepare_gen_fp4_ffn: L{l} baked ln2_w*inv_s_gu, "
                      f"quantized gate/up/down")
    torch.cuda.synchronize()
    return out


def gen_weight_shape(gname: str) -> tuple[int, int]:
    """Canonical (N, K) for each gen projection at D=3584, FFN=18944, K_DIM=512."""
    D = 3584; FFN = 18944; K_DIM = 512
    return {
        'q':    (D,     D),
        'k':    (K_DIM, D),
        'v':    (K_DIM, D),
        'o':    (D,     D),
        'gate': (FFN,   D),
        'up':   (FFN,   D),
        'down': (D,     FFN),
    }[gname]
