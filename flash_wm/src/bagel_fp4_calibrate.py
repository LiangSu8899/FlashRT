"""B3 activation amax collection for AWQ FP4 calibration (gen path only).

Walks BAGEL's real MoT trajectory (text=BF16 und, VAE=FP8 gen) once and
snapshots per-input-channel fp16 activation amax at the 4 GEMM entry points
of the gen expert, for VAE rows only:

  L × gemm input         | shape                  | scope
  ----------------------------------------------------------
  layer i | qkv input    | rms_norm(buf_x, gnm.in)[vi]  | [M, D]
  layer i | o   input    | ao[vi]                        | [M, D]
  layer i | gate+up input| rms_norm(buf_x, gnm.pn)[vi]  | [M, D]
  layer i | down input   | silu(gate_vae)*up_vae         | [M, FFN]

Text rows are excluded — und path stays FP8, its amax shouldn't pollute
gen-path scale estimation.

Invariants preserved:
  * Uses engine.buf_x, engine._k_merged/_v_merged, engine._rope_cos/_sin
    exactly as forward_step does. Trajectory is bit-compatible.
  * Calls engine._fp8_gemm for VAE (same FP8 weights/scales) — so the
    downstream state we feed into layer i+1 is the production state.
  * Makes no permanent modifications to engine; restores buf_x on exit.

Usage:
    from bagel_fp4_calibrate import collect_gen_activation_amax
    amax = collect_gen_activation_amax(engine)
    # amax[l] = {'qkv': fp32[D], 'o': fp32[D], 'gu': fp32[D], 'dn': fp32[FFN]}
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk

from bagel_fp8_engine import (
    rms_norm_bf16, apply_rotary_pos_emb,
    N_LAYERS, D, H, KV_H, HD, K_DIM, FFN, FP8, bf16, fp16,
)

GEN_GEMMS = ('qkv', 'o', 'gu', 'dn')
GEMM_DIM = {'qkv': D, 'o': D, 'gu': D, 'dn': FFN}

# Mapping from "which amax bucket" to "which production weights share it".
# Q, K, V share the qkv input; Gate, Up share the gu input; O and Down each
# own their input amax.
GROUP_TO_WEIGHTS = {
    'qkv': ('q', 'k', 'v'),
    'o':   ('o',),
    'gu':  ('gate', 'up'),
    'dn':  ('down',),
}


@torch.no_grad()
def recalibrate_act_scales_mixed_multi_t(engine, fp4_layers, fp4_ffn, x_inputs):
    """Multi-sample variant of recalibrate_act_scales_mixed: walks the
    forward N times (one per x_input tensor) and keeps per-(layer, gemm)
    MAX of all measured amaxes before committing to engine.act_scales.

    Needed for low-t failures (L9 at t=0.4): post-rms-norm magnitudes vary
    modestly with t but the POST-residual / POST-attn magnitudes vary a lot,
    and single-t scales undersize some channels at low t → FP8 overflow.
    """
    # Collect amax tables from N walks without committing scales, then
    # max-reduce. Simplest implementation: snapshot engine.act_scales /
    # und_act_scales / alphas / und_alphas after each walk into a stash,
    # then elementwise max.
    from bagel_fp8_engine import N_LAYERS, GEMM_NAMES
    snapshots = []
    for x_input in x_inputs:
        recalibrate_act_scales_mixed(engine, fp4_layers, fp4_ffn, x_input)
        snap_gen = [[engine.act_scales[i][j].clone()
                     for j in range(len(engine.act_scales[i]))]
                    for i in range(N_LAYERS)]
        snap_und = [[engine.und_act_scales[i][j].clone()
                     for j in range(len(engine.und_act_scales[i]))]
                    for i in range(N_LAYERS)]
        snapshots.append((snap_gen, snap_und))
    # Elementwise max across snapshots
    for i in range(N_LAYERS):
        for j in range(len(engine.act_scales[i])):
            mx = max(float(snap[0][i][j].item()) for snap in snapshots)
            engine.act_scales[i][j].fill_(mx)
        for j in range(len(engine.und_act_scales[i])):
            mx = max(float(snap[1][i][j].item()) for snap in snapshots)
            engine.und_act_scales[i][j].fill_(mx)

    for i in range(N_LAYERS):
        ws = engine.gen_w_scales[i]
        uws = engine.und_w_scales[i]
        for j in range(7):
            engine.alphas[i][j] = float(engine.act_scales[i][j].item()) * ws[GEMM_NAMES[j]]
        engine.und_alphas[i]['q']    = float(engine.und_act_scales[i][0].item()) * uws['q']
        engine.und_alphas[i]['k']    = float(engine.und_act_scales[i][0].item()) * uws['k']
        engine.und_alphas[i]['v']    = float(engine.und_act_scales[i][0].item()) * uws['v']
        engine.und_alphas[i]['o']    = float(engine.und_act_scales[i][1].item()) * uws['o']
        engine.und_alphas[i]['gate'] = float(engine.und_act_scales[i][2].item()) * uws['gate']
        engine.und_alphas[i]['up']   = float(engine.und_act_scales[i][2].item()) * uws['up']
        engine.und_alphas[i]['down'] = float(engine.und_act_scales[i][3].item()) * uws['down']


@torch.no_grad()
def recalibrate_act_scales_mixed(engine, fp4_layers, fp4_ffn, x_input):
    """Walk a full MoT forward with FP4 engaged on fp4_layers, snapshot
    amax at every gen GEMM input AND every und GEMM input, update
    ``engine.act_scales`` / ``engine.alphas`` / ``engine.und_act_scales`` /
    ``engine.und_alphas`` in place.

    Rationale (from L9 NaN debug): engine.calibrate() produces scales on
    a pure-FP8 trajectory. When FP4 replaces an FFN, the residual stream
    shifts slightly; the FP8 scales in all subsequent layers get out of
    calibration and eventually saturate (at low t we saw b_x amax = 160K
    and FP8 quantize overflow). Recalibrating on the mixed FP4+FP8 path
    is the pi05-style fix.

    Args:
        engine: BagelFP8Engine (already pre-calibrated once).
        fp4_layers: iterable of layer indices running FP4 FFN.
        fp4_ffn: output of prepare_gen_fp4_ffn — packed FP4 weights +
            ln_baked for those layers.
        x_input: [Sq, D] bf16 — the same diffusion-step input used for
            initial calibration.

    Mutates engine state in place. Caller must re-run
    ``ck.set_act_scales_from_engine(engine)`` afterwards.
    """
    import flash_rt.flash_rt_kernels as fvk
    import flash_rt.flash_rt_fp4 as fvk_fp4
    import flash_wm_kernels as fwk
    from flash_rt.executors.fp4_utils import FP4ActScratch
    from bagel_fp8_engine import (
        rms_norm_bf16, apply_rotary_pos_emb,
        N_LAYERS, D, H, KV_H, HD, K_DIM, FFN, FP8, GEMM_NAMES,
    )

    fp4_set = set(int(l) for l in fp4_layers)

    Sq = engine.Sq; M = engine.M
    kv_len = engine.kv_len
    total_kv = kv_len + Sq
    ti = engine._text_idx
    vi = engine._vae_idx
    dev = engine.device

    # Reset buf_x to the input sample the caller used for calibrate()
    engine.buf_x.copy_(x_input)

    # Transient FP4 scratches — reused across layers
    sc_gu = FP4ActScratch(Sq, D,   device=dev)
    sc_dn = FP4ActScratch(Sq, FFN, device=dev)

    # One-liner AWQ apply + FP4 quant wrapping the FP4 FFN pipeline.
    # pre_residual: buf_x BEFORE post-attn residual was added.
    # o_all:        attn output (gen FP8 on all rows, und BF16 overlay on text).
    def _fp4_ffn(i, pre_residual, o_all):
        fp4 = fp4_ffn[i]
        # Compute post-attn residual temporarily for the baked rms_norm.
        tmp_x = pre_residual + o_all
        # bf16 rms_norm with baked ln*inv_s_gu → bf16 normed
        normed = torch.empty(Sq, D, dtype=torch.bfloat16, device=dev)
        fvk.rms_norm(tmp_x.contiguous().data_ptr(), fp4['ln_baked'].data_ptr(),
                      normed.data_ptr(), Sq, D, 1e-6, 0)
        normed_fp16 = torch.empty(Sq, D, dtype=torch.float16, device=dev)
        fwk.cast_bf16_to_fp16(normed.data_ptr(), normed_fp16.data_ptr(),
                               Sq * D, 0)
        fvk_fp4.quantize_fp4_dynamic_sfa_fp16(
            normed_fp16.data_ptr(),
            sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
            Sq, D, False, 0)
        gate_fp16 = torch.empty(Sq, FFN, dtype=torch.float16, device=dev)
        up_fp16   = torch.empty(Sq, FFN, dtype=torch.float16, device=dev)
        fvk_fp4.cutlass_fp4_gemm_variant(
            6,  # V_GATE
            sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
            fp4['gate']['packed'].data_ptr(), fp4['gate']['sfb'].data_ptr(),
            gate_fp16.data_ptr(), Sq, FFN, D, 1.0, 0.0, 0)
        fvk_fp4.cutlass_fp4_gemm_variant(
            6,  # V_UP
            sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
            fp4['up']['packed'].data_ptr(), fp4['up']['sfb'].data_ptr(),
            up_fp16.data_ptr(), Sq, FFN, D, 1.0, 0.0, 0)
        hid_fp16 = torch.empty(Sq, FFN, dtype=torch.float16, device=dev)
        fwk.silu_mul_fp16(gate_fp16.data_ptr(), up_fp16.data_ptr(),
                           hid_fp16.data_ptr(), Sq * FFN, 0)
        fvk_fp4.quantize_fp4_dynamic_sfa_fp16(
            hid_fp16.data_ptr(),
            sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
            Sq, FFN, False, 0)
        down_fp16 = torch.empty(Sq, D, dtype=torch.float16, device=dev)
        fvk_fp4.cutlass_fp4_gemm_variant(
            8,  # V_DOWN
            sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
            fp4['down']['packed'].data_ptr(), fp4['down']['sfb'].data_ptr(),
            down_fp16.data_ptr(), Sq, D, FFN, 1.0, 0.0, 0)
        down_bf16 = torch.empty(Sq, D, dtype=torch.bfloat16, device=dev)
        fwk.cast_fp16_to_bf16(down_fp16.data_ptr(), down_bf16.data_ptr(),
                               Sq * D, 0)
        return down_bf16  # gen FFN output; text rows overwritten by caller

    for i in range(N_LAYERS):
        gnm = engine.gen_norms[i]; gbi = engine.gen_biases[i]; ws = engine.gen_w_scales[i]
        unm = engine.und_norms[i]; ubi = engine.und_biases[i]
        uw = engine.und_w[i]; uws = engine.und_w_scales[i]

        engine.buf_res.copy_(engine.buf_x)

        # ── 1. QKV input amax (gen) ─────────────────────────
        xn_gen = rms_norm_bf16(engine.buf_x, gnm['in'])       # [Sq, D]
        s_qkv = max(xn_gen.abs().float().max().item() / 448.0, 1e-12)
        for j in range(3):
            engine.act_scales[i][j].fill_(s_qkv)
            engine.alphas[i][j] = s_qkv * ws[GEMM_NAMES[j]]
        # und QKV
        xn_text = rms_norm_bf16(engine.buf_x[ti], unm['in'])
        s_und_qkv = max(xn_text.abs().float().max().item() / 448.0, 1e-12)
        engine.und_act_scales[i][0].fill_(s_und_qkv)
        for gname in ('q', 'k', 'v'):
            engine.und_alphas[i][gname] = s_und_qkv * uws[gname]

        # ── FP8 QKV for VAE rows, und for text (same as forward_step) ─
        fvk.rms_norm_fp8(
            engine.buf_x.data_ptr(), gnm['in'].data_ptr(), engine.act_fp8.data_ptr(),
            Sq, D, 1e-6, engine.act_scales[i][0].data_ptr(), 0)
        engine._fp8_gemm(i, 0, engine.buf_q, Sq, D, D)
        engine._fp8_gemm(i, 1, engine.buf_k, Sq, K_DIM, D)
        engine._fp8_gemm(i, 2, engine.buf_v, Sq, K_DIM, D)
        engine.buf_q[ti] = F.linear(xn_text, uw['q'], ubi['q'])
        engine.buf_k[ti] = F.linear(xn_text, uw['k'], ubi['k'])
        engine.buf_v[ti] = F.linear(xn_text, uw['v'], ubi['v'])

        q = engine.buf_q.clone(); k = engine.buf_k.clone(); v = engine.buf_v.clone()
        q[vi] = q[vi] + gbi['q']; k[vi] = k[vi] + gbi['k']; v[vi] = v[vi] + gbi['v']
        q = q.view(Sq, H, HD); k = k.view(Sq, KV_H, HD); v = v.view(Sq, KV_H, HD)
        q_n = torch.empty_like(q); k_n = torch.empty_like(k)
        q_n[ti] = rms_norm_bf16(q[ti], unm['qn']); q_n[vi] = rms_norm_bf16(q[vi], gnm['qn'])
        k_n[ti] = rms_norm_bf16(k[ti], unm['kn']); k_n[vi] = rms_norm_bf16(k[vi], gnm['kn'])
        q_n, k_n = apply_rotary_pos_emb(q_n, k_n, engine._rope_cos, engine._rope_sin)
        q_n = q_n.to(torch.bfloat16); k_n = k_n.to(torch.bfloat16); v = v.to(torch.bfloat16)
        engine._k_merged[i][kv_len:kv_len + Sq].copy_(k_n)
        engine._v_merged[i][kv_len:kv_len + Sq].copy_(v)

        q_4d = q_n.unsqueeze(0).transpose(1, 2)
        k_4d = engine._k_merged[i][:total_kv].unsqueeze(0).transpose(1, 2)
        v_4d = engine._v_merged[i][:total_kv].unsqueeze(0).transpose(1, 2)
        ao = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=False,
                                             enable_gqa=True)
        ao = ao.transpose(1, 2).contiguous().view(Sq, D)

        # ── 2. O input amax ─────────────────────────────────
        s_o = max(ao.abs().float().max().item() / 448.0, 1e-12)
        engine.act_scales[i][3].fill_(s_o)
        engine.alphas[i][3] = s_o * ws['o']
        s_und_o = max(ao[ti].abs().float().max().item() / 448.0, 1e-12)
        engine.und_act_scales[i][1].fill_(s_und_o)
        engine.und_alphas[i]['o'] = s_und_o * uws['o']

        # Gen O (FP8) + und O (BF16)
        fvk.quantize_fp8_static(
            ao.data_ptr(), engine.act_fp8.data_ptr(),
            engine.act_scales[i][3].data_ptr(), Sq * D, 0)
        o_all = torch.empty(Sq, D, dtype=torch.bfloat16, device=dev)
        fvk.cutlass_fp8_sq_bf16out(
            engine.act_fp8.data_ptr(), engine.gen_fp8_w[i]['o'].data_ptr(),
            o_all.data_ptr(), Sq, D, D, engine.alphas[i][3], 0.0, 0)
        o_all[ti] = F.linear(ao[ti], uw['o'])  # und O bf16
        pre_residual = engine.buf_res.clone()
        engine.buf_x.copy_(pre_residual + o_all)

        # ── FFN branch ──
        engine.buf_res.copy_(engine.buf_x)
        # Text FFN (BF16 und) — always the same
        xn_text_pn = rms_norm_bf16(engine.buf_x[ti], unm['pn'])
        s_und_gu = max(xn_text_pn.abs().float().max().item() / 448.0, 1e-12)
        engine.und_act_scales[i][2].fill_(s_und_gu)
        engine.und_alphas[i]['gate'] = s_und_gu * uws['gate']
        engine.und_alphas[i]['up']   = s_und_gu * uws['up']
        g_t = F.linear(xn_text_pn, uw['gate']); u_t = F.linear(xn_text_pn, uw['up'])
        silu_t = F.silu(g_t) * u_t
        s_und_dn = max(silu_t.abs().float().max().item() / 448.0, 1e-12)
        engine.und_act_scales[i][3].fill_(s_und_dn)
        engine.und_alphas[i]['down'] = s_und_dn * uws['down']
        ffn_text = F.linear(silu_t, uw['down'])

        if i in fp4_set:
            # Engage FP4 FFN on the VAE-shared-with-text residual stream.
            # engine.buf_x currently holds (pre_residual + o_all).
            ffn_all_bf16 = _fp4_ffn(i, pre_residual, o_all)
            # act_scales[l][4..6] unused by FP4; leave as-is.
            ffn_out = torch.empty(Sq, D, dtype=torch.bfloat16, device=dev)
            ffn_out[ti] = ffn_text
            ffn_out[vi] = ffn_all_bf16[vi]
            engine.buf_x.copy_(engine.buf_res + ffn_out)
        else:
            # FP8 gen FFN. Measure scales like engine.calibrate does.
            xn_gen_pn = rms_norm_bf16(engine.buf_x, gnm['pn'])
            s_gu = max(xn_gen_pn.abs().float().max().item() / 448.0, 1e-12)
            for j in (4, 5):
                engine.act_scales[i][j].fill_(s_gu)
                engine.alphas[i][j] = s_gu * ws[GEMM_NAMES[j]]
            fvk.rms_norm_fp8(
                engine.buf_x.data_ptr(), gnm['pn'].data_ptr(),
                engine.act_fp8.data_ptr(), Sq, D, 1e-6,
                engine.act_scales[i][4].data_ptr(), 0)
            _cal_gate = torch.empty(Sq, FFN, dtype=torch.bfloat16, device=dev)
            _cal_up   = torch.empty(Sq, FFN, dtype=torch.bfloat16, device=dev)
            engine._fp8_gemm(i, 4, _cal_gate, Sq, FFN, D)
            engine._fp8_gemm(i, 5, _cal_up,   Sq, FFN, D)
            silu_mul = F.silu(_cal_gate) * _cal_up
            s_dn = max(silu_mul.abs().float().max().item() / 448.0, 1e-12)
            engine.act_scales[i][6].fill_(s_dn)
            engine.alphas[i][6] = s_dn * ws['down']
            fvk.quantize_fp8_static(
                silu_mul.data_ptr(), engine.act_fp8.data_ptr(),
                engine.act_scales[i][6].data_ptr(), Sq * FFN, 0)
            _cal_down = torch.empty(Sq, D, dtype=torch.bfloat16, device=dev)
            engine._fp8_gemm(i, 6, _cal_down, Sq, D, FFN)
            ffn_out = torch.empty(Sq, D, dtype=torch.bfloat16, device=dev)
            ffn_out[ti] = ffn_text
            ffn_out[vi] = _cal_down[vi]
            engine.buf_x.copy_(engine.buf_res + ffn_out)

    torch.cuda.synchronize()


def compute_awq_scales(amax: dict, alpha: float = 0.5,
                        s_min: float = 0.25, s_max: float = 4.0,
                        dn_alpha: float = 1.0,
                        dn_s_min: float = 1.0, dn_s_max: float = 64.0,
                        per_layer_alpha: dict = None,
                        disable_awq_layers=None) -> dict:
    """AWQ per-input-channel scale: s[k] = (a[k] / a.mean())^alpha clamped.

    Down direction uses wider clamp + higher alpha by default because:
      * silu(gate)*up magnitudes can spike at low-t (see L9 @ t=0.4
        diagnosis: gate amax 283, silu*up amax 40K in fp16).
      * Down activation needs more aggressive re-scaling to keep the
        subsequent FP4 down-GEMM fp16 output below 65504.

    Args:
        amax: output of collect_gen_activation_amax — {layer: {qkv,o,gu,dn: fp32[K]}}.
        alpha: pow exponent for qkv/o/gu (AWQ paper default 0.5).
        s_min, s_max: clamp range for qkv/o/gu.
        dn_alpha / dn_s_min / dn_s_max: overrides for the dn group.

    Returns:
        {layer: {qkv,o,gu,dn: {'s': fp16[K], 'inv_s': fp16[K]}}}
    """
    per_layer_alpha = per_layer_alpha or {}
    disable_set = set(int(l) for l in (disable_awq_layers or ()))
    out = {}
    for l, per_g in amax.items():
        alpha_l = per_layer_alpha.get(l, alpha)
        entry = {}
        for g, a in per_g.items():
            a_f = a.float().clamp(min=1e-12)
            K = a.numel()
            if l in disable_set:
                # Force s = inv_s = 1 → no per-channel scaling. Used for
                # layers where AWQ misfires due to extreme sparsity
                # (e.g. L0: dead-channel clamp inversion shrinks active
                # channels, hurting precision; see tests/diagnose_l0.py).
                s = torch.ones(K, dtype=torch.float32, device=a.device)
            else:
                mean = a_f.mean()
                if g == 'dn':
                    s = (a_f / mean).pow(dn_alpha).clamp(min=dn_s_min, max=dn_s_max)
                else:
                    s = (a_f / mean).pow(alpha_l).clamp(min=s_min, max=s_max)
            inv_s = (1.0 / s).to(fp16).contiguous()
            s_fp16 = s.to(fp16).contiguous()
            entry[g] = {'s': s_fp16, 'inv_s': inv_s}
        out[l] = entry
    return out


@torch.no_grad()
def collect_gen_activation_amax(engine, snapshot_layers=None) -> dict:
    """Run one real MoT forward and collect per-input-channel fp16 amax on
    VAE rows at each gen GEMM input. Returns dict[layer][{qkv,o,gu,dn}] = fp32[K].

    If ``snapshot_layers`` is provided (iterable of layer indices), also
    dumps the full fp16 activation tile at each snapshot point for those
    layers under the 'snapshots' dict:
      snapshots[l][{qkv,o,gu,dn}] = fp16 [M, K] tensor (cpu, cloned)

    Engine state requirements:
      * engine.buf_x already prepared (via engine._prepare_input).
      * engine.calibrated == True (FP8 act scales for VAE path).
      * engine.kv_len / _k_merged / _v_merged / _rope_cos/_sin set up.
    """
    assert engine.calibrated, "engine.calibrate() must run before amax collection"

    Sq = engine.Sq; M = engine.M
    kv_len = engine.kv_len
    total_kv = kv_len + Sq
    ti = engine._text_idx
    vi = engine._vae_idx
    dev = engine.device

    # Allocate per-channel amax tensors up front on CUDA
    amax = {
        l: {
            'qkv': torch.zeros(D,   dtype=torch.float32, device=dev),
            'o':   torch.zeros(D,   dtype=torch.float32, device=dev),
            'gu':  torch.zeros(D,   dtype=torch.float32, device=dev),
            'dn':  torch.zeros(FFN, dtype=torch.float32, device=dev),
        } for l in range(N_LAYERS)
    }
    snap_set = set(snapshot_layers or ())
    snapshots = {l: {} for l in snap_set}

    # Preserve buf_x so repeat runs can be driven by a single _prepare_input
    # in the caller. We restore at exit.
    saved_buf_x = engine.buf_x.clone()
    try:
        _walk(engine, amax, ti, vi, Sq, M, kv_len, total_kv, dev,
              snap_set=snap_set, snapshots=snapshots)
    finally:
        engine.buf_x.copy_(saved_buf_x)

    torch.cuda.synchronize()
    if snap_set:
        return amax, snapshots
    return amax


def _snap(fp16_tile: torch.Tensor, amax_buf: torch.Tensor):
    """fp16_tile: [M, K]. Updates amax_buf[K] with per-column amax."""
    a = fp16_tile.abs().float().amax(dim=0)  # [K]
    torch.maximum(amax_buf, a, out=amax_buf)


def _walk(engine, amax, ti, vi, Sq, M, kv_len, total_kv, dev,
          snap_set=None, snapshots=None):
    """Faithful copy of BagelFP8Engine.forward_step with snapshots inserted
    at the 4 gen-GEMM inputs. Text path runs in BF16 und exactly as prod;
    VAE path runs in FP8 gen exactly as prod."""
    for i in range(N_LAYERS):
        gnm = engine.gen_norms[i]; gbi = engine.gen_biases[i]
        unm = engine.und_norms[i]; ubi = engine.und_biases[i]
        uw = engine.und_w[i]

        engine.buf_res.copy_(engine.buf_x)

        # ── 1. Gen input-norm (for QKV). Snap VAE rows.
        xn_gen_all = rms_norm_bf16(engine.buf_x, gnm['in'])  # [Sq, D]
        _qkv_tile = xn_gen_all[vi].to(fp16).contiguous()
        _snap(_qkv_tile, amax[i]['qkv'])
        if snap_set and i in snap_set:
            snapshots[i]['qkv'] = _qkv_tile.clone()

        # Drive production FP8 QKV path (reuse rms_norm_fp8 to keep act_scales
        # consumption identical to forward_step).
        fvk.rms_norm_fp8(
            engine.buf_x.data_ptr(), gnm['in'].data_ptr(), engine.act_fp8.data_ptr(),
            Sq, D, 1e-6, engine.act_scales[i][0].data_ptr(), 0)
        engine._fp8_gemm(i, 0, engine.buf_q, Sq, D,     D)
        engine._fp8_gemm(i, 1, engine.buf_k, Sq, K_DIM, D)
        engine._fp8_gemm(i, 2, engine.buf_v, Sq, K_DIM, D)

        # Text BF16 und override
        x_text = rms_norm_bf16(engine.buf_res[ti], unm['in'])
        engine.buf_q[ti] = F.linear(x_text, uw['q'], ubi['q'])
        engine.buf_k[ti] = F.linear(x_text, uw['k'], ubi['k'])
        engine.buf_v[ti] = F.linear(x_text, uw['v'], ubi['v'])

        q = engine.buf_q.clone(); k = engine.buf_k.clone(); v = engine.buf_v.clone()
        q[vi] = q[vi] + gbi['q']
        k[vi] = k[vi] + gbi['k']
        v[vi] = v[vi] + gbi['v']
        q = q.view(Sq, H, HD); k = k.view(Sq, KV_H, HD); v = v.view(Sq, KV_H, HD)

        q_n = torch.empty_like(q); k_n = torch.empty_like(k)
        q_n[ti] = rms_norm_bf16(q[ti], unm['qn']); q_n[vi] = rms_norm_bf16(q[vi], gnm['qn'])
        k_n[ti] = rms_norm_bf16(k[ti], unm['kn']); k_n[vi] = rms_norm_bf16(k[vi], gnm['kn'])
        q_n, k_n = apply_rotary_pos_emb(q_n, k_n, engine._rope_cos, engine._rope_sin)
        q_n = q_n.to(bf16); k_n = k_n.to(bf16); v = v.to(bf16)

        engine._k_merged[i][kv_len:kv_len + Sq].copy_(k_n)
        engine._v_merged[i][kv_len:kv_len + Sq].copy_(v)

        q_4d = q_n.unsqueeze(0).transpose(1, 2)
        k_4d = engine._k_merged[i][:total_kv].unsqueeze(0).transpose(1, 2)
        v_4d = engine._v_merged[i][:total_kv].unsqueeze(0).transpose(1, 2)
        ao = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=False,
                                             enable_gqa=True)
        ao = ao.transpose(1, 2).contiguous().view(Sq, D)

        # ── 2. Snap O input on VAE rows.
        _o_tile = ao[vi].to(fp16).contiguous()
        _snap(_o_tile, amax[i]['o'])
        if snap_set and i in snap_set:
            snapshots[i]['o'] = _o_tile.clone()

        # Production O projection
        o_out = torch.empty_like(ao)
        o_out[ti] = F.linear(ao[ti], uw['o'])
        ao_vae = ao[vi].contiguous()
        o_vae = torch.empty(M, D, dtype=bf16, device=dev)
        fvk.quantize_fp8_static(
            ao_vae.data_ptr(), engine.act_fp8.data_ptr(),
            engine.act_scales[i][3].data_ptr(), M * D, 0)
        fvk.cutlass_fp8_sq_bf16out(
            engine.act_fp8.data_ptr(), engine.gen_fp8_w[i]['o'].data_ptr(),
            o_vae.data_ptr(), M, D, D, engine.alphas[i][3], 0.0, 0)
        o_out[vi] = o_vae
        engine.buf_x.copy_(engine.buf_res + o_out)

        # ── FFN ──
        engine.buf_res.copy_(engine.buf_x)

        # Text und FFN (BF16)
        xn_text = rms_norm_bf16(engine.buf_x[ti], unm['pn'])
        g_t = F.linear(xn_text, uw['gate']); u_t = F.linear(xn_text, uw['up'])
        ffn_text = F.linear(F.silu(g_t) * u_t, uw['down'])

        # ── 3. Gen post-attn norm for Gate/Up. Snap VAE rows.
        xn_gen_pn = rms_norm_bf16(engine.buf_x[vi], gnm['pn'])  # [M, D]
        _gu_tile = xn_gen_pn.to(fp16).contiguous()
        _snap(_gu_tile, amax[i]['gu'])
        if snap_set and i in snap_set:
            snapshots[i]['gu'] = _gu_tile.clone()

        # Production VAE gen FFN (FP8)
        x_vae = engine.buf_x[vi].contiguous()
        act_fp8_vae = torch.empty(M * FFN, dtype=FP8, device=dev)
        fvk.rms_norm_fp8(
            x_vae.data_ptr(), gnm['pn'].data_ptr(), act_fp8_vae.data_ptr(),
            M, D, 1e-6, engine.act_scales[i][4].data_ptr(), 0)
        gate_vae = torch.empty(M, FFN, dtype=bf16, device=dev)
        up_vae   = torch.empty(M, FFN, dtype=bf16, device=dev)
        fvk.cutlass_fp8_t1_bf16out(
            act_fp8_vae.data_ptr(), engine.gen_fp8_w[i]['gate'].data_ptr(),
            gate_vae.data_ptr(), M, FFN, D, engine.alphas[i][4], 0.0, 0)
        fvk.cutlass_fp8_t1_bf16out(
            act_fp8_vae.data_ptr(), engine.gen_fp8_w[i]['up'].data_ptr(),
            up_vae.data_ptr(), M, FFN, D, engine.alphas[i][5], 0.0, 0)
        silu_vae = F.silu(gate_vae) * up_vae

        # ── 4. Snap Down input on VAE rows.
        _dn_tile = silu_vae.to(fp16).contiguous()
        _snap(_dn_tile, amax[i]['dn'])
        if snap_set and i in snap_set:
            snapshots[i]['dn'] = _dn_tile.clone()

        fvk.quantize_fp8_static(
            silu_vae.data_ptr(), act_fp8_vae.data_ptr(),
            engine.act_scales[i][6].data_ptr(), M * FFN, 0)
        down_vae = torch.empty(M, D, dtype=bf16, device=dev)
        fvk.cutlass_fp8_wide_bf16out(
            act_fp8_vae.data_ptr(), engine.gen_fp8_w[i]['down'].data_ptr(),
            down_vae.data_ptr(), M, D, FFN, engine.alphas[i][6], 0.0, 0)

        ffn_out = torch.empty(Sq, D, dtype=bf16, device=dev)
        ffn_out[ti] = ffn_text
        ffn_out[vi] = down_vae
        engine.buf_x.copy_(engine.buf_res + ffn_out)
