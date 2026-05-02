"""CKernelBagelFP4 — per-layer FP4 FFN swap for BAGEL world-model.

Inherits CKernelBagel. For layers in ``fp4_layers`` the FFN (Gate+Up+Down)
runs NVFP4 with AWQ-baked ln2_w * inv_s_gu in the rms-norm and s_gu baked
into gate/up FP4 weights. Down uses no AWQ per B4 (Δcos<0.001).

Other layers run bit-identical to CKernelBagel (FP8 everywhere).

Hot-loop kernel chain for an FP4 FFN layer i (residual stream stays BF16):
   1. fvk.residual_add      (b_x += b_o)          — bf16, safe
   2. fvk.rms_norm          (ln_baked weight)     — bf16 normed out
   3. fwk.cast_bf16_to_fp16                        — safe (amax~30)
   4. fvk_fp4.quantize_fp4_dynamic_sfa_fp16       — FP4 gate/up input
   5. fvk_fp4.cutlass_fp4_gemm_variant V6 (gate)  → fp16 [Sq, FFN]
   6. fvk_fp4.cutlass_fp4_gemm_variant V6 (up)    → fp16 [Sq, FFN]
   7. fwk.silu_mul_fp16                            → fp16 hidden
   8. fvk_fp4.quantize_fp4_dynamic_sfa_fp16       — FP4 down input
   9. fvk_fp4.cutlass_fp4_gemm_variant V8 (down)  → fp16 [Sq, D]
  10. fwk.cast_fp16_to_bf16                        → bf16 b_down
  11. text FFN FP8 overlay (parent path, unchanged)
  12. parent's F2 fuse-B: residual_add_rms_norm_fp8(b_x, b_down, next_ln_w, ...)
"""
from __future__ import annotations
import os, sys, math, torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_WM = os.path.dirname(_THIS)
_VLA = os.path.dirname(_WM)
sys.path.insert(0, _VLA)
sys.path.insert(0, _WM)

import flash_rt.flash_rt_kernels as fvk
import flash_rt.flash_rt_fp4 as fvk_fp4
import flash_wm_kernels as fwk
from flash_attn import flash_attn_func
from flash_rt.executors.fp4_utils import FP4ActScratch

from ckernel_bagel import CKernelBagel, D, NHQ, NHKV, HD, K_DIM, FFN, L, bf16, FP8

fp16 = torch.float16

# Variant picks from B1 shape bench @ Sq=786 (not pi05's Se=968 defaults)
V_GATE = 6   # [N=18944, K=3584]  wide-N cluster1x1x1
V_UP   = 6
V_DOWN = 8   # [N=3584, K=18944]  wide-NK cluster1x1x1


class CKernelBagelFP4(CKernelBagel):
    """Per-layer FP4 FFN swap. Drop-in compatible with CKernelBagel."""

    def __init__(self, *args, fp4_layers=(), fp4_ffn=None, **kwargs):
        """
        Args (in addition to CKernelBagel's):
            fp4_layers: iterable of layer indices to route through FP4 FFN.
            fp4_ffn: dict from prepare_gen_fp4_ffn — required if fp4_layers
                non-empty. Provides gate/up/down FP4 packed+SFB and ln_baked.
        """
        super().__init__(*args, **kwargs)
        self.fp4_layers = frozenset(int(l) for l in fp4_layers)
        self.fp4_ffn = fp4_ffn if self.fp4_layers else None
        # Path C safety clamp for FP4 Down accumulator (L5/L9 overflow
        # workaround). Default 0.0 = no clamp = old behaviour unchanged.
        # silu_clamp_max_abs may also be a dict {layer_idx: max_abs} to
        # apply clamp only on specific layers (others run unclamped).
        self.silu_clamp_max_abs = 0.0
        # skip_und: when True, skip the und-expert path entirely for the 2
        # bracket text tokens at rows [0] and [Sq-1]. They inherit gen-expert
        # outputs instead. Paper has text on und, but bracket tokens carry no
        # prompt content \u2014 A/B validated. Saves ~25 ms / fresh fwd.
        self.skip_und = False
        if self.fp4_layers:
            assert fp4_ffn is not None, "fp4_ffn required when fp4_layers non-empty"
            missing = [l for l in self.fp4_layers if l not in fp4_ffn]
            assert not missing, f"fp4_ffn missing layers: {missing}"
            self._alloc_fp4_buffers()

    def _alloc_fp4_buffers(self):
        """One-time scratch allocations for FP4 FFN branch."""
        Sq = self.Sq
        dev = 'cuda'
        self.b_normed_bf16 = torch.empty(Sq, D,   dtype=bf16, device=dev)
        self.b_normed_fp16 = torch.empty(Sq, D,   dtype=fp16, device=dev)
        # Merged gate+up fp16 buffer [Sq, 2*FFN] for Class 1b Wab GEMM.
        # Columns [0..FFN) = gate, [FFN..2*FFN) = up — layout required by
        # gate_silu_mul_fp4_sfa_v2_fp16 (1c fused kernel).
        self.b_gateup_fp16 = torch.empty(Sq, 2 * FFN, dtype=fp16, device=dev)
        self.b_down_fp16   = torch.empty(Sq, D,   dtype=fp16, device=dev)
        # FP4 activation scratches (packed + SFA)
        self.sc_gu = FP4ActScratch(Sq, D,   device=dev)
        self.sc_dn = FP4ActScratch(Sq, FFN, device=dev)

    # ── Forward ──
    # Structured as a copy of CKernelBagel.forward with FFN-section branch.
    def forward(self, s=0):
        Sq = self.Sq
        kv_len = self.kv_len
        total_kv = self.total_kv
        scale = 1.0 / math.sqrt(HD)
        LAST_D    = (Sq - 1) * D * 2
        LAST_KDIM = (Sq - 1) * K_DIM * 2
        act_scales_base = self.act_scales.data_ptr()

        # Initial input norm — F2-fuse-A of layer 0
        fvk.rms_norm_fp8(
            self.b_x.data_ptr(), self.layers[0]['ln_w'].data_ptr(),
            self.b_fp8.data_ptr(), Sq, D, 1e-6,
            act_scales_base + 0 * 4, s)

        for i in range(L):
            w = self.layers[i]
            a = self._alphas[i]
            is_fp4 = i in self.fp4_layers
            def _asp(j): return act_scales_base + (i * 7 + j) * 4

            # ── Gen Q/K/V (FP8 with fused per-col bias epilogue — Class-1a) ──
            fwk.cutlass_fp8_sq_bias_bf16out(
                self.b_fp8.data_ptr(), w['q_fp8'].data_ptr(), w['q_bias'].data_ptr(),
                self.b_q.data_ptr(), Sq, D, D, a['q'], s)
            fwk.cutlass_fp8_sq_bias_bf16out(
                self.b_fp8.data_ptr(), w['k_fp8'].data_ptr(), w['k_bias'].data_ptr(),
                self.b_k.data_ptr(), Sq, K_DIM, D, a['k'], s)
            fwk.cutlass_fp8_sq_bias_bf16out(
                self.b_fp8.data_ptr(), w['v_fp8'].data_ptr(), w['v_bias'].data_ptr(),
                self.b_v.data_ptr(), Sq, K_DIM, D, a['v'], s)

            # Class D: fused per-head rms_norm + RoPE on gen Q/K.
            fwk.qk_rmsnorm_rope_fused_bf16(
                self.b_q.data_ptr(), w['qn_w'].data_ptr(),
                self.cos_table.data_ptr(), self.sin_table.data_ptr(),
                Sq, NHQ, HD, 1e-6, s)
            fwk.qk_rmsnorm_rope_fused_bf16(
                self.b_k.data_ptr(), w['kn_w'].data_ptr(),
                self.cos_table.data_ptr(), self.sin_table.data_ptr(),
                Sq, NHKV, HD, 1e-6, s)

            # ── Und text Q/K/V (FP8) ──────────────────────── (skipped when skip_und)
            if not self.skip_und:
                ua = self._und_alphas[i]
                und_asp = self.und_act_scales.data_ptr() + (i * 4) * 4
                fvk.gpu_copy(self.b_text_x.data_ptr(),
                             self.b_x.data_ptr(),              D * 2, s)
                fvk.gpu_copy(self.b_text_x.data_ptr() + D * 2,
                             self.b_x.data_ptr() + LAST_D,     D * 2, s)
                fvk.rms_norm(self.b_text_x.data_ptr(), w['und_ln_w'].data_ptr(),
                             self.b_text_x.data_ptr(), 2, D, 1e-6, s)
                fvk.quantize_fp8_static(
                    self.b_text_x.data_ptr(), self.b_text_fp8.data_ptr(),
                    und_asp, 2 * D, s)
                fwk.cutlass_fp8_sq_bias_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_q_fp8'].data_ptr(), w['und_q_bias'].data_ptr(),
                    self.b_text_q.data_ptr(), 2, D,     D, ua['q'], s)
                fwk.cutlass_fp8_sq_bias_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_k_fp8'].data_ptr(), w['und_k_bias'].data_ptr(),
                    self.b_text_k.data_ptr(), 2, K_DIM, D, ua['k'], s)
                fwk.cutlass_fp8_sq_bias_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_v_fp8'].data_ptr(), w['und_v_bias'].data_ptr(),
                    self.b_text_v.data_ptr(), 2, K_DIM, D, ua['v'], s)
                # Class D: fused rms_norm + RoPE on und text rows (M=2).
                fwk.qk_rmsnorm_rope_fused_bf16(
                    self.b_text_q.data_ptr(), w['und_qn_w'].data_ptr(),
                    self.cos_table.data_ptr(), self.sin_table.data_ptr(),
                    2, NHQ, HD, 1e-6, s)
                fwk.qk_rmsnorm_rope_fused_bf16(
                    self.b_text_k.data_ptr(), w['und_kn_w'].data_ptr(),
                    self.cos_table.data_ptr(), self.sin_table.data_ptr(),
                    2, NHKV, HD, 1e-6, s)

                # Overwrite text rows in gen QKV
                fvk.gpu_copy(self.b_q.data_ptr(),
                             self.b_text_q.data_ptr(),               D * 2, s)
                fvk.gpu_copy(self.b_q.data_ptr() + LAST_D,
                             self.b_text_q.data_ptr() + D * 2,       D * 2, s)
                fvk.gpu_copy(self.b_k.data_ptr(),
                             self.b_text_k.data_ptr(),               K_DIM * 2, s)
                fvk.gpu_copy(self.b_k.data_ptr() + LAST_KDIM,
                             self.b_text_k.data_ptr() + K_DIM * 2,   K_DIM * 2, s)
                fvk.gpu_copy(self.b_v.data_ptr(),
                             self.b_text_v.data_ptr(),               K_DIM * 2, s)
                fvk.gpu_copy(self.b_v.data_ptr() + LAST_KDIM,
                             self.b_text_v.data_ptr() + K_DIM * 2,   K_DIM * 2, s)

            # RoPE already applied inside qk_rmsnorm_rope_fused_bf16 (Class D);
            # und path's fused kernel pre-rotates text rows before scatter.

            # Append K/V to merged cache
            fvk.gpu_copy(self.b_k_merged[i].data_ptr() + kv_len * K_DIM * 2,
                         self.b_k.data_ptr(), Sq * K_DIM * 2, s)
            fvk.gpu_copy(self.b_v_merged[i].data_ptr() + kv_len * K_DIM * 2,
                         self.b_v.data_ptr(), Sq * K_DIM * 2, s)

            # Flash Attention
            q_view = self.b_q.view(1, Sq, NHQ, HD)
            k_view = self.b_k_merged[i][:total_kv].view(1, total_kv, NHKV, HD)
            v_view = self.b_v_merged[i][:total_kv].view(1, total_kv, NHKV, HD)
            out_ao = flash_attn_func(q_view, k_view, v_view,
                                     softmax_scale=scale, causal=False)
            ao_ptr = out_ao.data_ptr()

            # O projection (gen FP8 for all Sq, und bf16 for text)
            fvk.quantize_fp8_static(
                ao_ptr, self.b_fp8.data_ptr(),
                _asp(3), Sq * D, s)
            fvk.cutlass_fp8_sq_bf16out(
                self.b_fp8.data_ptr(), w['o_fp8'].data_ptr(),
                self.b_o.data_ptr(), Sq, D, D, a['o'], 0.0, s)

            if not self.skip_und:
                fvk.gpu_copy(self.b_text_ao.data_ptr(),
                             ao_ptr,                               D * 2, s)
                fvk.gpu_copy(self.b_text_ao.data_ptr() + D * 2,
                             ao_ptr + LAST_D,                      D * 2, s)
                fvk.quantize_fp8_static(
                    self.b_text_ao.data_ptr(), self.b_text_fp8.data_ptr(),
                    self.und_act_scales.data_ptr() + (i * 4 + 1) * 4, 2 * D, s)
                fvk.cutlass_fp8_sq_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_o_fp8'].data_ptr(),
                    self.b_text_o.data_ptr(), 2, D, D, ua['o'], 0.0, s)
                fvk.gpu_copy(self.b_o.data_ptr(),
                             self.b_text_o.data_ptr(),              D * 2, s)
                fvk.gpu_copy(self.b_o.data_ptr() + LAST_D,
                             self.b_text_o.data_ptr() + D * 2,      D * 2, s)

            # ── FFN: FP8 or FP4 branch ──────────────────────────
            if is_fp4:
                fp4 = self.fp4_ffn[i]
                # ROI A fused: x ← x+b_o (bf16 in-place) + rms_norm(× ln_baked)
                # + FP4 quant + SFA — one kernel. Eliminates b_normed_fp16
                # bridge buffer. bf16 residual preserved (L5/L9 safety).
                fwk.bagel_res_rms_mul_fp4_sfa_bf16(
                    self.b_x.data_ptr(), self.b_o.data_ptr(),
                    fp4['ln_baked'].data_ptr(),
                    self.sc_gu.packed.data_ptr(),
                    self.sc_gu.sfa.data_ptr(),
                    Sq, D, 1e-6, s)
                # 5. Merged Wab GEMM (Class 1b): one FP4 GEMM at N=2*FFN
                #    writing gate cols [0..FFN) and up cols [FFN..2*FFN) of
                #    a [Sq, 2*FFN] fp16 buffer. Replaces 2 separate GEMMs.
                fvk_fp4.cutlass_fp4_gemm_variant(
                    V_GATE,
                    self.sc_gu.packed.data_ptr(), self.sc_gu.sfa.data_ptr(),
                    fp4['gateup']['packed'].data_ptr(),
                    fp4['gateup']['sfb'].data_ptr(),
                    self.b_gateup_fp16.data_ptr(),
                    Sq, 2 * FFN, D, 1.0, 0.0, s)
                # 6. Class 1c fused: SiLU(gate)*up → FP4 + SFA in one kernel.
                #    Consumes merged [Sq, 2*FFN] fp16 directly, produces
                #    FP4 down-input + SFA. Replaces silu_mul_fp16 + quantize.
                #    Note: silu_clamp path deprecated under 1c (no clamp
                #    variant for fused FP4 output yet). Reverts to old split
                #    path if a non-zero clamp is requested.
                _clamp = 0.0
                if isinstance(self.silu_clamp_max_abs, dict):
                    _clamp = float(self.silu_clamp_max_abs.get(i, 0.0))
                elif self.silu_clamp_max_abs:
                    _clamp = float(self.silu_clamp_max_abs)
                assert _clamp == 0.0, (
                    "silu_clamp_max_abs incompatible with 1c fused path")
                # Use flash_wm's TRUE-SiLU variant (upstream's kernel is
                # named _silu_ but implements GELU-tanh — wrong for BAGEL).
                fwk.bagel_silu_mul_fp4_sfa_v2_fp16(
                    self.b_gateup_fp16.data_ptr(),
                    self.sc_dn.packed.data_ptr(),
                    self.sc_dn.sfa.data_ptr(),
                    Sq, FFN, s)
                # 9. FP4 down GEMM → BF16 directly (flash_wm cutlass_fp4_gemm_bf16out).
                #    Replaces the old two-step "FP4→fp16 + cast_fp16_to_bf16".
                #    BF16 accumulator range (3.4e38) is safely above the fp16
                #    overflow that broke L5 / L9 at low-t (see debug_l9_t04).
                fwk.cutlass_fp4_gemm_bf16out_variant(
                    V_DOWN,
                    self.sc_dn.packed.data_ptr(), self.sc_dn.sfa.data_ptr(),
                    fp4['down']['packed'].data_ptr(), fp4['down']['sfb'].data_ptr(),
                    self.b_down.data_ptr(),
                    Sq, D, FFN, 1.0, 0.0, s)
            else:
                # FP8 branch — bit-identical to parent
                fvk.residual_add_rms_norm_fp8(
                    self.b_x.data_ptr(), self.b_o.data_ptr(),
                    w['ln2_w'].data_ptr(), self.b_fp8.data_ptr(),
                    Sq, D, 1e-6, _asp(4), s)

                fvk.cutlass_fp8_t1_bf16out(
                    self.b_fp8.data_ptr(), w['gate_fp8'].data_ptr(),
                    self.b_gate.data_ptr(), Sq, FFN, D, a['gate'], 0.0, s)
                fvk.cutlass_fp8_t1_bf16out(
                    self.b_fp8.data_ptr(), w['up_fp8'].data_ptr(),
                    self.b_up.data_ptr(), Sq, FFN, D, a['up'], 0.0, s)
                fwk.silu_mul_split_fp8_bf16(
                    self.b_gate.data_ptr(), self.b_up.data_ptr(),
                    self.b_fp8.data_ptr(), Sq * FFN, _asp(6), s)
                fvk.cutlass_fp8_wide_bf16out(
                    self.b_fp8.data_ptr(), w['down_fp8'].data_ptr(),
                    self.b_down.data_ptr(), Sq, D, FFN, a['down'], 0.0, s)

            # ── Text FFN (FP8 M=2) ── (skipped when skip_und)
            if not self.skip_und:
                fvk.gpu_copy(self.b_text_x.data_ptr(),
                             self.b_x.data_ptr(),             D * 2, s)
                fvk.gpu_copy(self.b_text_x.data_ptr() + D * 2,
                             self.b_x.data_ptr() + LAST_D,    D * 2, s)
                fvk.rms_norm(self.b_text_x.data_ptr(), w['und_pn_w'].data_ptr(),
                             self.b_text_x.data_ptr(), 2, D, 1e-6, s)
                fvk.quantize_fp8_static(
                    self.b_text_x.data_ptr(), self.b_text_fp8.data_ptr(),
                    self.und_act_scales.data_ptr() + (i * 4 + 2) * 4, 2 * D, s)
                fvk.cutlass_fp8_t1_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_gate_fp8'].data_ptr(),
                    self.b_text_gate.data_ptr(), 2, FFN, D, ua['gate'], 0.0, s)
                fvk.cutlass_fp8_t1_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_up_fp8'].data_ptr(),
                    self.b_text_up.data_ptr(), 2, FFN, D, ua['up'], 0.0, s)
                fwk.silu_mul_split_fp8_bf16(
                    self.b_text_gate.data_ptr(), self.b_text_up.data_ptr(),
                    self.b_text_fp8.data_ptr(), 2 * FFN,
                    self.und_act_scales.data_ptr() + (i * 4 + 3) * 4, s)
                fvk.cutlass_fp8_wide_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_down_fp8'].data_ptr(),
                    self.b_text_down.data_ptr(), 2, D, FFN, ua['down'], 0.0, s)

                fvk.gpu_copy(self.b_down.data_ptr(),
                             self.b_text_down.data_ptr(),            D * 2, s)
                fvk.gpu_copy(self.b_down.data_ptr() + LAST_D,
                             self.b_text_down.data_ptr() + D * 2,    D * 2, s)

            # ── F2 fuse B: post-FFN residual + NEXT layer input norm ─
            if i < L - 1:
                fvk.residual_add_rms_norm_fp8(
                    self.b_x.data_ptr(), self.b_down.data_ptr(),
                    self.layers[i + 1]['ln_w'].data_ptr(),
                    self.b_fp8.data_ptr(),
                    Sq, D, 1e-6,
                    act_scales_base + ((i + 1) * 7 + 0) * 4, s)
            else:
                fvk.residual_add(self.b_x.data_ptr(), self.b_down.data_ptr(),
                                 Sq * D, s)

        # ── Final norm ────────────────────────────────────────────
        fvk.rms_norm(self.b_x.data_ptr(), self.gen_final_norm.data_ptr(),
                     self.b_xn.data_ptr(), Sq, D, 1e-6, s)
        if not self.skip_und:
            fvk.gpu_copy(self.b_text_x.data_ptr(),
                         self.b_x.data_ptr(),             D * 2, s)
            fvk.gpu_copy(self.b_text_x.data_ptr() + D * 2,
                         self.b_x.data_ptr() + LAST_D,    D * 2, s)
            fvk.rms_norm(self.b_text_x.data_ptr(), self.und_final_norm.data_ptr(),
                         self.b_text_x.data_ptr(), 2, D, 1e-6, s)
            fvk.gpu_copy(self.b_xn.data_ptr(),
                         self.b_text_x.data_ptr(),             D * 2, s)
            fvk.gpu_copy(self.b_xn.data_ptr() + LAST_D,
                         self.b_text_x.data_ptr() + D * 2,     D * 2, s)
