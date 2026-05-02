"""CKernelBagelFP4Batched — B-wide batched forward for Pi0.7 3-subgoal denoise.

Collapses N sequential denoise forwards for N independent subgoals into a
single forward with leading batch dim B. The shared obs+text prefilled KV
cache is replicated B times; each batch carries its own Sq=786 queries
(2 text bracket + 784 VAE tokens) and has independent attention.

Design (per NVFP4_PLAN.md Phase D):

* Main buffers flattened to [B*Sq, N]. Every GEMM runs at M=B*Sq — no
  weight changes, no new kernels.
* Text rows at [b*Sq + 0] and [b*Sq + Sq-1] for each batch b; gather to
  [2*B, N] text scratch, process with M=2*B, scatter back.
* Per-layer merged KV sized [B, total_kv, K_DIM] (flat [B*total_kv, K_DIM]).
  Per-step K/V append: B separate gpu_copy launches (one per batch) at byte
  offset (b*total_kv + kv_len)*K_DIM*2.
* flash_attn uses native batch dim: q_view [B, Sq, NHQ, HD] × kv_view
  [B, total_kv, NHKV, HD]. Independent attention per batch, no custom mask.
* RoPE cos/sin tables replicated B× to [B*Sq, HD]; all subgoals share
  absolute RoPE position.
* FP4 activation scratches (sc_gu, sc_dn) allocated at max_M = B*Sq.

At B=1 the forward is bit-identical to CKernelBagelFP4 (same call sequence,
same offsets), so this subclass is a strict generalisation.
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

from ckernel_bagel import D, NHQ, NHKV, HD, K_DIM, FFN, L, bf16, FP8
from ckernel_bagel_fp4 import CKernelBagelFP4, V_GATE, V_UP, V_DOWN

fp16 = torch.float16


class CKernelBagelFP4Batched(CKernelBagelFP4):
    """B-wide batched FP4 forward. Drop-in compatible at B=1."""

    def __init__(self, *args, B: int = 1, **kwargs):
        assert B >= 1, "B must be >= 1"
        self.B = int(B)
        # Parent __init__ calls _alloc_buffers and _alloc_fp4_buffers which
        # read self.B to size leading-dim accordingly.
        super().__init__(*args, **kwargs)
        # Skip und-expert path for the 2 "bracket" text tokens
        # ([start_of_image] / [end_of_image]). When True, these tokens ride
        # the gen expert path (same as VAE tokens) instead of being processed
        # by a separate FP8 M=2*B QKV+O+FFN stack. Paper has them on und
        # expert, but bracket tokens carry no prompt content \u2014 A/B validated
        # quality holds. Saves ~25 ms / fresh fwd (\u2248 -0.75 s Pi0.7 E2E).
        self.skip_und = False

    # ── Buffers ─────────────────────────────────────────────────────

    def _alloc_buffers(self):
        """Override CKernelBagel._alloc_buffers to use BSq."""
        Sq = self.Sq; B = self.B
        BSq = B * Sq
        total_kv = self.total_kv
        dev = 'cuda'

        # Main state [B*Sq, N]
        self.b_x  = torch.empty(BSq, D, dtype=bf16, device=dev)
        self.b_xn = torch.empty(BSq, D, dtype=bf16, device=dev)
        # Shared FP8 scratch sized for worst case (gen FFN: BSq * FFN bytes)
        self.b_fp8 = torch.empty(BSq * FFN, dtype=torch.uint8, device=dev)
        # Text FP8 scratch sized for text FFN at M=2*B (2*B*FFN bytes)
        self.b_text_fp8 = torch.empty(2 * B * FFN, dtype=torch.uint8, device=dev)
        # Gen QKV/output
        self.b_q = torch.empty(BSq, D,     dtype=bf16, device=dev)
        self.b_k = torch.empty(BSq, K_DIM, dtype=bf16, device=dev)
        self.b_v = torch.empty(BSq, K_DIM, dtype=bf16, device=dev)
        self.b_o = torch.empty(BSq, D,     dtype=bf16, device=dev)
        # FFN gen
        self.b_gate = torch.empty(BSq, FFN, dtype=bf16, device=dev)
        self.b_up   = torch.empty(BSq, FFN, dtype=bf16, device=dev)
        self.b_down = torch.empty(BSq, D,   dtype=bf16, device=dev)
        # Text scratch [2*B, N]
        self.b_text_x    = torch.empty(2 * B, D,     dtype=bf16, device=dev)
        self.b_text_q    = torch.empty(2 * B, D,     dtype=bf16, device=dev)
        self.b_text_k    = torch.empty(2 * B, K_DIM, dtype=bf16, device=dev)
        self.b_text_v    = torch.empty(2 * B, K_DIM, dtype=bf16, device=dev)
        self.b_text_ao   = torch.empty(2 * B, D,     dtype=bf16, device=dev)
        self.b_text_o    = torch.empty(2 * B, D,     dtype=bf16, device=dev)
        self.b_text_gate = torch.empty(2 * B, FFN,   dtype=bf16, device=dev)
        self.b_text_up   = torch.empty(2 * B, FFN,   dtype=bf16, device=dev)
        self.b_text_silu = torch.empty(2 * B, FFN,   dtype=bf16, device=dev)
        self.b_text_down = torch.empty(2 * B, D,     dtype=bf16, device=dev)
        # Per-layer merged KV, [B, total_kv, K_DIM] flat
        self.b_k_merged = [torch.empty(B * total_kv, K_DIM, dtype=bf16, device=dev)
                           for _ in range(L)]
        self.b_v_merged = [torch.empty(B * total_kv, K_DIM, dtype=bf16, device=dev)
                           for _ in range(L)]

    def _alloc_fp4_buffers(self):
        """FP4 FFN scratches sized at BSq."""
        Sq = self.Sq; B = self.B
        BSq = B * Sq
        dev = 'cuda'
        self.b_normed_bf16 = torch.empty(BSq, D,   dtype=bf16, device=dev)
        self.b_normed_fp16 = torch.empty(BSq, D,   dtype=fp16, device=dev)
        # Merged gate+up fp16 buffer for Class 1b Wab + 1c silu_mul_fp4 fused.
        self.b_gateup_fp16 = torch.empty(BSq, 2 * FFN, dtype=fp16, device=dev)
        self.b_down_fp16   = torch.empty(BSq, D,   dtype=fp16, device=dev)
        self.sc_gu = FP4ActScratch(BSq, D,   device=dev)
        self.sc_dn = FP4ActScratch(BSq, FFN, device=dev)

    # ── RoPE ────────────────────────────────────────────────────────

    def _precompute_rope(self, rope_pos=0):
        """Same absolute position for all Sq tokens, replicated B× for batched."""
        Sq = self.Sq; B = self.B
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, HD, 2,
                        dtype=torch.float32, device='cuda') / HD))
        pos = torch.full((Sq,), rope_pos, dtype=torch.float32, device='cuda')
        angles = torch.outer(pos, freqs)  # [Sq, HD//2]
        # rope_rotate_half_bf16 expects [total_rows, HD]; we need BSq rows,
        # each batch seeing the same per-token rope angles.
        cos_one = torch.zeros(Sq, HD, dtype=bf16, device='cuda')
        sin_one = torch.zeros(Sq, HD, dtype=bf16, device='cuda')
        cos_one[:, :HD // 2] = angles.cos().to(bf16)
        sin_one[:, :HD // 2] = angles.sin().to(bf16)
        # Replicate along a leading dim: shape [B*Sq, HD]
        self.cos_table = cos_one.repeat(B, 1).contiguous()
        self.sin_table = sin_one.repeat(B, 1).contiguous()

    # ── KV prefill (replicate across batch) ─────────────────────────

    def set_kv_cache(self, kv_cache):
        """Copy prefilled KV B× into merged buffers. kv_cache: list of (K,V) per layer,
        each [kv_len, NHKV, HD] (or [kv_len, K_DIM] after reshape)."""
        kv_len = self.kv_len
        total_kv = self.total_kv
        B = self.B
        for i in range(L):
            k_flat = kv_cache[i][0].reshape(kv_len, K_DIM).contiguous()
            v_flat = kv_cache[i][1].reshape(kv_len, K_DIM).contiguous()
            for b in range(B):
                # Each batch slot starts at b*total_kv rows, prefill lives in [0, kv_len)
                k_off = (b * total_kv) * K_DIM * 2
                v_off = (b * total_kv) * K_DIM * 2
                fvk.gpu_copy(self.b_k_merged[i].data_ptr() + k_off,
                             k_flat.data_ptr(), kv_len * K_DIM * 2, 0)
                fvk.gpu_copy(self.b_v_merged[i].data_ptr() + v_off,
                             v_flat.data_ptr(), kv_len * K_DIM * 2, 0)
        torch.cuda.synchronize()

    # ── Forward ─────────────────────────────────────────────────────

    def forward(self, s=0):
        Sq = self.Sq
        B = self.B
        BSq = B * Sq
        kv_len = self.kv_len
        total_kv = self.total_kv
        scale = 1.0 / math.sqrt(HD)
        act_scales_base = self.act_scales.data_ptr()

        D_BYTES     = D * 2
        KDIM_BYTES  = K_DIM * 2
        SQ_D_BYTES  = Sq * D_BYTES         # one batch's full-D stride
        SQ_K_BYTES  = Sq * KDIM_BYTES      # one batch's full-KDIM stride
        ROW_D       = D_BYTES              # bytes for one row at D
        ROW_K       = KDIM_BYTES           # bytes for one row at K_DIM
        LAST_D_OFF  = (Sq - 1) * D_BYTES   # byte offset to last row in a batch (D)
        LAST_K_OFF  = (Sq - 1) * KDIM_BYTES

        # Per-batch text row offsets inside b_x / b_q / b_k / b_v / b_o / b_down
        # batch b: first text row at byte (b*Sq)*D_BYTES, last at (b*Sq + Sq-1)*D_BYTES
        def _bx_row0_off(b, row_bytes, sq_stride_bytes):
            return b * sq_stride_bytes
        def _bx_rowLast_off(b, row_bytes, sq_stride_bytes, last_off):
            return b * sq_stride_bytes + last_off

        # Text scratch offsets: batch b → rows [2*b, 2*b+1]
        def _tx_off(b, row_bytes):
            return (2 * b) * row_bytes

        # ---- Class 1d: single-kernel gather/scatter for text bracket rows.
        # Old path did 2*B gpu_copy launches per call; new path = 1 kernel
        # launch. row_bytes = N*2 (bf16), so N = row_bytes // 2. The old
        # sq_stride_bytes / last_off arguments are unused here (derived from
        # Sq and N) but kept for call-site parity.
        def _gather_text(src_ptr, dst_ptr, sq_stride_bytes, last_off, row_bytes, s_):
            fwk.bf16_text_gather(src_ptr, dst_ptr, B, Sq, row_bytes // 2, s_)

        def _scatter_text(dst_ptr, src_ptr, sq_stride_bytes, last_off, row_bytes, s_):
            fwk.bf16_text_scatter(dst_ptr, src_ptr, B, Sq, row_bytes // 2, s_)

        # Initial input norm (F2-fuse-A for layer 0) over all B*Sq rows
        fvk.rms_norm_fp8(
            self.b_x.data_ptr(), self.layers[0]['ln_w'].data_ptr(),
            self.b_fp8.data_ptr(), BSq, D, 1e-6,
            act_scales_base + 0 * 4, s)

        for i in range(L):
            w = self.layers[i]
            a = self._alphas[i]
            is_fp4 = i in self.fp4_layers
            def _asp(j): return act_scales_base + (i * 7 + j) * 4

            # ── Gen Q/K/V (FP8 with fused per-col bias epilogue) at M=BSq ─
            # Class-1a: cutlass_fp8_sq_bias_bf16out fuses GEMM + add_bias in
            # one kernel. Strictly more accurate than 2-call (1 fewer bf16
            # round); math-equivalent within 1 ulp.
            fwk.cutlass_fp8_sq_bias_bf16out(
                self.b_fp8.data_ptr(), w['q_fp8'].data_ptr(), w['q_bias'].data_ptr(),
                self.b_q.data_ptr(), BSq, D, D, a['q'], s)
            fwk.cutlass_fp8_sq_bias_bf16out(
                self.b_fp8.data_ptr(), w['k_fp8'].data_ptr(), w['k_bias'].data_ptr(),
                self.b_k.data_ptr(), BSq, K_DIM, D, a['k'], s)
            fwk.cutlass_fp8_sq_bias_bf16out(
                self.b_fp8.data_ptr(), w['v_fp8'].data_ptr(), w['v_bias'].data_ptr(),
                self.b_v.data_ptr(), BSq, K_DIM, D, a['v'], s)

            # Class D: fused per-head rms_norm + RoPE (gen Q/K). Text rows in
            # b_q/b_k will be overwritten by und scatter below (und path also
            # calls the fused kernel with und_qn_w/und_kn_w, so the overlaid
            # text rows already carry correct norm + rope).
            fwk.qk_rmsnorm_rope_fused_bf16(
                self.b_q.data_ptr(), w['qn_w'].data_ptr(),
                self.cos_table.data_ptr(), self.sin_table.data_ptr(),
                BSq, NHQ, HD, 1e-6, s)
            fwk.qk_rmsnorm_rope_fused_bf16(
                self.b_k.data_ptr(), w['kn_w'].data_ptr(),
                self.cos_table.data_ptr(), self.sin_table.data_ptr(),
                BSq, NHKV, HD, 1e-6, s)

            # ── Und text Q/K/V (FP8) at M=2*B ────────────────────
            # Skipped when self.skip_und: bracket text tokens ride the gen
            # expert (already processed by gen QKV above).
            if not self.skip_und:
                ua = self._und_alphas[i]
                und_asp = self.und_act_scales.data_ptr() + (i * 4) * 4
                _gather_text(self.b_x.data_ptr(), self.b_text_x.data_ptr(),
                             SQ_D_BYTES, LAST_D_OFF, ROW_D, s)
                fvk.rms_norm(self.b_text_x.data_ptr(), w['und_ln_w'].data_ptr(),
                             self.b_text_x.data_ptr(), 2 * B, D, 1e-6, s)
                fvk.quantize_fp8_static(
                    self.b_text_x.data_ptr(), self.b_text_fp8.data_ptr(),
                    und_asp, 2 * B * D, s)
                # Und QKV with fused bias epilogue (Class-1a).
                fwk.cutlass_fp8_sq_bias_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_q_fp8'].data_ptr(), w['und_q_bias'].data_ptr(),
                    self.b_text_q.data_ptr(), 2 * B, D,     D, ua['q'], s)
                fwk.cutlass_fp8_sq_bias_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_k_fp8'].data_ptr(), w['und_k_bias'].data_ptr(),
                    self.b_text_k.data_ptr(), 2 * B, K_DIM, D, ua['k'], s)
                fwk.cutlass_fp8_sq_bias_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_v_fp8'].data_ptr(), w['und_v_bias'].data_ptr(),
                    self.b_text_v.data_ptr(), 2 * B, K_DIM, D, ua['v'], s)
                # Class D: fused rms_norm + RoPE on und text rows.
                fwk.qk_rmsnorm_rope_fused_bf16(
                    self.b_text_q.data_ptr(), w['und_qn_w'].data_ptr(),
                    self.cos_table.data_ptr(), self.sin_table.data_ptr(),
                    2 * B, NHQ, HD, 1e-6, s)
                fwk.qk_rmsnorm_rope_fused_bf16(
                    self.b_text_k.data_ptr(), w['und_kn_w'].data_ptr(),
                    self.cos_table.data_ptr(), self.sin_table.data_ptr(),
                    2 * B, NHKV, HD, 1e-6, s)

                # Scatter text rows back into gen Q/K/V
                _scatter_text(self.b_q.data_ptr(), self.b_text_q.data_ptr(),
                              SQ_D_BYTES, LAST_D_OFF, ROW_D, s)
                _scatter_text(self.b_k.data_ptr(), self.b_text_k.data_ptr(),
                              SQ_K_BYTES, LAST_K_OFF, ROW_K, s)
                _scatter_text(self.b_v.data_ptr(), self.b_text_v.data_ptr(),
                              SQ_K_BYTES, LAST_K_OFF, ROW_K, s)

            # RoPE already applied inside qk_rmsnorm_rope_fused_bf16 (Class D),
            # and und path's fused kernel pre-rotates text rows before scatter,
            # so no separate rope pass needed here.

            # ── Append K/V to merged cache (per-batch at offset kv_len) ──
            for b in range(B):
                k_dst = self.b_k_merged[i].data_ptr() + (b * total_kv + kv_len) * KDIM_BYTES
                v_dst = self.b_v_merged[i].data_ptr() + (b * total_kv + kv_len) * KDIM_BYTES
                k_src = self.b_k.data_ptr() + b * SQ_K_BYTES
                v_src = self.b_v.data_ptr() + b * SQ_K_BYTES
                fvk.gpu_copy(k_dst, k_src, Sq * KDIM_BYTES, s)
                fvk.gpu_copy(v_dst, v_src, Sq * KDIM_BYTES, s)

            # ── Flash Attention (native batch dim, independent per batch) ──
            q_view = self.b_q.view(B, Sq, NHQ, HD)
            k_view = self.b_k_merged[i].view(B, total_kv, NHKV, HD)
            v_view = self.b_v_merged[i].view(B, total_kv, NHKV, HD)
            out_ao = flash_attn_func(q_view, k_view, v_view,
                                     softmax_scale=scale, causal=False)
            ao_ptr = out_ao.data_ptr()

            # ── O projection (gen FP8 for all BSq; und FP8 overwrite for text) ──
            fvk.quantize_fp8_static(
                ao_ptr, self.b_fp8.data_ptr(),
                _asp(3), BSq * D, s)
            fvk.cutlass_fp8_sq_bf16out(
                self.b_fp8.data_ptr(), w['o_fp8'].data_ptr(),
                self.b_o.data_ptr(), BSq, D, D, a['o'], 0.0, s)

            # Text O — gather ao rows, apply und_o, scatter back (skipped when skip_und)
            if not self.skip_und:
                _gather_text(ao_ptr, self.b_text_ao.data_ptr(),
                             SQ_D_BYTES, LAST_D_OFF, ROW_D, s)
                fvk.quantize_fp8_static(
                    self.b_text_ao.data_ptr(), self.b_text_fp8.data_ptr(),
                    self.und_act_scales.data_ptr() + (i * 4 + 1) * 4, 2 * B * D, s)
                fvk.cutlass_fp8_sq_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_o_fp8'].data_ptr(),
                    self.b_text_o.data_ptr(), 2 * B, D, D, ua['o'], 0.0, s)
                _scatter_text(self.b_o.data_ptr(), self.b_text_o.data_ptr(),
                              SQ_D_BYTES, LAST_D_OFF, ROW_D, s)

            # ── FFN: FP8 or FP4 branch ──────────────────────────
            if is_fp4:
                fp4 = self.fp4_ffn[i]
                # ROI A fused: x ← x+b_o + rms_norm×ln_baked + FP4 + SFA.
                fwk.bagel_res_rms_mul_fp4_sfa_bf16(
                    self.b_x.data_ptr(), self.b_o.data_ptr(),
                    fp4['ln_baked'].data_ptr(),
                    self.sc_gu.packed.data_ptr(),
                    self.sc_gu.sfa.data_ptr(),
                    BSq, D, 1e-6, s)
                # Class 1b Wab: merged gate+up FP4 GEMM at N=2*FFN writing
                # [BSq, 2*FFN] fp16 buffer (gate cols [0..FFN), up cols
                # [FFN..2*FFN)). Replaces 2 separate GEMMs.
                fvk_fp4.cutlass_fp4_gemm_variant(
                    V_GATE,
                    self.sc_gu.packed.data_ptr(), self.sc_gu.sfa.data_ptr(),
                    fp4['gateup']['packed'].data_ptr(),
                    fp4['gateup']['sfb'].data_ptr(),
                    self.b_gateup_fp16.data_ptr(),
                    BSq, 2 * FFN, D, 1.0, 0.0, s)
                # Class 1c: fused SiLU(gate)*up → FP4 + SFA in one launch.
                _clamp = 0.0
                if isinstance(self.silu_clamp_max_abs, dict):
                    _clamp = float(self.silu_clamp_max_abs.get(i, 0.0))
                elif self.silu_clamp_max_abs:
                    _clamp = float(self.silu_clamp_max_abs)
                assert _clamp == 0.0, (
                    "silu_clamp_max_abs incompatible with 1c fused path")
                fwk.bagel_silu_mul_fp4_sfa_v2_fp16(
                    self.b_gateup_fp16.data_ptr(),
                    self.sc_dn.packed.data_ptr(),
                    self.sc_dn.sfa.data_ptr(),
                    BSq, FFN, s)
                fwk.cutlass_fp4_gemm_bf16out_variant(
                    V_DOWN,
                    self.sc_dn.packed.data_ptr(), self.sc_dn.sfa.data_ptr(),
                    fp4['down']['packed'].data_ptr(), fp4['down']['sfb'].data_ptr(),
                    self.b_down.data_ptr(),
                    BSq, D, FFN, 1.0, 0.0, s)
            else:
                fvk.residual_add_rms_norm_fp8(
                    self.b_x.data_ptr(), self.b_o.data_ptr(),
                    w['ln2_w'].data_ptr(), self.b_fp8.data_ptr(),
                    BSq, D, 1e-6, _asp(4), s)

                fvk.cutlass_fp8_t1_bf16out(
                    self.b_fp8.data_ptr(), w['gate_fp8'].data_ptr(),
                    self.b_gate.data_ptr(), BSq, FFN, D, a['gate'], 0.0, s)
                fvk.cutlass_fp8_t1_bf16out(
                    self.b_fp8.data_ptr(), w['up_fp8'].data_ptr(),
                    self.b_up.data_ptr(), BSq, FFN, D, a['up'], 0.0, s)
                fwk.silu_mul_split_fp8_bf16(
                    self.b_gate.data_ptr(), self.b_up.data_ptr(),
                    self.b_fp8.data_ptr(), BSq * FFN, _asp(6), s)
                fvk.cutlass_fp8_wide_bf16out(
                    self.b_fp8.data_ptr(), w['down_fp8'].data_ptr(),
                    self.b_down.data_ptr(), BSq, D, FFN, a['down'], 0.0, s)

            # ── Text FFN (FP8 at M=2*B) ─────────────────────────
            # Skipped when skip_und: text rows already have gen FFN output in b_down.
            if not self.skip_und:
                _gather_text(self.b_x.data_ptr(), self.b_text_x.data_ptr(),
                             SQ_D_BYTES, LAST_D_OFF, ROW_D, s)
                fvk.rms_norm(self.b_text_x.data_ptr(), w['und_pn_w'].data_ptr(),
                             self.b_text_x.data_ptr(), 2 * B, D, 1e-6, s)
                fvk.quantize_fp8_static(
                    self.b_text_x.data_ptr(), self.b_text_fp8.data_ptr(),
                    self.und_act_scales.data_ptr() + (i * 4 + 2) * 4, 2 * B * D, s)
                fvk.cutlass_fp8_t1_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_gate_fp8'].data_ptr(),
                    self.b_text_gate.data_ptr(), 2 * B, FFN, D, ua['gate'], 0.0, s)
                fvk.cutlass_fp8_t1_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_up_fp8'].data_ptr(),
                    self.b_text_up.data_ptr(), 2 * B, FFN, D, ua['up'], 0.0, s)
                fwk.silu_mul_split_fp8_bf16(
                    self.b_text_gate.data_ptr(), self.b_text_up.data_ptr(),
                    self.b_text_fp8.data_ptr(), 2 * B * FFN,
                    self.und_act_scales.data_ptr() + (i * 4 + 3) * 4, s)
                fvk.cutlass_fp8_wide_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_down_fp8'].data_ptr(),
                    self.b_text_down.data_ptr(), 2 * B, D, FFN, ua['down'], 0.0, s)
                _scatter_text(self.b_down.data_ptr(), self.b_text_down.data_ptr(),
                              SQ_D_BYTES, LAST_D_OFF, ROW_D, s)

            # ── F2 fuse B: post-FFN residual + NEXT layer input norm ─
            if i < L - 1:
                fvk.residual_add_rms_norm_fp8(
                    self.b_x.data_ptr(), self.b_down.data_ptr(),
                    self.layers[i + 1]['ln_w'].data_ptr(),
                    self.b_fp8.data_ptr(),
                    BSq, D, 1e-6,
                    act_scales_base + ((i + 1) * 7 + 0) * 4, s)
            else:
                fvk.residual_add(self.b_x.data_ptr(), self.b_down.data_ptr(),
                                 BSq * D, s)

        # ── Final norm ──────────────────────────────────────────
        fvk.rms_norm(self.b_x.data_ptr(), self.gen_final_norm.data_ptr(),
                     self.b_xn.data_ptr(), BSq, D, 1e-6, s)
        if not self.skip_und:
            _gather_text(self.b_x.data_ptr(), self.b_text_x.data_ptr(),
                         SQ_D_BYTES, LAST_D_OFF, ROW_D, s)
            fvk.rms_norm(self.b_text_x.data_ptr(), self.und_final_norm.data_ptr(),
                         self.b_text_x.data_ptr(), 2 * B, D, 1e-6, s)
            _scatter_text(self.b_xn.data_ptr(), self.b_text_x.data_ptr(),
                          SQ_D_BYTES, LAST_D_OFF, ROW_D, s)
