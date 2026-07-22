#!/usr/bin/env python3
"""Cosmos3-Edge AV inverse-dynamics denoise compute path (Thor SM110).

Per docs/adding_new_model.md §0 rule 1 this is the model's own
``models/cosmos3_edge/pipeline_thor.py``: the quantized two-tower MoT
kernel-call sequence, the static und K/V cache, and the whole-denoise
CUDA-graph capture live here, self-contained.

The gen tower carries static vision conditioning plus 60 action tokens; the
head is the per-domain action projection. The und (text) tower is identical
across denoise steps, so its per-layer K/V (k_norm_und_for_gen + rope applied)
is computed once and cached; each step runs only the gen tower against the
cached und K/V through FA4 attention.

Precision (selected by the caller, not the environment):
  - quant="fp8"  : w8a8 FP8 E4M3 GEMMs (fp8_gemm_descale_bf16out) with
                   per-tensor weight scales and device-side dynamic activation
                   scales. Production default.
  - quant="bf16" : reference-accuracy path (bf16 GemmRunner GEMMs).
bf16_projs keeps named projections ("q", "k", "v", "o", "up", "down") in bf16.
This module reads no environment variables.
"""
from __future__ import annotations

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.frontends.torch._cosmos3_edge_thor_spec import SPEC
from flash_rt.models.cosmos3_edge.boundary_dump import EdgeBoundaryDump
from flash_rt.models.cosmos3_edge.dump_replay import EDGE_ACTION_MODEL_SHAPE, EDGE_FLAT_DIM
from flash_rt.models.cosmos3_edge.static_engine import EdgeStaticBufferEngine
from flash_rt.models.cosmos3_edge.static_unipc import EdgeStaticUniPCScheduler
from flash_rt.models.cosmos3_edge.weights import EdgeTransformerWeights

DEV = "cuda"
BF = torch.bfloat16
H, KVH, D, FF, HID, NL = (
    SPEC.num_heads,
    SPEC.num_kv_heads,
    SPEC.head_dim,
    SPEC.ffn_size,
    SPEC.hidden_size,
    SPEC.num_layers,
)
EPS = SPEC.rms_eps
NA, AD = EDGE_ACTION_MODEL_SHAPE
ALL_PROJS = ("q", "k", "v", "o", "up", "down")

_GEN_PROJ_KEYS = {
    "q": "self_attn.add_q_proj.weight",
    "k": "self_attn.add_k_proj.weight",
    "v": "self_attn.add_v_proj.weight",
    "o": "self_attn.to_add_out.weight",
    "up": "mlp_moe_gen.up_proj.weight",
    "down": "mlp_moe_gen.down_proj.weight",
}


class CosmosEdgeThor:
    """Static-buffer quantized gen-tower denoise engine for Cosmos3-Edge AV."""

    def __init__(
        self,
        weights: EdgeTransformerWeights,
        boundary_dump: EdgeBoundaryDump,
        timesteps: tuple[int, ...],
        *,
        quant: str = "fp8",
        bf16_projs: tuple[str, ...] = (),
        qkv_fused: bool = False,
        ffn_fp4: bool = False,
        slim_last: bool = True,
        shift: float,
    ):
        self.slim_last = bool(slim_last)
        if quant not in ("fp8", "bf16"):
            raise ValueError(f"unsupported quant mode: {quant}")
        self.quant = quant
        self.fp4 = None
        self.ffn_fp4 = False
        if ffn_fp4 and quant == "fp8" and not ({"up", "down"} & set(bf16_projs)):
            import flash_rt.flash_rt_fp4 as fp4mod

            if not fp4mod.has_nvfp4():
                raise RuntimeError("ffn_fp4 requested but flash_rt_fp4 lacks NVFP4 support")
            self.fp4 = fp4mod
            self.ffn_fp4 = True
        self.bf16_projs = set(ALL_PROJS) if quant == "bf16" else {p for p in bf16_projs if p}
        self.num_steps = len(timesteps)

        # Static conditioning, und K/V cache, action-head constants, and the
        # timestep-embed table come from the validated bring-up engine; the
        # denoise hot path below never calls back into it.
        base = EdgeStaticBufferEngine(weights, boundary_dump, device=DEV, dtype=BF)
        base.precompute_timestep_embeds(timesteps)
        fa4_getter = getattr(base.reference, "_get_fa4_fwd", None)
        self._fa4_fwd = fa4_getter() if fa4_getter is not None else None
        if self._fa4_fwd is None:
            raise RuntimeError("CosmosEdgeThor requires the FA4 forward entry on this build")

        self.NU = int(base.und_cache[0][0].shape[0])
        self.NG = int(base.full.shape[0])
        self.NJ = self.NU + self.NG
        NU, NG, NJ = self.NU, self.NG, self.NJ

        self.gemm = fvk.GemmRunner()
        z = lambda *s: torch.zeros(*s, device=DEV, dtype=BF)

        # --- weights: quantized gen-tower projections + bf16 norms ---
        def qf8(w_nk: torch.Tensor):  # per-tensor FP8 E4M3; GEMM wants B as [K,N]
            w = w_nk.t().contiguous()
            s = max(w.float().abs().max().item() / 448.0, 1e-12)
            f8 = (w.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
            return f8, torch.tensor([s], dtype=torch.float32, device=DEV)

        # QKV fused wide GEMM measured NEGATIVE on Thor (P50 6.056s vs 5.883s
        # separate: cuBLASLt N=4096 tactic worse than 3×separate + strided V
        # copy). Keep the path for re-evaluation; off by default.
        self.qkv_fused = bool(qkv_fused) and not ({"q", "k", "v"} & self.bf16_projs) and quant == "fp8"
        self.Wt = {}
        self.Wf8 = {}
        self.Wds = {}
        self.Wf4 = {}
        self.Wn = {}
        for li in range(NL):
            loaded = {}
            for nm, key in _GEN_PROJ_KEYS.items():
                loaded[nm] = weights.load_tensor(f"layers.{li}.{key}", device=DEV, dtype=BF)
            if self.qkv_fused:
                merged = torch.cat([loaded.pop("q"), loaded.pop("k"), loaded.pop("v")], dim=0)
                self.Wf8[(li, "qkv")], self.Wds[(li, "qkv")] = qf8(merged)
                del merged
            if self.ffn_fp4:
                for nm in ("up", "down"):
                    w16 = loaded.pop(nm).to(torch.float16).contiguous()
                    n_out, k_in = w16.shape
                    packed = torch.empty(n_out, k_in // 2, dtype=torch.uint8, device=DEV)
                    sfb = torch.empty(
                        self.fp4.sfa_size_bytes(n_out, k_in, True), dtype=torch.uint8, device=DEV)
                    rc = self.fp4.quantize_fp4_dynamic_sfa_mse_fp16(
                        w16.data_ptr(), packed.data_ptr(), sfb.data_ptr(), n_out, k_in, True, 0)
                    if rc != 0:
                        raise RuntimeError(f"fp4 weight quant failed rc={rc} layer {li} {nm}")
                    self.Wf4[(li, nm)] = (packed, sfb)
                    del w16
            for nm, w in loaded.items():
                if nm in self.bf16_projs:
                    self.Wt[(li, nm)] = w.t().contiguous()
                else:
                    self.Wf8[(li, nm)], self.Wds[(li, nm)] = qf8(w)
            for nm, key in (
                ("in_ln", "input_layernorm_moe_gen.weight"),
                ("post_ln", "post_attention_layernorm_moe_gen.weight"),
                ("q_norm", "self_attn.norm_added_q.weight"),
                ("k_norm", "self_attn.norm_added_k.weight"),
            ):
                self.Wn[(li, nm)] = weights.load_tensor(f"layers.{li}.{key}", device=DEV, dtype=BF)
        self.norm_g = weights.load_tensor("norm_moe_gen.weight", device=DEV, dtype=BF)

        # --- action head constants (already domain-selected by the base engine) ---
        self.action_in_w = base.action_in_w
        self.action_static_bias = base.action_static_bias
        self.action_out_w = base.action_out_w
        self.action_out_bias = base.action_out_bias
        self.raw_action_dim = base.raw_action_dim
        self.action_indexes = base.action_indexes.to(torch.int64).contiguous()
        self.t_emb = base.timestep_embed_cache
        if self.t_emb is None or self.t_emb.shape[0] != self.num_steps:
            raise RuntimeError("timestep embed table missing or mismatched")

        # --- static activation buffers ---
        self.gen_init = base.full.clone()  # vision conditioning; action rows zero
        self.Hb = z(NG, HID)
        self.nrm = z(NG, HID)
        self.nrm2 = z(NG, HID)
        self.Qt = z(NG, H * D)
        self.Kt = z(NG, KVH * D)
        self.QKVt = z(NG, (H + 2 * KVH) * D) if self.qkv_fused else None
        self.Qb = z(NG, H, D)
        self.attn = z(1, NG, H, D)
        self.lse = torch.empty(1, H, NG, dtype=torch.float32, device=DEV)
        self.ob = z(NG, HID)
        self.up = z(NG, FF)
        self.dn = z(NG, HID)
        self.action_input = z(NA, AD)
        self.action_encoded = z(NA, HID)
        self.action_hidden = z(NA, HID)
        self.action_norm = z(NA, HID)
        self.action_out = z(NA, AD)
        # Slim last layer: only the NA action rows feed the head, so the final
        # layer's Q/attention/o/FFN run at M=NA (K/V still need all rows).
        self.Qs = z(NA, H, D)
        self.attn_s = z(1, NA, H, D)
        self.lse_s = torch.empty(1, H, NA, dtype=torch.float32, device=DEV)
        self.velocity = z(EDGE_FLAT_DIM)
        self.latent = torch.zeros(EDGE_FLAT_DIM, device=DEV, dtype=torch.float32)
        if self.ffn_fp4:
            self.a4h = torch.empty(NG, HID // 2, dtype=torch.uint8, device=DEV)
            self.sfa_h = torch.empty(self.fp4.sfa_size_bytes(NG, HID, False), dtype=torch.uint8, device=DEV)
            self.up16 = torch.empty(NG, FF, dtype=torch.float16, device=DEV)
            self.a4f = torch.empty(NG, FF // 2, dtype=torch.uint8, device=DEV)
            self.sfa_f = torch.empty(self.fp4.sfa_size_bytes(NG, FF, False), dtype=torch.uint8, device=DEV)
            self.dn16 = torch.empty(NG, HID, dtype=torch.float16, device=DEV)
            # Fastest verified GEMM variant at the Edge FFN shapes; -1 = default API.
            self.fp4_variant = 3
            probe = torch.zeros(2, HID // 2, dtype=torch.uint8, device=DEV)
            probe_sfa = torch.zeros(self.fp4.sfa_size_bytes(2, HID, False), dtype=torch.uint8, device=DEV)
            probe_out = torch.empty(2, FF, dtype=torch.float16, device=DEV)
            rc = self.fp4.cutlass_fp4_gemm_variant(
                self.fp4_variant, probe.data_ptr(), probe_sfa.data_ptr(),
                self.Wf4[(0, "up")][0].data_ptr(), self.Wf4[(0, "up")][1].data_ptr(),
                probe_out.data_ptr(), 2, FF, HID, 1.0, 0.0, 0)
            if rc != 0:
                self.fp4_variant = -1
            probe_relu = torch.empty(2, FF // 2, dtype=torch.uint8, device=DEV)
            probe_relu_sfa = torch.empty(
                self.fp4.sfa_size_bytes(2, FF, False), dtype=torch.uint8, device=DEV)
            rc = self.fp4.cosmos3_edge_fp4_gemm_relu2_fp4out(
                probe.data_ptr(), probe_sfa.data_ptr(),
                self.Wf4[(0, "up")][0].data_ptr(), self.Wf4[(0, "up")][1].data_ptr(),
                probe_relu.data_ptr(), probe_relu_sfa.data_ptr(), 2, FF, HID, 0)
            self.fp4_relu2_epilogue = rc == 0
            del probe, probe_sfa, probe_out, probe_relu, probe_relu_sfa
        self.a8h = self.a8f = self.asc = None
        # Per-(layer, site) static activation scales for the fused quant chain.
        # Sites: 0 = qkv input, 1 = attention output, 2 = FFN input, 3 = down input.
        self.site_scale = torch.full((NL, 4), 1e-12, dtype=torch.float32, device=DEV)
        self.calibrated = False
        if quant == "fp8":
            self.a8h = torch.empty(NG, HID, dtype=torch.float8_e4m3fn, device=DEV)
            self.a8f = torch.empty(NG, FF, dtype=torch.float8_e4m3fn, device=DEV)
            self.asc = torch.empty(1, dtype=torch.float32, device=DEV)

        # --- per-layer joint K/V with the static und prefix pre-installed ---
        self.Kj = torch.zeros(NL, 1, NJ, KVH, D, device=DEV, dtype=BF)
        self.Vj = torch.zeros(NL, 1, NJ, KVH, D, device=DEV, dtype=BF)
        for li, (k_und, v_und) in enumerate(base.und_cache):
            self.Kj[li, 0, :NU].copy_(k_und)
            self.Vj[li, 0, :NU].copy_(v_und)

        # --- static rope tables (full_only sequence) ---
        t = boundary_dump.tensors
        self.cos_f = t["s00/layers/00/rope/cos/full_only_seq"].to(device=DEV, dtype=BF).contiguous()
        self.sin_f = t["s00/layers/00/rope/sin/full_only_seq"].to(device=DEV, dtype=BF).contiguous()

        self.compute_steps: frozenset[int] | None = None
        self.unipc = EdgeStaticUniPCScheduler(self.num_steps, device=torch.device(DEV), shift=shift)
        if not self.unipc.native_available:
            raise RuntimeError("native UniPC step binding is required")
        self.graph = None
        del base
        torch.cuda.synchronize()

    # ---- kernel helpers ----
    def _s(self):
        return torch.cuda.current_stream().cuda_stream

    def _site_ptr(self, li: int, site: int) -> int:
        return self.site_scale.data_ptr() + 4 * (li * 4 + site)

    def _proj(
        self,
        a_bf16: torch.Tensor,
        a_f8: torch.Tensor | None,
        li: int,
        nm: str,
        out: torch.Tensor,
        n_out: int,
        act_scale_ptr: int | None = None,
    ):
        m, k = a_bf16.shape
        if (li, nm) in self.Wf8:
            assert a_f8 is not None
            scale_ptr = self.asc.data_ptr() if act_scale_ptr is None else act_scale_ptr
            fvk.fp8_gemm_descale_bf16out(
                a_f8.data_ptr(), self.Wf8[(li, nm)].data_ptr(), out.data_ptr(),
                m, n_out, k, scale_ptr, self.Wds[(li, nm)].data_ptr(), self._s(),
            )
        else:
            self.gemm.bf16_nn(a_bf16.data_ptr(), self.Wt[(li, nm)].data_ptr(), out.data_ptr(), m, n_out, k, self._s())

    def _fp4_gemm(self, a_packed, a_sfa, li: int, nm: str, out16, m: int, n: int, k: int, s: int):
        w_packed, w_sfb = self.Wf4[(li, nm)]
        if self.fp4_variant >= 0:
            rc = self.fp4.cutlass_fp4_gemm_variant(
                self.fp4_variant, a_packed.data_ptr(), a_sfa.data_ptr(),
                w_packed.data_ptr(), w_sfb.data_ptr(), out16.data_ptr(), m, n, k, 1.0, 0.0, s)
        else:
            rc = self.fp4.cutlass_fp4_sq_fp16(
                a_packed.data_ptr(), a_sfa.data_ptr(),
                w_packed.data_ptr(), w_sfb.data_ptr(), out16.data_ptr(), m, n, k, 1.0, 0.0, s)
        if rc != 0:
            raise RuntimeError(f"fp4 GEMM failed rc={rc} layer {li} {nm}")

    def _ffn_fp4_block(self, li: int, x_bf16: torch.Tensor, s: int):
        """FP4 W4A4 FFN: fused bf16 res+rms -> fp4, up GEMM, relu2 -> fp4, down GEMM."""
        NG = self.NG
        self.fp4.cosmos3_edge_res_rms_fp4_sfa_bf16(
            self.Hb.data_ptr(), x_bf16.data_ptr(), self.Wn[(li, "post_ln")].data_ptr(),
            self.a4h.data_ptr(), self.sfa_h.data_ptr(), NG, HID, EPS, s)
        if self.fp4_relu2_epilogue:
            w_packed, w_sfb = self.Wf4[(li, "up")]
            rc = self.fp4.cosmos3_edge_fp4_gemm_relu2_fp4out(
                self.a4h.data_ptr(), self.sfa_h.data_ptr(),
                w_packed.data_ptr(), w_sfb.data_ptr(),
                self.a4f.data_ptr(), self.sfa_f.data_ptr(), NG, FF, HID, s)
            if rc != 0:
                raise RuntimeError(f"fused FP4 up+relu2 failed rc={rc} layer {li}")
        else:
            self._fp4_gemm(self.a4h, self.sfa_h, li, "up", self.up16, NG, FF, HID, s)
            self.fp4.cosmos3_edge_relu2_fp4_sfa_fp16(
                self.up16.data_ptr(), self.a4f.data_ptr(), self.sfa_f.data_ptr(), NG, FF, s)
        self._fp4_gemm(self.a4f, self.sfa_f, li, "down", self.dn16, NG, HID, FF, s)
        self.dn.copy_(self.dn16)

    def _quant(self, a_bf16: torch.Tensor, a_f8: torch.Tensor, needed: bool, li: int, site: int):
        """Dynamic quantization pass; accumulates the per-site scale ceiling."""
        if needed and self.quant == "fp8":
            fvk.quantize_fp8_device(a_bf16.data_ptr(), a_f8.data_ptr(), self.asc.data_ptr(), a_bf16.numel(), self._s())
            fvk.fp8_accumulate_scale_max(self.asc.data_ptr(), self._site_ptr(li, site), self._s())

    def _layer_needs_fp8(self, li: int, names: tuple[str, ...]) -> bool:
        return any((li, nm) in self.Wf8 for nm in names)

    @property
    def _fused_static(self) -> bool:
        return self.quant == "fp8" and self.calibrated and not self.bf16_projs

    # ---- per-step forward: latent(f32 flat) -> velocity(bf16 flat) ----
    def forward_step(self, step: int, latent: torch.Tensor) -> torch.Tensor:
        s = self._s()
        NG, NJ, NU = self.NG, self.NJ, self.NU

        fvk.cosmos3_edge_copy_action_tail_f32_to_bf16(
            latent.data_ptr(), self.action_input.data_ptr(), latent.numel(), NA * AD, s)
        self.gemm.bf16_nn(
            self.action_input.data_ptr(), self.action_in_w.data_ptr(),
            self.action_encoded.data_ptr(), NA, HID, AD, s)
        fvk.cosmos3_edge_add_action_bias_timestep_bf16(
            self.action_encoded.data_ptr(), self.action_static_bias.data_ptr(),
            self.t_emb[step : step + 1].data_ptr(), NA, HID, s)
        self.Hb.copy_(self.gen_init)
        fvk.cosmos3_edge_scatter_rows_bf16(
            self.action_encoded.data_ptr(), self.Hb.data_ptr(), self.action_indexes.data_ptr(), NA, HID, s)

        fused = self._fused_static
        if fused:
            fvk.rms_norm(self.Hb.data_ptr(), self.Wn[(0, "in_ln")].data_ptr(), self.nrm.data_ptr(), NG, HID, EPS, s)
            fvk.quantize_fp8_static(self.nrm.data_ptr(), self.a8h.data_ptr(), self._site_ptr(0, 0), NG * HID, s)
        else:
            fvk.rms_norm(self.Hb.data_ptr(), self.Wn[(0, "in_ln")].data_ptr(), self.nrm.data_ptr(), NG, HID, EPS, s)
        slim_done = False
        for li in range(NL):
            qkv_scale = self._site_ptr(li, 0) if fused else None
            if not fused:
                self._quant(
                    self.nrm, self.a8h,
                    self.qkv_fused or self._layer_needs_fp8(li, ("q", "k", "v")), li, 0)
            if self.qkv_fused:
                wide = (H + 2 * KVH) * D
                self._proj(self.nrm, self.a8h, li, "qkv", self.QKVt, wide, qkv_scale)
                fvk.cosmos3_edge_qk_norm_rope_strided_bf16(
                    self.QKVt.data_ptr(), self.QKVt.data_ptr() + 2 * H * D,
                    self.Wn[(li, "q_norm")].data_ptr(), self.Wn[(li, "k_norm")].data_ptr(),
                    self.cos_f.data_ptr(), self.sin_f.data_ptr(),
                    self.Qb.data_ptr(), self.Kj[li, 0, NU:NJ].data_ptr(),
                    NG, H, KVH, wide, wide, EPS, s)
                self.Vj[li, 0, NU:NJ].view(NG, KVH * D).copy_(self.QKVt[:, (H + KVH) * D :])
            else:
                self._proj(self.nrm, self.a8h, li, "q", self.Qt, H * D, qkv_scale)
                self._proj(self.nrm, self.a8h, li, "k", self.Kt, KVH * D, qkv_scale)
                self._proj(self.nrm, self.a8h, li, "v", self.Vj[li, 0, NU:NJ].view(NG, KVH * D), KVH * D, qkv_scale)
                fvk.cosmos3_edge_qk_norm_rope_bf16(
                    self.Qt.data_ptr(), self.Kt.data_ptr(),
                    self.Wn[(li, "q_norm")].data_ptr(), self.Wn[(li, "k_norm")].data_ptr(),
                    self.cos_f.data_ptr(), self.sin_f.data_ptr(),
                    self.Qb.data_ptr(), self.Kj[li, 0, NU:NJ].data_ptr(),
                    NG, H, KVH, D, D, EPS, s)
            if fused and self.slim_last and li == NL - 1:
                # Last layer: only the NA action rows are consumed by the head.
                # K/V (computed above) still cover all rows; Q/attention/o/FFN
                # shrink to M=NA. Site scales calibrated at full M are upper
                # bounds for the row subset.
                fvk.cosmos3_edge_gather_rows_bf16(
                    self.Qb.data_ptr(), self.Qs.data_ptr(), self.action_indexes.data_ptr(), NA, HID, s)
                self._fa4_fwd(
                    self.Qs.view(1, NA, H, D), self.Kj[li], self.Vj[li],
                    softmax_scale=D ** -0.5, causal=False, pack_gqa=True,
                    out=self.attn_s, lse=self.lse_s)
                attn_s2d = self.attn_s.view(NA, HID)
                fvk.quantize_fp8_static(
                    attn_s2d.data_ptr(), self.a8h.data_ptr(), self._site_ptr(li, 1), NA * HID, s)
                self._proj(attn_s2d, self.a8h, li, "o", self.ob, HID, self._site_ptr(li, 1))
                fvk.cosmos3_edge_gather_rows_bf16(
                    self.Hb.data_ptr(), self.action_hidden.data_ptr(), self.action_indexes.data_ptr(), NA, HID, s)
                if self.ffn_fp4:
                    self.fp4.cosmos3_edge_res_rms_fp4_sfa_bf16(
                        self.action_hidden.data_ptr(), self.ob.data_ptr(),
                        self.Wn[(li, "post_ln")].data_ptr(),
                        self.a4h.data_ptr(), self.sfa_h.data_ptr(), NA, HID, EPS, s)
                    if self.fp4_relu2_epilogue:
                        w_packed, w_sfb = self.Wf4[(li, "up")]
                        rc = self.fp4.cosmos3_edge_fp4_gemm_relu2_fp4out(
                            self.a4h.data_ptr(), self.sfa_h.data_ptr(),
                            w_packed.data_ptr(), w_sfb.data_ptr(),
                            self.a4f.data_ptr(), self.sfa_f.data_ptr(), NA, FF, HID, s)
                        if rc != 0:
                            raise RuntimeError(f"fused FP4 up+relu2 failed rc={rc} layer {li}")
                    else:
                        self._fp4_gemm(self.a4h, self.sfa_h, li, "up", self.up16, NA, FF, HID, s)
                        self.fp4.cosmos3_edge_relu2_fp4_sfa_fp16(
                            self.up16.data_ptr(), self.a4f.data_ptr(), self.sfa_f.data_ptr(), NA, FF, s)
                    self._fp4_gemm(self.a4f, self.sfa_f, li, "down", self.dn16, NA, HID, FF, s)
                    self.dn[:NA].copy_(self.dn16[:NA])
                else:
                    fvk.residual_add_rms_norm_fp8(
                        self.action_hidden.data_ptr(), self.ob.data_ptr(),
                        self.Wn[(li, "post_ln")].data_ptr(),
                        self.a8h.data_ptr(), NA, HID, EPS, self._site_ptr(li, 2), s)
                    self._proj(self.nrm2[:NA], self.a8h, li, "up", self.up, FF, self._site_ptr(li, 2))
                    fvk.cosmos3_edge_relu2_to_fp8_static_bf16(
                        self.up.data_ptr(), self.a8f.data_ptr(), self._site_ptr(li, 3), NA * FF, s)
                    self._proj(self.up[:NA], self.a8f, li, "down", self.dn, HID, self._site_ptr(li, 3))
                fvk.residual_add(self.action_hidden.data_ptr(), self.dn.data_ptr(), NA * HID, s)
                slim_done = True
                break
            self._fa4_fwd(
                self.Qb.view(1, NG, H, D), self.Kj[li], self.Vj[li],
                softmax_scale=D ** -0.5, causal=False, pack_gqa=True,
                out=self.attn, lse=self.lse)
            attn2d = self.attn.view(NG, HID)
            if fused:
                fvk.quantize_fp8_static(attn2d.data_ptr(), self.a8h.data_ptr(), self._site_ptr(li, 1), NG * HID, s)
                self._proj(attn2d, self.a8h, li, "o", self.ob, HID, self._site_ptr(li, 1))
                if self.ffn_fp4:
                    self._ffn_fp4_block(li, self.ob, s)
                else:
                    fvk.residual_add_rms_norm_fp8(
                        self.Hb.data_ptr(), self.ob.data_ptr(), self.Wn[(li, "post_ln")].data_ptr(),
                        self.a8h.data_ptr(), NG, HID, EPS, self._site_ptr(li, 2), s)
                    self._proj(self.nrm2, self.a8h, li, "up", self.up, FF, self._site_ptr(li, 2))
                    fvk.cosmos3_edge_relu2_to_fp8_static_bf16(
                        self.up.data_ptr(), self.a8f.data_ptr(), self._site_ptr(li, 3), NG * FF, s)
                    self._proj(self.up, self.a8f, li, "down", self.dn, HID, self._site_ptr(li, 3))
                if li + 1 < NL:
                    fvk.residual_add_rms_norm_fp8(
                        self.Hb.data_ptr(), self.dn.data_ptr(), self.Wn[(li + 1, "in_ln")].data_ptr(),
                        self.a8h.data_ptr(), NG, HID, EPS, self._site_ptr(li + 1, 0), s)
                else:
                    fvk.residual_add(self.Hb.data_ptr(), self.dn.data_ptr(), NG * HID, s)
            else:
                self._quant(attn2d, self.a8h, self._layer_needs_fp8(li, ("o",)), li, 1)
                self._proj(attn2d, self.a8h, li, "o", self.ob, HID)
                if self.ffn_fp4:
                    self._ffn_fp4_block(li, self.ob, s)
                else:
                    fvk.residual_add_rms_norm(
                        self.Hb.data_ptr(), self.ob.data_ptr(), self.Wn[(li, "post_ln")].data_ptr(),
                        self.nrm2.data_ptr(), NG, HID, EPS, s)
                    self._quant(self.nrm2, self.a8h, self._layer_needs_fp8(li, ("up",)), li, 2)
                    self._proj(self.nrm2, self.a8h, li, "up", self.up, FF)
                    fvk.relu2_inplace_bf16(self.up.data_ptr(), NG * FF, s)
                    self._quant(self.up, self.a8f, self._layer_needs_fp8(li, ("down",)), li, 3)
                    self._proj(self.up, self.a8f, li, "down", self.dn, HID)
                if li + 1 < NL:
                    fvk.residual_add_rms_norm(
                        self.Hb.data_ptr(), self.dn.data_ptr(), self.Wn[(li + 1, "in_ln")].data_ptr(),
                        self.nrm.data_ptr(), NG, HID, EPS, s)
                else:
                    fvk.residual_add(self.Hb.data_ptr(), self.dn.data_ptr(), NG * HID, s)

        if not slim_done:
            fvk.cosmos3_edge_gather_rows_bf16(
                self.Hb.data_ptr(), self.action_hidden.data_ptr(), self.action_indexes.data_ptr(), NA, HID, s)
        fvk.rms_norm(
            self.action_hidden.data_ptr(), self.norm_g.data_ptr(), self.action_norm.data_ptr(), NA, HID, EPS, s)
        self.gemm.bf16_nn(
            self.action_norm.data_ptr(), self.action_out_w.data_ptr(), self.action_out.data_ptr(), NA, AD, HID, s)
        fvk.cosmos3_edge_add_bias_zero_action_tail_bf16(
            self.action_out.data_ptr(), self.action_out_bias.data_ptr(), NA, AD, self.raw_action_dim, s)
        fvk.cosmos3_edge_fill_flat_velocity_bf16(
            self.action_out.data_ptr(), self.velocity.data_ptr(), self.velocity.numel(), NA * AD, s)
        return self.velocity

    def set_teacache(self, compute_steps: tuple[int, ...] | list[int] | None) -> None:
        """Fixed compute-step schedule: skipped steps reuse the last computed
        velocity (the static velocity buffer) while UniPC still advances every
        step. Step 0 must always compute. Call before ``capture``."""
        if self.graph is not None:
            raise RuntimeError("set_teacache must be called before graph capture")
        if compute_steps is None:
            self.compute_steps = None
            return
        steps = sorted(set(int(v) for v in compute_steps))
        if not steps or steps[0] != 0 or steps[-1] >= self.num_steps:
            raise ValueError(f"invalid TeaCache compute schedule: {steps}")
        self.compute_steps = frozenset(steps)

    # ---- whole-denoise loop: forward + native UniPC per step ----
    def run_loop(self) -> torch.Tensor:
        # UniPC state buffers are allocated once; their contents are fully
        # rewritten in step order on every run (step 0 reads no history).
        if self.unipc.prev_m1 is None:
            self.unipc.reset(self.latent)
        for step in range(self.num_steps):
            if self.compute_steps is None or step in self.compute_steps:
                velocity = self.forward_step(step, self.latent)
            else:
                velocity = self.velocity
            self.unipc.step(self.latent, velocity, step)
        return self.latent

    def calibrate(self, noise_flat_f32: torch.Tensor, *, margin: float = 1.25) -> None:
        """One dynamic-quant denoise pass to record per-site scale ceilings.

        The recorded ceilings (times ``margin``) become the static scales for
        the fused quant chain; FP8 E4M3 saturates gracefully on the rare
        excursion past the ceiling.
        """
        if self.quant != "fp8" or self.bf16_projs:
            return
        self.calibrated = False
        self.latent.copy_(noise_flat_f32.to(device=DEV, dtype=torch.float32))
        self.run_loop()
        torch.cuda.synchronize()
        self.site_scale.mul_(margin)
        self.calibrated = True

    def capture(self, warmup_noise: torch.Tensor | None = None) -> None:
        if warmup_noise is not None:
            self.latent.copy_(warmup_noise.to(device=DEV, dtype=torch.float32))
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(2):
                self.run_loop()
        torch.cuda.current_stream().wait_stream(st)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.run_loop()

    def denoise(self, noise_flat_f32: torch.Tensor) -> torch.Tensor:
        """Full 30-step denoise from flat noise; returns the final flat latent."""
        self.latent.copy_(noise_flat_f32.to(device=DEV, dtype=torch.float32))
        if self.graph is not None:
            self.graph.replay()
        else:
            self.run_loop()
        return self.latent
