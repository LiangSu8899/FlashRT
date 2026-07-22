#!/usr/bin/env python3
"""Cosmos3-Edge Reasoner chat inference on Thor (batch 1, greedy).

The Reasoner is the checkpoint's understanding view (top-level
``model.safetensors.index.json``): the und text tower (28L / 2048 / GQA 16:8 /
head_dim 128 / non-gated relu^2 MLP, no qk-norm), a SigLIP2 vision encoder
(27L / 1152 / 16 heads / head_dim 72, per-media attention windows, no RoPE,
no deepstack), a PatchMerger projector, and ``embed_tokens`` / ``norm`` /
``lm_head`` (vocab 131072). Special tokens: image=19, video=18.

Two execution paths share the weights:
  - vision + prefill: exact torch replica of the official
    ``nemotron_3_dense_vl`` modules (parity-first; prefill is one pass).
  - decode: static-buffer BF16 or NVFP4 per-token loop over the KV cache,
    captured in a single CUDA graph.
"""
from __future__ import annotations

import json
import math
import operator
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from safetensors import safe_open

DEV = "cuda"
BF = torch.bfloat16

# Text tower (und pathway).
NL, HID, NH, NKV, HD, FF = 28, 2048, 16, 8, 128, 9216
EPS = 1e-5
ROPE_THETA = 1e8
MROPE_SECTION = (24, 20, 20)
VOCAB = 131072
IMAGE_TOKEN_ID = 19
VIDEO_TOKEN_ID = 18
VISION_START_ID = 20  # verified against tokenizer at load time
ATTN_SPLITS = 24

_REASONER_COMMON_SYMBOLS = (
    "rms_norm",
    "relu2_inplace_bf16",
)
_REASONER_BF16_SYMBOLS = (
    "ht_gemv_bf16_m1_w4",
    "cosmos3_reasoner_rope_kv_bf16",
    "cosmos3_reasoner_decode_attn_bf16",
)
_REASONER_FP4_SYMBOLS = (
    "cosmos3_reasoner_gemv_w4a16_bf16",
    "cosmos3_reasoner_rope_kv_fp8_bf16",
    "cosmos3_reasoner_decode_attn_fp8kv_bf16",
)


def _require_reasoner_kernels(fvk: object, quant: str) -> None:
    required = _REASONER_COMMON_SYMBOLS + (
        _REASONER_FP4_SYMBOLS if quant == "fp4" else _REASONER_BF16_SYMBOLS
    )
    missing = [name for name in required if not hasattr(fvk, name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "FlashRT was built without the required Cosmos3 Reasoner kernels "
            f"for quant={quant}: {joined}. Rebuild with "
            "-DGPU_ARCH=110 -DFLASHRT_ENABLE_COSMOS3_REASONER=ON."
        )

# Vision tower (SigLIP2).
V_NL, V_HID, V_NH, V_HD, V_FF = 27, 1152, 16, 72, 4304
V_EPS = 1e-6
MERGE = 2
PROJ_IN = V_HID * MERGE * MERGE  # 4608
PROJ_MID = 11520


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (w.float() * xf).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class CosmosReasonerThor:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        max_seq: int = 6144,
        max_new_tokens: int = 256,
        quant: str = "bf16",
        use_graph: bool = True,
    ):
        if quant not in ("bf16", "fp4"):
            raise ValueError(f"unsupported quant: {quant}")
        if isinstance(max_seq, bool) or not isinstance(max_seq, int) or max_seq <= 0:
            raise ValueError(f"max_seq must be a positive integer, got {max_seq!r}")
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or
                not 1 <= max_new_tokens <= max_seq):
            raise ValueError(
                "max_new_tokens must be an integer in "
                f"[1, max_seq={max_seq}], got {max_new_tokens!r}"
            )
        try:
            import flash_rt.flash_rt_kernels as fvk
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Cosmos3 Reasoner kernels are unavailable. Rebuild with "
                "-DGPU_ARCH=110 -DFLASHRT_ENABLE_COSMOS3_REASONER=ON."
            ) from exc
        _require_reasoner_kernels(fvk, quant)

        self.quant = quant
        self.use_graph = bool(use_graph)
        self.fvk = fvk
        self.ckpt = Path(checkpoint)
        self.max_seq = max_seq
        self.max_new = max_new_tokens
        idx = json.loads((self.ckpt / "model.safetensors.index.json").read_text())
        self._wmap = idx["weight_map"]
        self._shards: dict[str, object] = {}
        self.w: dict[str, torch.Tensor] = {}
        for key in self._wmap:
            self.w[key] = self._load(key)

        cfg = json.loads((self.ckpt / "config.json").read_text())
        self.eos_token_id = int(cfg["text_config"]["eos_token_id"])

        # Text rope: interleaved mRoPE inv_freq (identical to official).
        inv = 1.0 / (ROPE_THETA ** (torch.arange(0, HD, 2, dtype=torch.float32, device=DEV) / HD))
        self.inv_freq = inv  # [64]

        # KV cache [NL, max_seq, NKV, HD] bf16.
        self.k_cache = torch.zeros(NL, self.max_seq, NKV, HD, device=DEV, dtype=BF)
        self.v_cache = torch.zeros(NL, self.max_seq, NKV, HD, device=DEV, dtype=BF)
        self.seq_len = 0
        self.rope_delta = 0

        if self.quant == "fp4":
            self._init_fp4_decode()
        else:
            self._init_bf16_decode()

    # ---------------- NVFP4 W4A4 decode path ----------------
    @staticmethod
    def _swz_bytes(rows: int, cols: int) -> int:
        return ((rows + 127) // 128) * ((cols // 16 + 3) // 4) * 512

    def _init_fp4_decode(self) -> None:
        levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEV)

        def qw(w: torch.Tensor):
            """e2m1 codes packed 2/byte [N, K/2] + per-16 bf16 scales [N, K/16]."""
            n, k = w.shape
            packed = torch.empty(n, k // 2, dtype=torch.uint8, device=DEV)
            scales = torch.empty(n, k // 16, dtype=BF, device=DEV)
            chunk = max(1, (1 << 26) // k)
            for r0 in range(0, n, chunk):
                wb = w[r0 : r0 + chunk].float().view(-1, k // 16, 16)
                sc = (wb.abs().amax(-1) / 6.0).clamp_(min=1e-8)
                q = wb / sc[..., None]
                idx = (q.abs()[..., None] - levels).abs().argmin(-1)
                code = (idx + (q < 0) * 8).to(torch.uint8).view(-1, k)
                packed[r0 : r0 + chunk] = code[:, 0::2] | (code[:, 1::2] << 4)
                scales[r0 : r0 + chunk] = sc.to(BF)
            return packed, scales

        self.W4 = {}
        for li in range(NL):
            for nm, key in (("o", "self_attn.to_out.weight"),
                            ("up", "mlp.up_proj.weight"), ("down", "mlp.down_proj.weight")):
                self.W4[(li, nm)] = qw(self.w[f"layers.{li}.{key}"])
            # q/k/v rows concatenated: one N=(NH+2*NKV)*HD GEMV per layer.
            parts = [qw(self.w[f"layers.{li}.self_attn.to_{nm}.weight"]) for nm in ("q", "k", "v")]
            self.W4[(li, "qkv")] = (
                torch.cat([p[0] for p in parts], dim=0).contiguous(),
                torch.cat([p[1] for p in parts], dim=0).contiguous(),
            )
            del parts
        self.W4["lm_head"] = qw(self.w["lm_head.weight"])
        torch.cuda.synchronize()

        # Static decode buffers.
        self.d_tok = torch.zeros(1, dtype=torch.long, device=DEV)
        self.d_pos = torch.zeros(1, dtype=torch.long, device=DEV)
        self.d_slot = torch.zeros(1, dtype=torch.long, device=DEV)
        self.d_x = torch.zeros(1, HID, device=DEV, dtype=BF)
        self.d_nrm = torch.zeros(1, HID, device=DEV, dtype=BF)
        self.d_ap_h = torch.empty(1, HID // 2, dtype=torch.uint8, device=DEV)
        self.d_sf_h = torch.zeros(self._swz_bytes(1, HID), dtype=torch.uint8, device=DEV)
        self.d_ap_f = torch.empty(1, FF // 2, dtype=torch.uint8, device=DEV)
        self.d_sf_f = torch.zeros(self._swz_bytes(1, FF), dtype=torch.uint8, device=DEV)
        self.d_qkv = torch.empty(1, (NH + 2 * NKV) * HD, device=DEV, dtype=BF)
        self.d_q = self.d_qkv[:, : NH * HD]
        self.d_k = self.d_qkv[:, NH * HD : (NH + NKV) * HD]
        self.d_v = self.d_qkv[:, (NH + NKV) * HD :]
        self.d_attn = torch.empty(1, HID, device=DEV, dtype=BF)
        self.d_o = torch.empty(1, HID, device=DEV, dtype=BF)
        self.d_up = torch.empty(1, FF, device=DEV, dtype=BF)
        self.d_dn = torch.empty(1, HID, device=DEV, dtype=BF)
        self.d_logits = torch.empty(1, VOCAB, device=DEV, dtype=BF)
        self.d_qr = torch.empty(NH, HD, device=DEV, dtype=BF)
        self.d_len = torch.zeros(1, dtype=torch.int32, device=DEV)
        self.d_step = torch.zeros(1, dtype=torch.long, device=DEV)
        self.d_out_tokens = torch.zeros(self.max_new + 1, dtype=torch.long, device=DEV)
        self.d_max_val = torch.empty(1, device=DEV, dtype=BF)
        self.d_pacc = torch.empty(NH, ATTN_SPLITS, HD, dtype=torch.float32, device=DEV)
        self.d_pml = torch.empty(NH, ATTN_SPLITS, 2, dtype=torch.float32, device=DEV)
        # e4m3 KV cache (uint8 bit-pattern storage): halves decode KV traffic.
        self.k8_cache = torch.zeros(NL, self.max_seq, NKV, HD, device=DEV, dtype=torch.uint8)
        self.v8_cache = torch.zeros(NL, self.max_seq, NKV, HD, device=DEV, dtype=torch.uint8)
        # Decode rope table: scalar position -> standard interleaved rope.
        p = torch.arange(self.max_seq, device=DEV, dtype=torch.float32)
        f = p[:, None] * self.inv_freq[None, :]
        emb = torch.cat((f, f), dim=-1)
        self.d_cos_table = emb.cos().to(BF)
        self.d_sin_table = emb.sin().to(BF)
        # Stable transposed KV views for SDPA: [NL][1, NKV, max_seq, HD].
        self._kT = [self.k_cache[li].permute(1, 0, 2).unsqueeze(0) for li in range(NL)]
        self._vT = [self.v_cache[li].permute(1, 0, 2).unsqueeze(0) for li in range(NL)]
        self._graph = None

    # ---------------- BF16 W16A16 decode path ----------------
    def _init_bf16_decode(self) -> None:
        # A single wide QKV GEMV is more efficient than three short-M launches.
        self.W16_qkv = []
        for li in range(NL):
            self.W16_qkv.append(torch.cat([
                self.w[f"layers.{li}.self_attn.to_q.weight"],
                self.w[f"layers.{li}.self_attn.to_k.weight"],
                self.w[f"layers.{li}.self_attn.to_v.weight"],
            ], dim=0).contiguous())

        self.d_tok = torch.zeros(1, dtype=torch.long, device=DEV)
        self.d_pos = torch.zeros(1, dtype=torch.long, device=DEV)
        self.d_slot = torch.zeros(1, dtype=torch.long, device=DEV)
        self.d_x = torch.zeros(1, HID, device=DEV, dtype=BF)
        self.d_nrm = torch.zeros(1, HID, device=DEV, dtype=BF)
        self.d_qkv = torch.empty(1, (NH + 2 * NKV) * HD, device=DEV, dtype=BF)
        self.d_q = self.d_qkv[:, : NH * HD]
        self.d_k = self.d_qkv[:, NH * HD : (NH + NKV) * HD]
        self.d_v = self.d_qkv[:, (NH + NKV) * HD :]
        self.d_attn = torch.empty(1, HID, device=DEV, dtype=BF)
        self.d_o = torch.empty(1, HID, device=DEV, dtype=BF)
        self.d_up = torch.empty(1, FF, device=DEV, dtype=BF)
        self.d_dn = torch.empty(1, HID, device=DEV, dtype=BF)
        self.d_logits = torch.empty(1, VOCAB, device=DEV, dtype=BF)
        self.d_qr = torch.empty(NH, HD, device=DEV, dtype=BF)
        self.d_len = torch.zeros(1, dtype=torch.int32, device=DEV)
        self.d_step = torch.zeros(1, dtype=torch.long, device=DEV)
        self.d_out_tokens = torch.zeros(self.max_new + 1, dtype=torch.long, device=DEV)
        self.d_max_val = torch.empty(1, device=DEV, dtype=BF)
        self.d_pacc = torch.empty(NH, ATTN_SPLITS, HD, dtype=torch.float32, device=DEV)
        self.d_pml = torch.empty(NH, ATTN_SPLITS, 2, dtype=torch.float32, device=DEV)
        p = torch.arange(self.max_seq, device=DEV, dtype=torch.float32)
        f = p[:, None] * self.inv_freq[None, :]
        emb = torch.cat((f, f), dim=-1)
        self.d_cos_table = emb.cos().to(BF)
        self.d_sin_table = emb.sin().to(BF)
        self._graph = None

    def _gemv_bf16(self, x: torch.Tensor, weight: torch.Tensor,
                    out: torch.Tensor, n: int, k: int) -> None:
        self.fvk.ht_gemv_bf16_m1_w4(
            x.data_ptr(), weight.data_ptr(), out.data_ptr(), 1, n, k, 1.0,
            torch.cuda.current_stream().cuda_stream)

    def _decode_step_bf16(self) -> None:
        fvk = self.fvk
        s = torch.cuda.current_stream().cuda_stream
        torch.index_select(self.w["embed_tokens.weight"], 0, self.d_tok, out=self.d_x)
        for li in range(NL):
            fvk.rms_norm(self.d_x.data_ptr(), self.w[f"layers.{li}.input_layernorm.weight"].data_ptr(),
                         self.d_nrm.data_ptr(), 1, HID, EPS, s)
            self._gemv_bf16(self.d_nrm, self.W16_qkv[li], self.d_qkv,
                            (NH + 2 * NKV) * HD, HID)
            fvk.cosmos3_reasoner_rope_kv_bf16(
                self.d_q.data_ptr(), self.d_k.data_ptr(), self.d_v.data_ptr(),
                self.d_cos_table.data_ptr(), self.d_sin_table.data_ptr(),
                self.d_pos.data_ptr(), self.d_slot.data_ptr(),
                self.d_qr.data_ptr(), self.k_cache[li].data_ptr(), self.v_cache[li].data_ptr(),
                NH, NKV, s)
            fvk.cosmos3_reasoner_decode_attn_bf16(
                self.d_qr.data_ptr(), self.k_cache[li].data_ptr(), self.v_cache[li].data_ptr(),
                self.d_len.data_ptr(), self.d_attn.data_ptr(),
                self.d_pacc.data_ptr(), self.d_pml.data_ptr(), NH, NKV, HD ** -0.5, s)
            self._gemv_bf16(
                self.d_attn, self.w[f"layers.{li}.self_attn.to_out.weight"],
                self.d_o, HID, HID)
            self.d_x.add_(self.d_o)
            fvk.rms_norm(self.d_x.data_ptr(), self.w[f"layers.{li}.post_attention_layernorm.weight"].data_ptr(),
                         self.d_nrm.data_ptr(), 1, HID, EPS, s)
            self._gemv_bf16(
                self.d_nrm, self.w[f"layers.{li}.mlp.up_proj.weight"],
                self.d_up, FF, HID)
            fvk.relu2_inplace_bf16(self.d_up.data_ptr(), FF, s)
            self._gemv_bf16(
                self.d_up, self.w[f"layers.{li}.mlp.down_proj.weight"],
                self.d_dn, HID, FF)
            self.d_x.add_(self.d_dn)
        fvk.rms_norm(self.d_x.data_ptr(), self.w["norm.weight"].data_ptr(),
                     self.d_nrm.data_ptr(), 1, HID, EPS, s)
        self._gemv_bf16(self.d_nrm, self.w["lm_head.weight"],
                        self.d_logits, VOCAB, HID)
        torch.max(self.d_logits.view(VOCAB), 0, out=(self.d_max_val[0], self.d_tok[0]))
        self.d_out_tokens.index_copy_(0, self.d_step, self.d_tok)
        self.d_pos.add_(1)
        self.d_slot.add_(1)
        self.d_len.add_(1)
        self.d_step.add_(1)

    def _gemv(self, x_bf16, key, out, n, k):
        wp, ws = self.W4[key]
        self.fvk.cosmos3_reasoner_gemv_w4a16_bf16(
            wp.data_ptr(), ws.data_ptr(), x_bf16.data_ptr(), out.data_ptr(), n, k,
            torch.cuda.current_stream().cuda_stream)

    def _decode_step_fp4(self) -> None:
        fvk = self.fvk
        s = torch.cuda.current_stream().cuda_stream
        torch.index_select(self.w["embed_tokens.weight"], 0, self.d_tok, out=self.d_x)
        for li in range(NL):
            fvk.rms_norm(self.d_x.data_ptr(), self.w[f"layers.{li}.input_layernorm.weight"].data_ptr(),
                         self.d_nrm.data_ptr(), 1, HID, EPS, s)
            self._gemv(self.d_nrm, (li, "qkv"), self.d_qkv, (NH + 2 * NKV) * HD, HID)
            fvk.cosmos3_reasoner_rope_kv_fp8_bf16(
                self.d_q.data_ptr(), self.d_k.data_ptr(), self.d_v.data_ptr(),
                self.d_cos_table.data_ptr(), self.d_sin_table.data_ptr(),
                self.d_pos.data_ptr(), self.d_slot.data_ptr(),
                self.d_qr.data_ptr(), self.k8_cache[li].data_ptr(), self.v8_cache[li].data_ptr(),
                NH, NKV, s)
            fvk.cosmos3_reasoner_decode_attn_fp8kv_bf16(
                self.d_qr.data_ptr(), self.k8_cache[li].data_ptr(), self.v8_cache[li].data_ptr(),
                self.d_len.data_ptr(), self.d_attn.data_ptr(),
                self.d_pacc.data_ptr(), self.d_pml.data_ptr(), NH, NKV, HD ** -0.5, s)
            self._gemv(self.d_attn, (li, "o"), self.d_o, HID, HID)
            self.d_x.add_(self.d_o)
            fvk.rms_norm(self.d_x.data_ptr(), self.w[f"layers.{li}.post_attention_layernorm.weight"].data_ptr(),
                         self.d_nrm.data_ptr(), 1, HID, EPS, s)
            self._gemv(self.d_nrm, (li, "up"), self.d_up, FF, HID)
            fvk.relu2_inplace_bf16(self.d_up.data_ptr(), FF, s)
            self._gemv(self.d_up, (li, "down"), self.d_dn, HID, FF)
            self.d_x.add_(self.d_dn)
        fvk.rms_norm(self.d_x.data_ptr(), self.w["norm.weight"].data_ptr(), self.d_nrm.data_ptr(), 1, HID, EPS, s)
        self._gemv(self.d_nrm, "lm_head", self.d_logits, VOCAB, HID)
        # In-graph greedy step: sample, record, and advance the loop state so
        # the host loop is nothing but graph replays.
        torch.max(self.d_logits.view(VOCAB), 0, out=(self.d_max_val[0], self.d_tok[0]))
        self.d_out_tokens.index_copy_(0, self.d_step, self.d_tok)
        self.d_pos.add_(1)
        self.d_slot.add_(1)
        self.d_len.add_(1)
        self.d_step.add_(1)

    def _ensure_graph(self) -> None:
        if self._graph is not None or not self.use_graph:
            return
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        step = self._decode_step_fp4 if self.quant == "fp4" else self._decode_step_bf16
        with torch.cuda.stream(st):
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(st)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            step()

    # ---------------- weights ----------------
    def _load(self, key: str) -> torch.Tensor:
        shard = self.ckpt / self._wmap[key]
        sp = str(shard)
        if sp not in self._shards:
            self._shards[sp] = safe_open(sp, framework="pt", device="cpu")
        t = self._shards[sp].get_tensor(key)
        return t.to(device=DEV, dtype=BF if t.dtype in (torch.float32, torch.bfloat16, torch.float16) else t.dtype)

    # ---------------- vision (exact torch replica) ----------------
    def _vision_pos_embed(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pe = self.w["model.visual.embeddings.position_embedding.weight"]  # [256, 1152]
        side = int(math.isqrt(pe.shape[0]))
        pe4 = pe.reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0).float()
        outs = []
        for t, h, w_ in grid_thw.tolist():
            r = F.interpolate(pe4, size=(h, w_), mode="bilinear", align_corners=False, antialias=True)
            r = r.reshape(pe.shape[1], -1).transpose(0, 1).to(BF)
            outs.append(r.repeat(t, 1))
        return torch.cat(outs, dim=0)

    def encode_vision(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """pixel_values [N_patches, 768] -> projected [N_merged, 2048]."""
        vw = lambda k: self.w[f"model.visual.{k}"]
        x = pixel_values.to(device=DEV, dtype=BF)
        x = F.linear(x, vw("embeddings.patch_embedding.weight"), vw("embeddings.patch_embedding.bias"))
        x = x + self._vision_pos_embed(grid_thw)

        # Per-media attention windows (video frames share one window per t*h*w run).
        seg_lens = (grid_thw[:, 1] * grid_thw[:, 2]).repeat_interleave(grid_thw[:, 0]).tolist()
        for li in range(V_NL):
            lw = lambda k: self.w[f"model.visual.encoder.layers.{li}.{k}"]
            res = x
            h = F.layer_norm(x.float(), (V_HID,), lw("layer_norm1.weight").float(), lw("layer_norm1.bias").float(), V_EPS).to(BF)
            q = F.linear(h, lw("self_attn.q_proj.weight"), lw("self_attn.q_proj.bias"))
            k = F.linear(h, lw("self_attn.k_proj.weight"), lw("self_attn.k_proj.bias"))
            v = F.linear(h, lw("self_attn.v_proj.weight"), lw("self_attn.v_proj.bias"))
            outs = []
            off = 0
            for sl in seg_lens:
                qs = q[off : off + sl].view(sl, V_NH, V_HD).transpose(0, 1)
                ks = k[off : off + sl].view(sl, V_NH, V_HD).transpose(0, 1)
                vs = v[off : off + sl].view(sl, V_NH, V_HD).transpose(0, 1)
                o = F.scaled_dot_product_attention(qs, ks, vs)
                outs.append(o.transpose(0, 1).reshape(sl, V_HID))
                off += sl
            o = torch.cat(outs, dim=0)
            o = F.linear(o, lw("self_attn.out_proj.weight"), lw("self_attn.out_proj.bias"))
            x = res + o
            res = x
            h = F.layer_norm(x.float(), (V_HID,), lw("layer_norm2.weight").float(), lw("layer_norm2.bias").float(), V_EPS).to(BF)
            h = F.linear(h, lw("mlp.fc1.weight"), lw("mlp.fc1.bias"))
            h = F.gelu(h, approximate="tanh")
            h = F.linear(h, lw("mlp.fc2.weight"), lw("mlp.fc2.bias"))
            x = res + h
        x = F.layer_norm(x.float(), (V_HID,), vw("post_layernorm.weight").float(), vw("post_layernorm.bias").float(), V_EPS).to(BF)

        # 2x2 spatial merge per media ('t (h h1) (w w1) c -> t h w (h1 w1 c)').
        merged = []
        off = 0
        for t, h_, w_ in grid_thw.tolist():
            n = t * h_ * w_
            m = x[off : off + n].view(t, h_ // MERGE, MERGE, w_ // MERGE, MERGE, V_HID)
            m = m.permute(0, 1, 3, 2, 4, 5).reshape(-1, PROJ_IN)
            # norm applies pre-shuffle (use_postshuffle_norm=False): per 1152 slice
            merged.append(m)
            off += n
        y = torch.cat(merged, dim=0)
        pw = lambda k: self.w[f"model.projector.{k}"]
        yn = F.layer_norm(y.view(-1, V_HID).float(), (V_HID,), pw("norm.weight").float(), pw("norm.bias").float(), 1e-6).to(BF).view(-1, PROJ_IN)
        z = F.linear(yn, pw("linear_fc1.weight"), pw("linear_fc1.bias"))
        z = F.gelu(z)
        z = F.linear(z, pw("linear_fc2.weight"), pw("linear_fc2.bias"))
        return z

    # ---------------- mrope ----------------
    def _cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids [3, S] -> interleaved-mrope cos/sin [S, HD]."""
        freqs = position_ids.float()[:, :, None] * self.inv_freq[None, None, :]  # [3, S, 64]
        ft = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            idx = slice(offset, MROPE_SECTION[axis] * 3, 3)
            ft[:, idx] = freqs[axis][:, idx]
        emb = torch.cat((ft, ft), dim=-1)
        return emb.cos().to(BF), emb.sin().to(BF)

    def _rope_index(self, input_ids: torch.Tensor, grid_thw: torch.Tensor | None, is_video: bool) -> tuple[torch.Tensor, int]:
        """[S] ids -> [3, S] t/h/w position ids via the shared Qwen3-VL walk."""
        from flash_rt.frontends.torch._qwen3_vl_geometry import mrope_position_ids

        pos = mrope_position_ids(
            input_ids,
            image_grid_thw=None if is_video else grid_thw,
            video_grid_thw=grid_thw if is_video else None,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            vision_start_token_id=VISION_START_ID,
            spatial_merge_size=MERGE,
        )
        return pos.to(DEV), int(pos.max().item()) + 1

    # ---------------- text tower ----------------
    def _layer_torch(self, li: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                     start: int, causal: bool) -> torch.Tensor:
        lw = lambda k: self.w[f"layers.{li}.{k}"]
        S = x.shape[0]
        h = _rms(x, lw("input_layernorm.weight"))
        q = F.linear(h, lw("self_attn.to_q.weight")).view(S, NH, HD)
        k = F.linear(h, lw("self_attn.to_k.weight")).view(S, NKV, HD)
        v = F.linear(h, lw("self_attn.to_v.weight")).view(S, NKV, HD)
        cu = cos.unsqueeze(1)
        su = sin.unsqueeze(1)
        q = (q * cu) + (_rotate_half(q) * su)
        k = (k * cu) + (_rotate_half(k) * su)
        self.k_cache[li, start : start + S] = k
        self.v_cache[li, start : start + S] = v
        kk = self.k_cache[li, : start + S].transpose(0, 1)
        vv = self.v_cache[li, : start + S].transpose(0, 1)
        qq = q.transpose(0, 1)
        o = F.scaled_dot_product_attention(qq, kk, vv, is_causal=causal and S > 1, enable_gqa=True)
        o = o.transpose(0, 1).reshape(S, HID)
        x = x + F.linear(o, lw("self_attn.to_out.weight"))
        h = _rms(x, lw("post_attention_layernorm.weight"))
        h = F.linear(h, lw("mlp.up_proj.weight"))
        h = F.relu(h).square().to(BF)
        h = F.linear(h, lw("mlp.down_proj.weight"))
        return x + h

    def _forward(self, embeds: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 start: int, causal: bool) -> torch.Tensor:
        x = embeds
        for li in range(NL):
            x = self._layer_torch(li, x, cos, sin, start, causal)
        x = _rms(x[-1:], self.w["norm.weight"])
        return F.linear(x, self.w["lm_head.weight"])  # [1, VOCAB]

    # ---------------- public API ----------------
    def _validate_request(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        is_video: bool,
        max_new_tokens: int | None,
    ) -> tuple[int, int]:
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
            shape = getattr(input_ids, "shape", None)
            raise ValueError(f"input_ids must be a rank-1 tensor, got shape={shape}")
        if input_ids.dtype != torch.long:
            raise ValueError(f"input_ids must have dtype=torch.long, got {input_ids.dtype}")
        S = int(input_ids.shape[0])
        if S <= 0:
            raise ValueError("input_ids must contain at least one prompt token")

        if max_new_tokens is None:
            max_new = self.max_new
        else:
            if isinstance(max_new_tokens, bool):
                raise ValueError(
                    f"max_new_tokens must be an integer in [1, {self.max_new}], got {max_new_tokens!r}"
                )
            try:
                max_new = operator.index(max_new_tokens)
            except TypeError as exc:
                raise ValueError(
                    f"max_new_tokens must be an integer in [1, {self.max_new}], got {max_new_tokens!r}"
                ) from exc
        if not 1 <= max_new <= self.max_new:
            raise ValueError(
                f"max_new_tokens must be in [1, {self.max_new}], got {max_new}"
            )
        if S + max_new > self.max_seq:
            raise ValueError(
                f"prompt_tokens + max_new_tokens exceeds max_seq: "
                f"{S} + {max_new} > {self.max_seq}"
            )

        has_pixels = pixel_values is not None
        has_grid = grid_thw is not None
        if has_pixels != has_grid:
            raise ValueError("pixel_values and grid_thw must be provided together")
        vision_token = VIDEO_TOKEN_ID if is_video else IMAGE_TOKEN_ID
        vision_tokens = int((input_ids == vision_token).sum().item())
        if not has_pixels:
            if vision_tokens:
                raise ValueError(
                    f"input_ids contains {vision_tokens} vision tokens but no visual input was provided"
                )
            return S, max_new

        pixel_values = cast(torch.Tensor, pixel_values)
        grid_thw = cast(torch.Tensor, grid_thw)
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 2:
            shape = getattr(pixel_values, "shape", None)
            raise ValueError(f"pixel_values must be a rank-2 tensor, got shape={shape}")
        if not isinstance(grid_thw, torch.Tensor) or grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            shape = getattr(grid_thw, "shape", None)
            raise ValueError(f"grid_thw must have shape [num_media, 3], got shape={shape}")
        if grid_thw.shape[0] <= 0 or grid_thw.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        ):
            raise ValueError(
                f"grid_thw must be a non-empty integer tensor, got shape={grid_thw.shape} "
                f"dtype={grid_thw.dtype}"
            )
        grid_cpu = grid_thw.detach().to(device="cpu", dtype=torch.long)
        if bool((grid_cpu <= 0).any()):
            raise ValueError(f"grid_thw entries must be positive, got {grid_cpu.tolist()}")
        if bool((grid_cpu[:, 1:] % MERGE != 0).any()):
            raise ValueError(
                f"grid_thw spatial dimensions must be divisible by merge={MERGE}, "
                f"got {grid_cpu.tolist()}"
            )
        expected_patches = int(grid_cpu.prod(dim=1).sum().item())
        if pixel_values.shape[0] != expected_patches:
            raise ValueError(
                "pixel_values rows do not match grid_thw: "
                f"rows={pixel_values.shape[0]}, expected={expected_patches}, "
                f"grid_thw={grid_cpu.tolist()}"
            )
        expected_width = int(self.w["model.visual.embeddings.patch_embedding.weight"].shape[1])
        if pixel_values.shape[1] != expected_width:
            raise ValueError(
                f"pixel_values width must be {expected_width}, got {pixel_values.shape[1]}"
            )
        expected_tokens = int(
            (grid_cpu[:, 0] * (grid_cpu[:, 1] // MERGE) * (grid_cpu[:, 2] // MERGE)).sum().item()
        )
        if vision_tokens != expected_tokens:
            raise ValueError(
                f"input_ids contains {vision_tokens} token_id={vision_token} entries, "
                f"but grid_thw requires {expected_tokens} merged vision tokens"
            )
        return S, max_new

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,  # [S] long
        *,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        is_video: bool = False,
        max_new_tokens: int | None = None,
        ignore_eos: bool = False,
    ) -> tuple[list[int], dict[str, float]]:
        import time

        S, max_new = self._validate_request(
            input_ids, pixel_values, grid_thw, is_video, max_new_tokens
        )
        # Capture before prefill. Graph warmup writes decode scratch KV slots;
        # the real prefill below then overwrites the complete valid prompt range.
        self._ensure_graph()
        input_ids = input_ids.to(DEV)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        embeds = F.embedding(input_ids, self.w["embed_tokens.weight"])
        if pixel_values is not None:
            vision_grid = cast(torch.Tensor, grid_thw)
            feats = self.encode_vision(pixel_values, vision_grid)
            token = VIDEO_TOKEN_ID if is_video else IMAGE_TOKEN_ID
            mask = input_ids == token
            if int(mask.sum()) != feats.shape[0]:
                raise ValueError(
                    f"encoded vision rows changed unexpectedly: tokens={int(mask.sum())}, "
                    f"features={feats.shape[0]}"
                )
            embeds = embeds.clone()
            embeds[mask] = feats
            pos, nxt = self._rope_index(input_ids.cpu(), vision_grid.cpu(), is_video)
        else:
            pos = torch.arange(S, device=DEV).view(1, S).expand(3, S)
            nxt = S
        cos, sin = self._cos_sin(pos)
        logits = self._forward(embeds, cos, sin, 0, causal=True)
        next_tok = int(logits.argmax(-1))
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0

        out = [next_tok]
        self.seq_len = S
        t0 = time.perf_counter()
        if self.quant == "fp4":
            self.k8_cache[:, :S].copy_(self.k_cache[:, :S].to(torch.float8_e4m3fn).view(torch.uint8))
            self.v8_cache[:, :S].copy_(self.v_cache[:, :S].to(torch.float8_e4m3fn).view(torch.uint8))
        if self.quant in ("fp4", "bf16"):
            self.d_tok.fill_(out[0])
            self.d_pos.fill_(nxt)
            self.d_slot.fill_(self.seq_len)
            self.d_len.fill_(self.seq_len + 1)
            self.d_step.zero_()
            for _ in range(max_new - 1):
                if self._graph is not None:
                    self._graph.replay()
                elif self.quant == "fp4":
                    self._decode_step_fp4()
                else:
                    self._decode_step_bf16()
            toks = self.d_out_tokens[: max_new - 1].tolist()
            for t in toks:
                out.append(int(t))
                self.seq_len += 1
                if not ignore_eos and t == self.eos_token_id:
                    break
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t0
        n_dec = len(out) - 1
        return out, {
            "prompt_tokens": S,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "decode_tok_s": (n_dec / decode_s) if n_dec and decode_s > 0 else 0.0,
            "new_tokens": len(out),
        }
