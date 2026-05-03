"""FlashRT -- RTX Pi0 JAX frontend.

Loads Pi0 Orbax checkpoints (the JAX-native format shipped by openpi) and
drives the same framework-agnostic ``Pi0Pipeline`` as the torch frontend.

Design mirrors :mod:`flash_rt.frontends.jax.pi05_rtx` — a **thin shim**
over :class:`Pi0TorchFrontendRtx`. The JAX-specific work is the weight
loader (Orbax -> FP16 torch tensors with the same dict schema as
``convert_pi0_safetensors``). Once weights match, every downstream step
(FP8 quantize, time_proj precompute, FP8 calibration, CUDA Graph capture,
infer) is shared with the torch path.

Pi0 vs Pi0.5 Orbax schema differences
-------------------------------------
* **Decoder norms**: Pi0 uses plain RMSNorm → keys end in ``.scale``
  (shape ``(18, 1024)``). Pi0.5 uses AdaRMSNorm →
  ``.Dense_0.kernel/.bias``. For Pi0 we pre-fold ``(1 + scale)`` into
  QKV/gate+up like the encoder.
* **Final decoder norm**: Pi0 ``final_norm_1.scale``, Pi0.5
  ``final_norm_1.Dense_0.kernel``.
* **action_time_mlp**: Pi0 has two Dense (in + out); the in layer maps
  ``concat(action, time_emb)`` → action (``(2048, 1024)`` kernel), split
  row-wise into action half ``kernel[:1024]`` and time half
  ``kernel[1024:]``. Pi0.5 has ``time_mlp_in/out`` instead.
* **state_proj**: Pi0-only, maps 32-dim state → 1024-dim prefix token.

Usage::

    from flash_rt.frontends.jax.pi0_rtx import Pi0JaxFrontendRtx
    pipe = Pi0JaxFrontendRtx("/path/to/pi0_base", num_views=2)
    pipe.set_prompt("pick up the red block")
    pipe.calibrate_with_real_data([obs])
    out = pipe.infer({"image": img, "wrist_image": wrist, "state": state})
    actions = out["actions"]
"""

from __future__ import annotations

import logging
import pathlib
from typing import Union

import ml_dtypes
import numpy as np
import torch

from flash_rt.models.pi0.pipeline_rtx import (
    ACTION_DIM,
    DEC_D,
    DEC_L,
    ENC_L,
    NUM_STEPS_DEFAULT,
    VIS_L,
)
from flash_rt.frontends.torch.pi0_rtx import (
    CHUNK_SIZE_DEFAULT,
    MAX_PROMPT_LEN_DEFAULT,
    Pi0TorchFrontendRtx,
)

logger = logging.getLogger(__name__)

fp16 = torch.float16


# ════════════════════════════════════════════════════════════════════
#   Orbax → FP16 torch dict (rtx schema, identical to safetensors path)
# ════════════════════════════════════════════════════════════════════


def _to_fp16_cuda(arr: np.ndarray) -> torch.Tensor:
    """Numpy → contiguous FP16 cuda tensor.

    Accepts fp32 / fp16 / ml_dtypes.bfloat16 input. Routes through fp32
    staging (when needed) so the final ``.to(fp16)`` cast is well-defined.
    """
    if arr.dtype == ml_dtypes.bfloat16:
        u16 = np.ascontiguousarray(arr).view(np.uint16)
        t = torch.from_numpy(u16).view(torch.bfloat16)
        return t.to("cuda", non_blocking=False).to(fp16).contiguous()
    return torch.from_numpy(np.ascontiguousarray(arr)).to(
        device="cuda", dtype=fp16
    ).contiguous()


def _interleave_qk_np(w: np.ndarray, num_heads: int) -> np.ndarray:
    """Numpy QK head-dim pair-interleave (matches fused-RoPE kernel)."""
    out_dim, in_dim = w.shape
    head_dim = out_dim // num_heads
    return (
        w.reshape(num_heads, head_dim, in_dim)
         .reshape(num_heads, 2, head_dim // 2, in_dim)
         .transpose(0, 2, 1, 3)
         .reshape(out_dim, in_dim)
    )


def convert_pi0_orbax(
    checkpoint_dir: Union[str, pathlib.Path],
) -> dict:
    """Convert a Pi0 Orbax JAX checkpoint to the rtx pipeline weight dict.

    Output schema matches ``convert_pi0_safetensors`` (torch FP16 cuda
    tensors keyed by rtx names) so the same downstream FP8 quant,
    time_proj precompute, and pipeline build apply to both frontends.
    """
    from flash_rt.core.weights.loader import _load_orbax

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    logger.info("Loading Pi0 Orbax checkpoint: %s", checkpoint_dir)
    raw = _load_orbax(str(checkpoint_dir))

    # Bit-truncate fp32 → bf16 → fp32 so the FP8 quant scales we compute
    # later match the torch path bit-for-bit (torch frontend loads
    # safetensors that were saved as bf16).
    raw = {
        k: v.astype(ml_dtypes.bfloat16).astype(np.float32)
        if v.dtype == np.float32 else v
        for k, v in raw.items()
    }

    ckpt: dict = {}

    # ── Vision (27 SigLIP layers) — identical layout to pi05 Orbax ──
    pe_w = raw["PaliGemma.img.embedding.kernel"]   # (14, 14, 3, 1152)
    ckpt["vision_patch_embedding_w"] = _to_fp16_cuda(pe_w)
    ckpt["vision_patch_embedding_b"] = _to_fp16_cuda(
        raw["PaliGemma.img.embedding.bias"])
    pos_emb = raw["PaliGemma.img.pos_embedding"].squeeze(0)  # (256, 1152)
    ckpt["vision_position_embedding"] = _to_fp16_cuda(pos_emb)

    qkv_w_list, qkv_b_list = [], []
    o_w_list, o_b_list = [], []
    up_w_list, up_b_list = [], []
    down_w_list, down_b_list = [], []
    ln1_w_list, ln1_b_list = [], []
    ln2_w_list, ln2_b_list = [], []

    enc_blk = "PaliGemma.img.Transformer.encoderblock"
    for i in range(VIS_L):
        ln1_w_list.append(raw[f"{enc_blk}.LayerNorm_0.scale"][i])
        ln1_b_list.append(raw[f"{enc_blk}.LayerNorm_0.bias"][i])
        ln2_w_list.append(raw[f"{enc_blk}.LayerNorm_1.scale"][i])
        ln2_b_list.append(raw[f"{enc_blk}.LayerNorm_1.bias"][i])

        q_w = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.query.kernel"][i]
        k_w = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.key.kernel"][i]
        v_w = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.value.kernel"][i]
        q_2d = q_w.reshape(1152, -1)
        k_2d = k_w.reshape(1152, -1)
        v_2d = v_w.reshape(1152, -1)
        qkv_w_list.append(np.concatenate([q_2d, k_2d, v_2d], axis=1))
        q_b = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.query.bias"][i].reshape(-1)
        k_b = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.key.bias"][i].reshape(-1)
        v_b = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.value.bias"][i].reshape(-1)
        qkv_b_list.append(np.concatenate([q_b, k_b, v_b]))

        o_w = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.out.kernel"][i]
        o_w_list.append(o_w.reshape(-1, 1152))
        o_b_list.append(
            raw[f"{enc_blk}.MultiHeadDotProductAttention_0.out.bias"][i])

        up_w_list.append(raw[f"{enc_blk}.MlpBlock_0.Dense_0.kernel"][i])
        up_b_list.append(raw[f"{enc_blk}.MlpBlock_0.Dense_0.bias"][i])
        down_w_list.append(raw[f"{enc_blk}.MlpBlock_0.Dense_1.kernel"][i])
        down_b_list.append(raw[f"{enc_blk}.MlpBlock_0.Dense_1.bias"][i])

    ckpt["vision_attn_qkv_w"] = _to_fp16_cuda(np.stack(qkv_w_list))
    ckpt["vision_attn_qkv_b"] = _to_fp16_cuda(np.stack(qkv_b_list))
    ckpt["vision_attn_o_w"] = _to_fp16_cuda(np.stack(o_w_list))
    ckpt["vision_attn_o_b"] = _to_fp16_cuda(np.stack(o_b_list))
    ckpt["vision_ffn_up_w"] = _to_fp16_cuda(np.stack(up_w_list))
    ckpt["vision_ffn_up_b"] = _to_fp16_cuda(np.stack(up_b_list))
    ckpt["vision_ffn_down_w"] = _to_fp16_cuda(np.stack(down_w_list))
    ckpt["vision_ffn_down_b"] = _to_fp16_cuda(np.stack(down_b_list))
    ckpt["vision_pre_attn_norm_w"] = _to_fp16_cuda(np.stack(ln1_w_list))
    ckpt["vision_pre_attn_norm_b"] = _to_fp16_cuda(np.stack(ln1_b_list))
    ckpt["vision_pre_ffn_norm_w"] = _to_fp16_cuda(np.stack(ln2_w_list))
    ckpt["vision_pre_ffn_norm_b"] = _to_fp16_cuda(np.stack(ln2_b_list))
    ckpt["vision_final_norm_w"] = _to_fp16_cuda(
        raw["PaliGemma.img.Transformer.encoder_norm.scale"])
    ckpt["vision_final_norm_b"] = _to_fp16_cuda(
        raw["PaliGemma.img.Transformer.encoder_norm.bias"])

    # ── Multi-modal projector ──
    ckpt["encoder_multi_modal_projector_w"] = _to_fp16_cuda(
        raw["PaliGemma.img.head.kernel"])
    ckpt["encoder_multi_modal_projector_b"] = _to_fp16_cuda(
        raw["PaliGemma.img.head.bias"])

    # ── Encoder (18 Gemma-2B layers, (1+scale) fold into QKV/gate_up) ──
    enc_qkv_list, enc_o_list = [], []
    enc_gate_list, enc_up_list, enc_down_list = [], [], []

    for i in range(ENC_L):
        # Fuse in fp32 to avoid bf16 rounding near -1.0 → 0 collapse
        attn_scale = raw[
            "PaliGemma.llm.layers.pre_attention_norm.scale"][i].astype(np.float32)
        fuse_attn = 1.0 + attn_scale  # (2048,)

        q_w = raw["PaliGemma.llm.layers.attn.q_einsum.w"][i].astype(np.float32)
        q_2d = q_w.transpose(0, 2, 1).reshape(-1, q_w.shape[1])  # (2048, 2048)
        q_2d = _interleave_qk_np(q_2d, 8)
        q_2d = q_2d * fuse_attn[None, :]

        kv_w = raw["PaliGemma.llm.layers.attn.kv_einsum.w"][i].astype(np.float32)
        k_2d = kv_w[0].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])  # (256, 2048)
        v_2d = kv_w[1].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])
        k_2d = _interleave_qk_np(k_2d, 1)
        k_2d = k_2d * fuse_attn[None, :]
        v_2d = v_2d * fuse_attn[None, :]

        qkv = np.concatenate([q_2d, k_2d, v_2d], axis=0).T  # (2048, 2560)
        enc_qkv_list.append(qkv)

        o_w = raw[
            "PaliGemma.llm.layers.attn.attn_vec_einsum.w"][i].astype(np.float32)
        enc_o_list.append(o_w.reshape(-1, o_w.shape[-1]))

        ffn_scale = raw[
            "PaliGemma.llm.layers.pre_ffw_norm.scale"][i].astype(np.float32)
        fuse_ffn = 1.0 + ffn_scale

        gu_w = raw["PaliGemma.llm.layers.mlp.gating_einsum"][i].astype(np.float32)
        gate_w = gu_w[0] * fuse_ffn[:, None]
        up_w = gu_w[1] * fuse_ffn[:, None]
        enc_gate_list.append(gate_w)
        enc_up_list.append(up_w)

        enc_down_list.append(
            raw["PaliGemma.llm.layers.mlp.linear"][i].astype(np.float32))

    ckpt["encoder_attn_qkv_w"] = _to_fp16_cuda(np.stack(enc_qkv_list))
    ckpt["encoder_attn_o_w"] = _to_fp16_cuda(np.stack(enc_o_list))
    ckpt["encoder_ffn_gate_w"] = _to_fp16_cuda(np.stack(enc_gate_list))
    ckpt["encoder_ffn_up_w"] = _to_fp16_cuda(np.stack(enc_up_list))
    ckpt["encoder_ffn_down_w"] = _to_fp16_cuda(np.stack(enc_down_list))

    # ── Decoder (18 Gemma-300M layers, Pi0 plain RMSNorm + (1+w) fold) ──
    dec_qkv_list, dec_o_list = [], []
    dec_gate_list, dec_up_list, dec_down_list = [], [], []

    for i in range(DEC_L):
        # Pi0 decoder pre-attention norm is PLAIN .scale (18, 1024), NOT
        # AdaRMS .Dense_0.kernel. Fold (1+scale) into QKV like encoder.
        attn_scale = raw[
            "PaliGemma.llm.layers.pre_attention_norm_1.scale"][i].astype(np.float32)
        fuse_attn = 1.0 + attn_scale  # (1024,)

        q_w = raw[
            "PaliGemma.llm.layers.attn.q_einsum_1.w"][i].astype(np.float32)
        q_2d = q_w.transpose(0, 2, 1).reshape(-1, q_w.shape[1])  # (2048, 1024)
        q_2d = _interleave_qk_np(q_2d, 8)
        q_2d = q_2d * fuse_attn[None, :]

        kv_w = raw[
            "PaliGemma.llm.layers.attn.kv_einsum_1.w"][i].astype(np.float32)
        k_2d = kv_w[0].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])  # (256, 1024)
        v_2d = kv_w[1].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])
        k_2d = _interleave_qk_np(k_2d, 1)
        k_2d = k_2d * fuse_attn[None, :]
        v_2d = v_2d * fuse_attn[None, :]

        qkv = np.concatenate([q_2d, k_2d, v_2d], axis=0).T  # (1024, 2560)
        dec_qkv_list.append(qkv)

        o_w = raw[
            "PaliGemma.llm.layers.attn.attn_vec_einsum_1.w"][i].astype(np.float32)
        dec_o_list.append(o_w.reshape(-1, o_w.shape[-1]))

        ffn_scale = raw[
            "PaliGemma.llm.layers.pre_ffw_norm_1.scale"][i].astype(np.float32)
        fuse_ffn = 1.0 + ffn_scale

        gu_w = raw["PaliGemma.llm.layers.mlp_1.gating_einsum"][i].astype(np.float32)
        gate_w = gu_w[0] * fuse_ffn[:, None]  # (1024, 4096)
        up_w = gu_w[1] * fuse_ffn[:, None]
        dec_gate_list.append(gate_w)
        dec_up_list.append(up_w)

        dec_down_list.append(
            raw["PaliGemma.llm.layers.mlp_1.linear"][i].astype(np.float32))

    ckpt["decoder_attn_qkv_w"] = _to_fp16_cuda(np.stack(dec_qkv_list))
    ckpt["decoder_attn_o_w"] = _to_fp16_cuda(np.stack(dec_o_list))
    ckpt["decoder_ffn_gate_w"] = _to_fp16_cuda(np.stack(dec_gate_list))
    ckpt["decoder_ffn_up_w"] = _to_fp16_cuda(np.stack(dec_up_list))
    ckpt["decoder_ffn_down_w"] = _to_fp16_cuda(np.stack(dec_down_list))

    # Final decoder norm: Pi0 plain scale. Fold (1 + scale) in fp32 → fp16.
    final_scale = raw["PaliGemma.llm.final_norm_1.scale"].astype(np.float32)
    ckpt["decoder_final_norm_w"] = _to_fp16_cuda(1.0 + final_scale)

    # ── Pi0 action / state projections ──
    # JAX kernels are stored in (in, out) layout, which is what the rtx
    # pipeline's row-major GEMM expects. No transpose needed.
    ckpt["state_proj_w"] = _to_fp16_cuda(raw["state_proj.kernel"])  # (32, 1024)
    ckpt["state_proj_b"] = _to_fp16_cuda(raw["state_proj.bias"])

    ckpt["decoder_action_in_proj_w"] = _to_fp16_cuda(
        raw["action_in_proj.kernel"])  # (32, 1024)
    ckpt["decoder_action_in_proj_b"] = _to_fp16_cuda(
        raw["action_in_proj.bias"])

    # action_time_mlp_in kernel: (2048, 1024) = (2*Da, Da). Input is
    # concat(action, time) so rows [:Da] map action input, rows [Da:] map
    # time input. Torch convention stores wa as (Da, Da) GEMM-B in the
    # rtx pipeline, and wt_raw in torch Linear (out, in) layout for the
    # precompute step. JAX kernel is (in, out): wa = kernel[:Da] directly
    # in (in, out); wt_raw must be transposed to match torch's (out, in).
    atmlp_in_full = raw["action_time_mlp_in.kernel"].astype(np.float32)  # (2048, 1024)
    ckpt["action_time_mlp_in_wa_w"] = _to_fp16_cuda(
        atmlp_in_full[:DEC_D, :])   # (1024, 1024) (in=action, out)
    ckpt["_action_time_mlp_in_wt_raw"] = _to_fp16_cuda(
        atmlp_in_full[DEC_D:, :].T.copy())  # (1024, 1024) (out, in=time)
    ckpt["_action_time_mlp_in_b"] = _to_fp16_cuda(
        raw["action_time_mlp_in.bias"])

    ckpt["action_time_mlp_out_w"] = _to_fp16_cuda(
        raw["action_time_mlp_out.kernel"])  # (1024, 1024) (in, out)
    ckpt["action_time_mlp_out_b"] = _to_fp16_cuda(
        raw["action_time_mlp_out.bias"])

    # action_out_proj — pre-scale by -1/num_steps later (in __init__ like
    # the torch frontend does). Store raw for now.
    ckpt["decoder_action_out_proj_w"] = _to_fp16_cuda(
        raw["action_out_proj.kernel"])  # (1024, 32) (in, out)
    ckpt["decoder_action_out_proj_b"] = _to_fp16_cuda(
        raw["action_out_proj.bias"])

    # ── Embedding matrix (prompt tokenisation) ──
    ckpt["embedding_weight"] = _to_fp16_cuda(
        raw["PaliGemma.llm.embedder.input_embedding"])

    logger.info("Converted %d Pi0 weight groups from Orbax", len(ckpt))
    return ckpt


# ════════════════════════════════════════════════════════════════════
#   Pi0JaxFrontendRtx — JAX Orbax frontend (thin shim over Pi0TorchFrontendRtx)
# ════════════════════════════════════════════════════════════════════


class Pi0JaxFrontendRtx(Pi0TorchFrontendRtx):
    """RTX consumer GPU Pi0 frontend backed by a JAX Orbax checkpoint.

    Thin override of :class:`Pi0TorchFrontendRtx`: only the weight loader
    changes (Orbax instead of safetensors). All downstream paths — FP8
    weight quant, time_proj precompute, FP8 calibration, CUDA Graph
    capture, the ``infer`` hot path, ``set_prompt``,
    ``calibrate_with_real_data`` — are inherited verbatim and produce
    byte-identical output to the torch frontend (modulo the fp32 → bf16
    → fp32 bit-truncation step applied to raw Orbax weights).
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, pathlib.Path],
        num_views: int = 2,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        max_prompt_len: int = MAX_PROMPT_LEN_DEFAULT,
        use_fp8: bool = True,
        use_fp8_decoder: bool = True,
    ):
        # Don't chain to Pi0TorchFrontendRtx.__init__ — it expects a
        # safetensors file. We replicate the body and swap the loader.
        import ctypes
        from flash_rt.hardware.rtx.attn_backend import RtxFlashAttnBackend
        from flash_rt import flash_rt_kernels as fvk
        from flash_rt.frontends.torch.pi0_rtx import _precompute_time_proj_all

        checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.num_views = int(num_views)
        self.chunk_size = int(chunk_size)          # Sa
        self.S_dec = self.chunk_size + 1            # 1 state + Sa actions
        self.max_prompt_len = int(max_prompt_len)
        self._use_fp8 = bool(use_fp8)
        self._use_fp8_decoder = bool(use_fp8_decoder)

        self.latency_records: list[float] = []
        self.calibrated = False
        self.graph_recorded = False
        self.current_prompt_len = 0
        self.pipeline = None

        # ── norm_stats (Orbax checkpoints ship per-embodiment stats under
        #    assets/<robot>/; reuse the torch frontend's search but also
        #    look at a sibling ``<name>_pytorch`` directory that is shipped
        #    with openpi's PyTorch conversion and contains
        #    ``assets/physical-intelligence/libero/norm_stats.json``.) ──
        self._load_norm_stats_jax(checkpoint_dir)

        # ── Load + convert Orbax ──
        self._checkpoint_path = str(checkpoint_dir)
        raw_ckpt = convert_pi0_orbax(checkpoint_dir)

        self._ckpt_fp16: dict = {
            k: v.contiguous() if isinstance(v, torch.Tensor) else v
            for k, v in raw_ckpt.items()
        }
        self.embedding_weight = self._ckpt_fp16["embedding_weight"]

        # Pre-scale decoder action output projection by -1/num_steps
        # (matches the torch frontend's pre-scaling step that bakes the
        # flow-matching residual coefficient into the weights).
        dt_scale = -1.0 / NUM_STEPS_DEFAULT
        self._ckpt_fp16["decoder_action_out_proj_w"] = (
            self._ckpt_fp16["decoder_action_out_proj_w"] * dt_scale
        ).contiguous()
        self._ckpt_fp16["decoder_action_out_proj_b"] = (
            self._ckpt_fp16["decoder_action_out_proj_b"] * dt_scale
        ).contiguous()

        # ── Pre-compute time_proj_all (shared helper from torch frontend) ──
        self._time_proj_all = _precompute_time_proj_all(
            self._ckpt_fp16, self.chunk_size, num_steps=NUM_STEPS_DEFAULT)

        # ── FP8 quantize large GEMMs (shared method) ──
        self._fp8_weights: dict = {}
        self._fp8_store: list = []
        self._quantize_all_fp8()

        # ── Attention backend ──
        enc_seq_max = self.num_views * 256 + self.max_prompt_len
        self.attn_backend = RtxFlashAttnBackend(
            num_views=self.num_views,
            encoder_seq_max=enc_seq_max,
            chunk_size=self.S_dec,
            num_encoder_layers=ENC_L,
            dtype=fp16,
        )

        self.fvk = fvk
        self.gemm = fvk.GemmRunner()

        IMG_HW = 224
        self._img_buf = torch.empty(
            self.num_views, IMG_HW, IMG_HW, 3, dtype=fp16, device="cuda"
        )
        self._noise_buf = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=fp16, device="cuda"
        )
        self._noise_out = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=fp16, device="cuda"
        )
        self._state_buf_host = torch.empty(
            1, ACTION_DIM, dtype=fp16, device="cuda"
        )
        self._cudart = ctypes.CDLL("libcudart.so")

        logger.info(
            "Pi0JaxFrontendRtx initialised (num_views=%d, Sa=%d)",
            self.num_views, self.chunk_size,
        )

    def _load_norm_stats_jax(self, checkpoint_dir: pathlib.Path) -> None:
        """Like the torch frontend's ``_load_norm_stats`` but also looks
        at the sibling ``<name>_pytorch`` directory (where openpi ships
        the LIBERO assets when the Orbax checkpoint itself only carries
        per-embodiment assets) and tolerates the lerobot HF release's
        ``meta/stats.json`` schema.
        """
        from flash_rt.core.utils.norm_stats import (
            load_norm_stats, lerobot_candidates,
        )
        name = checkpoint_dir.name
        candidates = [
            checkpoint_dir / "assets" / "physical-intelligence" / "libero"
            / "norm_stats.json",
            checkpoint_dir.parent / f"{name}_pytorch" / "assets"
            / "physical-intelligence" / "libero" / "norm_stats.json",
            checkpoint_dir.parent / "pi0_base_pytorch" / "assets"
            / "physical-intelligence" / "libero" / "norm_stats.json",
            checkpoint_dir / "norm_stats.json",
            *lerobot_candidates(checkpoint_dir),
        ]
        try:
            self.norm_stats = load_norm_stats(
                candidates, checkpoint_dir=checkpoint_dir)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"norm_stats not found near Pi0 Orbax checkpoint "
                f"{checkpoint_dir}: {e}") from e
