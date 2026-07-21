"""Native Wan2.2 VAE encode helpers for Cosmos3-Edge Thor.

This module is intentionally narrow: it only fuses the ResidualBlock
``RMS_norm -> SiLU`` pair into FlashRT's existing BF16 NCDHW kernel and
leaves CausalConv3d on the official cuDNN BF16 path. That gives a low-risk
first native VAE subgraph while preserving the upstream cache contract.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk


def _native_rms_silu_ncdhw(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or x.device.type != "cuda":
        raise TypeError("native Wan VAE RMS+SiLU expects a CUDA BF16 tensor")
    if x.dim() != 5:
        raise ValueError(f"native Wan VAE RMS+SiLU expects NCDHW, got {tuple(x.shape)}")
    b, c, t, h, w = (int(dim) for dim in x.shape)
    x_c = x if x.is_contiguous() else x.contiguous()
    gamma_c = gamma.reshape(-1).contiguous()
    out = torch.empty_like(x_c)
    rc = fvk.bf16_rms_silu_ncdhw(
        int(x_c.data_ptr()),
        int(gamma_c.data_ptr()),
        int(out.data_ptr()),
        0,
        0,
        b,
        c,
        t,
        h,
        w,
        1e-12,
        torch.cuda.current_stream().cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"bf16_rms_silu_ncdhw failed rc={rc}")
    return out


def _native_avgdown3d_bf16(module: Any, x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or x.device.type != "cuda":
        raise TypeError("native Wan VAE AvgDown3D expects a CUDA BF16 tensor")
    if x.dim() != 5:
        raise ValueError(f"native Wan VAE AvgDown3D expects NCDHW, got {tuple(x.shape)}")
    b, c, t, h, w = (int(dim) for dim in x.shape)
    factor_t = int(module.factor_t)
    factor_s = int(module.factor_s)
    out_c = int(module.out_channels)
    group_size = int(module.group_size)
    pad_t = (factor_t - (t % factor_t)) % factor_t
    if h % factor_s != 0 or w % factor_s != 0:
        raise ValueError("native Wan VAE AvgDown3D requires divisible spatial dimensions")
    x_c = x if x.is_contiguous() else x.contiguous()
    out = torch.empty(
        (b, out_c, (t + pad_t) // factor_t, h // factor_s, w // factor_s),
        device=x.device,
        dtype=x.dtype,
    )
    rc = fvk.cosmos3_edge_avgdown3d_bf16(
        int(x_c.data_ptr()),
        int(out.data_ptr()),
        b,
        c,
        t,
        h,
        w,
        out_c,
        factor_t,
        factor_s,
        group_size,
        torch.cuda.current_stream().cuda_stream,
    )
    if rc is not None and rc != 0:
        raise RuntimeError(f"cosmos3_edge_avgdown3d_bf16 failed rc={rc}")
    return out


def _find_triplet(
    layers: list[nn.Module],
    start: int,
    *,
    rms_cls: type[Any],
    causal_conv_cls: type[Any],
) -> tuple[int, int] | None:
    if start >= len(layers) or not isinstance(layers[start], rms_cls):
        return None
    silu_idx = start + 1
    if silu_idx >= len(layers) or not isinstance(layers[silu_idx], nn.SiLU):
        return None
    conv_idx = silu_idx + 1
    while conv_idx < len(layers) and isinstance(layers[conv_idx], nn.Dropout):
        conv_idx += 1
    if conv_idx >= len(layers) or not isinstance(layers[conv_idx], causal_conv_cls):
        return None
    return silu_idx, conv_idx


def install_wan_vae_encode_native_rms_silu() -> dict[str, Any]:
    """Patch Wan2.2 encoder ResidualBlock.forward with native RMS+SiLU.

    Returns a small stats dict for traceability. The patch is process-local and
    idempotent for the imported upstream classes.
    """
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import (
        CausalConv3d,
        RMS_norm,
        ResidualBlock,
        _update_cache_and_apply,
    )

    if getattr(ResidualBlock, "_flashrt_cosmos3_edge_native_rms_silu", False):
        return {"patched": False, "reason": "already_patched"}

    original_forward = ResidualBlock.forward

    def patched_forward(self: Any, x: torch.Tensor, feat_cache: list[Any] | None = None, feat_idx: list[int] = [0]):
        h = self.shortcut(x)
        layers = list(self.residual)
        i = 0
        while i < len(layers):
            triplet = _find_triplet(layers, i, rms_cls=RMS_norm, causal_conv_cls=CausalConv3d)
            if triplet is None:
                layer = layers[i]
                if isinstance(layer, CausalConv3d) and feat_cache is not None:
                    x = _update_cache_and_apply(x, layer, feat_cache, feat_idx)
                else:
                    x = layer(x)
                i += 1
                continue

            _silu_idx, conv_idx = triplet
            rms = layers[i]
            conv = layers[conv_idx]
            try:
                s = _native_rms_silu_ncdhw(x, rms.gamma)
            except Exception:
                s = layers[i + 1](rms(x))
            for dropout_idx in range(i + 2, conv_idx):
                s = layers[dropout_idx](s)
            if feat_cache is not None:
                x = _update_cache_and_apply(s, conv, feat_cache, feat_idx)
            else:
                x = conv(s)
            i = conv_idx + 1
        return x + h

    ResidualBlock._flashrt_cosmos3_edge_original_forward = original_forward
    ResidualBlock.forward = patched_forward
    ResidualBlock._flashrt_cosmos3_edge_native_rms_silu = True
    return {"patched": True, "target": "Wan2.2 ResidualBlock RMS_norm+SiLU"}


def install_wan_vae_encode_native_avgdown3d() -> dict[str, Any]:
    """Patch Wan2.2 encoder AvgDown3D shortcut pooling to FlashRT CUDA."""
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import AvgDown3D

    if getattr(AvgDown3D, "_flashrt_cosmos3_edge_native_avgdown3d", False):
        return {"patched": False, "reason": "already_patched"}

    original_forward = AvgDown3D.forward

    def patched_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        try:
            return _native_avgdown3d_bf16(self, x)
        except Exception:
            return original_forward(self, x)

    AvgDown3D._flashrt_cosmos3_edge_original_forward = original_forward
    AvgDown3D.forward = patched_forward
    AvgDown3D._flashrt_cosmos3_edge_native_avgdown3d = True
    return {"patched": True, "target": "Wan2.2 AvgDown3D BF16"}


def _causal_conv3d_t1_as_conv2d(conv: Any, x: torch.Tensor) -> torch.Tensor | None:
    if x.dim() != 5 or int(x.shape[2]) != 1:
        return None
    if tuple(int(item) for item in conv.stride) != (1, 1, 1):
        return None
    if tuple(int(item) for item in conv.dilation) != (1, 1, 1):
        return None
    if int(conv.groups) != 1:
        return None
    weight = conv.weight
    if weight.dim() != 5:
        return None

    k_t = int(weight.shape[2])
    temporal_index = int(conv._padding[4])
    if temporal_index < 0 or temporal_index >= k_t:
        return None

    pad_w_l, pad_w_r, pad_h_t, pad_h_b = (int(item) for item in conv._padding[:4])
    x2 = x[:, :, 0, :, :]
    if pad_w_l == pad_w_r and pad_h_t == pad_h_b:
        y2 = F.conv2d(
            x2,
            weight[:, :, temporal_index, :, :],
            conv.bias,
            stride=(int(conv.stride[1]), int(conv.stride[2])),
            padding=(pad_h_t, pad_w_l),
            dilation=(int(conv.dilation[1]), int(conv.dilation[2])),
            groups=int(conv.groups),
        )
    else:
        y2 = F.conv2d(
            F.pad(x2, (pad_w_l, pad_w_r, pad_h_t, pad_h_b)),
            weight[:, :, temporal_index, :, :],
            conv.bias,
            stride=(int(conv.stride[1]), int(conv.stride[2])),
            padding=0,
            dilation=(int(conv.dilation[1]), int(conv.dilation[2])),
            groups=int(conv.groups),
        )
    return y2.unsqueeze(2).contiguous()


def install_wan_vae_encode_t1_conv2d() -> dict[str, Any]:
    """Patch Wan2.2 CausalConv3d prime chunks to equivalent Conv2d.

    For a no-cache, single-frame chunk, the causal temporal pad means a
    kernel-3 CausalConv3d only consumes the last temporal filter slice. This
    opt-in keeps steady cached chunks on the official Conv3d path and only
    rewrites the cheap but frequent prime/single-frame sites for A/B.
    """
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import CausalConv3d

    if getattr(CausalConv3d, "_flashrt_cosmos3_edge_t1_conv2d", False):
        return {"patched": False, "reason": "already_patched"}

    original_forward = CausalConv3d.forward

    def patched_forward(self: Any, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        if cache_x is None:
            y = _causal_conv3d_t1_as_conv2d(self, x)
            if y is not None:
                return y
        return original_forward(self, x, cache_x)

    CausalConv3d._flashrt_cosmos3_edge_original_forward = original_forward
    CausalConv3d.forward = patched_forward
    CausalConv3d._flashrt_cosmos3_edge_t1_conv2d = True
    return {"patched": True, "target": "Wan2.2 CausalConv3d T=1 no-cache Conv2d"}


def _causal_conv3d_channels_last3d_320(conv: Any, x: torch.Tensor, cache_x: torch.Tensor | None) -> torch.Tensor | None:
    if cache_x is None or x.dim() != 5 or x.dtype != torch.bfloat16 or x.device.type != "cuda":
        return None
    b, c, t, h, w = (int(dim) for dim in x.shape)
    if (c, h, w) != (320, 120, 208) or t <= 1:
        return None
    weight = conv.weight
    if weight.dim() != 5 or tuple(int(dim) for dim in weight.shape) != (320, 320, 3, 3, 3):
        return None
    if tuple(int(item) for item in conv.stride) != (1, 1, 1):
        return None
    if tuple(int(item) for item in conv.dilation) != (1, 1, 1):
        return None
    if int(conv.groups) != 1:
        return None

    padding = list(int(item) for item in conv._padding)
    if padding != [1, 1, 1, 1, 2, 0]:
        return None
    x_eff = x
    if padding[4] > 0:
        x_eff = torch.cat([cache_x.to(device=x.device, dtype=x.dtype), x_eff], dim=2)
        padding[4] -= int(cache_x.shape[2])
    x_eff = F.pad(x_eff, padding).contiguous(memory_format=torch.channels_last_3d)
    weight_cl = weight.contiguous(memory_format=torch.channels_last_3d)
    return F.conv3d(x_eff, weight_cl, conv.bias).contiguous()


def install_wan_vae_encode_channels_last3d_conv320() -> dict[str, Any]:
    """Patch steady 320-channel Wan2.2 CausalConv3d sites to channels_last_3d.

    The formal probe shows per-conv conversion is only positive on the 320x
    steady cached sites. This opt-in patch leaves 160/640-channel sites on the
    official path and keeps outputs contiguous for the upstream cache contract.
    """
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import CausalConv3d

    if getattr(CausalConv3d, "_flashrt_cosmos3_edge_channels_last3d_conv320", False):
        return {"patched": False, "reason": "already_patched"}

    original_forward = CausalConv3d.forward

    def patched_forward(self: Any, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        y = _causal_conv3d_channels_last3d_320(self, x, cache_x)
        if y is not None:
            return y
        return original_forward(self, x, cache_x)

    CausalConv3d._flashrt_cosmos3_edge_channels_last3d_original_forward = original_forward
    CausalConv3d.forward = patched_forward
    CausalConv3d._flashrt_cosmos3_edge_channels_last3d_conv320 = True
    return {"patched": True, "target": "Wan2.2 CausalConv3d 320ch channels_last_3d"}
