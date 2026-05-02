r"""Distributional value-function modules for RECAP.

Two flavours, sharing the soft-binning math from ``reward.py``:

``StandaloneValueFunction``
    State-only (and optionally image-only) MLP head. Used by the
    synthetic-LIBERO pipeline test in ``training/rl/train_value.py``
    to validate the loss + threshold path before the VLA backbone
    is involved. Mirrors openpi-compiler's prototype.

``ValueFunctionHead``
    Lightweight 3-layer MLP designed to attach to a *frozen* VLA
    prefix embedding (output of ``embed_prefix``, pooled). This is
    the production path: the VF reuses the encoder's vision +
    language features without re-running them, and only the head
    parameters are updated by ``train_value``.

Both heads emit logits over ``num_bins=201`` bins in
``[bin_min=-1, bin_max=0]``, matching paper Section IV-A.

Source: ported from ``openpi-compiler/RL/recap/value_function.py``
2026-04-25; the legacy ``compute_value_loss`` (hard cross-entropy)
is dropped in favour of ``reward.compute_soft_value_loss``.
"""

from __future__ import annotations

import torch
from torch import nn

from .reward import (
    DEFAULT_BIN_MAX,
    DEFAULT_BIN_MIN,
    DEFAULT_NUM_BINS,
    build_bin_centers,
    expected_value_from_logits,
)


class ValueFunctionHead(nn.Module):
    """Distributional VF head over a pooled VLA prefix embedding.

    The expected pi0.5 wiring::

        prefix_embs, _, _ = trainer.model.embed_prefix(images, ...)
        pooled = prefix_embs.mean(dim=1)              # (B, ENC_D)
        head = ValueFunctionHead(input_dim=2048)
        logits = head(pooled)                         # (B, num_bins)
        loss = compute_soft_value_loss(logits, targets, head.bin_centers)

    Args mirror ``StandaloneValueFunction`` for swap-ability.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        num_bins: int = DEFAULT_NUM_BINS,
        num_layers: int = 3,
        dropout: float = 0.1,
        bin_min: float = DEFAULT_BIN_MIN,
        bin_max: float = DEFAULT_BIN_MAX,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.input_dim = input_dim
        self.num_bins = num_bins
        self.bin_min = bin_min
        self.bin_max = bin_max

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_bins))
        self.mlp = nn.Sequential(*layers)

        self.register_buffer(
            "bin_centers",
            build_bin_centers(num_bins, bin_min, bin_max),
            persistent=False,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Distributional logits over bins, shape ``(B, num_bins)``."""
        return self.mlp(embeddings)

    def predict_value(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Expected continuous value, shape ``(B,)``."""
        logits = self.forward(embeddings)
        return expected_value_from_logits(logits, self.bin_centers.to(logits.device))


class StandaloneValueFunction(nn.Module):
    """State-only (optionally + image) VF for synthetic-data testing.

    Used by ``training/rl/train_value.py`` to validate the
    loss + threshold path before the VLA backbone is involved.
    Mirrors openpi-compiler's prototype.
    """

    def __init__(
        self,
        state_dim: int = 32,
        image_channels: int = 3,
        image_size: int = 224,
        hidden_dim: int = 256,
        num_bins: int = DEFAULT_NUM_BINS,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_images: bool = False,
        bin_min: float = DEFAULT_BIN_MIN,
        bin_max: float = DEFAULT_BIN_MAX,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.num_bins = num_bins
        self.use_images = use_images
        self.image_size = image_size

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if use_images:
            self.image_encoder = nn.Sequential(
                nn.Conv2d(image_channels, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, hidden_dim),
            )
            head_input_dim = hidden_dim * 2
        else:
            head_input_dim = hidden_dim

        value_layers: list[nn.Module] = []
        in_dim = head_input_dim
        for _ in range(num_layers - 1):
            value_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        value_layers.append(nn.Linear(in_dim, num_bins))
        self.value_head = nn.Sequential(*value_layers)

        self.register_buffer(
            "bin_centers",
            build_bin_centers(num_bins, bin_min, bin_max),
            persistent=False,
        )

    def forward(
        self,
        state: torch.Tensor,
        images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Distributional logits over bins, shape ``(B, num_bins)``."""
        state_feat = self.state_encoder(state)
        if self.use_images and images is not None:
            img_feat = self.image_encoder(images)
            feat = torch.cat([state_feat, img_feat], dim=-1)
        else:
            feat = state_feat
        return self.value_head(feat)

    def predict_value(
        self,
        state: torch.Tensor,
        images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expected continuous value, shape ``(B,)``."""
        logits = self.forward(state, images)
        return expected_value_from_logits(logits, self.bin_centers.to(logits.device))
