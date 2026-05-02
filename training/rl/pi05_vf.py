"""ValueFunctionHead wired to a (frozen) pi0.5 backbone.

This is the W6 production path: the distributional VF runs on
**real VLA prefix embeddings**, not on a synthetic state vector. The
pi0.5 encoder is frozen (no grad on SigLIP / Gemma-2B / Gemma-300M),
so only the small 3-layer head receives gradients during VF training
— matching paper Section IV-A.

Prefix embedding flow (matches openpi PI0Pytorch.embed_prefix):

    images, img_masks → SigLIP image features            (B, S_img, D_vlm)
    lang_tokens, lang_masks → Gemma embedding lookup     (B, S_lang, D_vlm)
    concatenated, masked → prefix_embs, prefix_pad_masks

The two helpers below close the loop:

* ``pool_masked_mean`` — token-axis mean over the valid (unpadded)
  positions. Identical to what the W7/W8 RECAP loop will use.
* ``Pi05ValueFunction`` — owns a frozen backbone reference + a
  trainable ``ValueFunctionHead``. Caller supplies prefix tensors
  to keep the wrapper model-agnostic at the data-loading level
  (mirrors ``Pi05Trainer``'s split: trainer owns the FP8 + LoRA
  plumbing, the data path is the caller's responsibility).
"""

from __future__ import annotations

import torch
from torch import nn

from flash_rt.core.rl.reward import (
    DEFAULT_BIN_MAX,
    DEFAULT_BIN_MIN,
    DEFAULT_NUM_BINS,
    expected_value_from_logits,
)
from flash_rt.core.rl.value_function import ValueFunctionHead


def pool_masked_mean(
    embeddings: torch.Tensor,
    pad_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean-pool ``embeddings`` along the token axis using ``pad_mask``.

    Args:
        embeddings: ``(B, T, D)`` token embeddings.
        pad_mask: ``(B, T)`` boolean / 0-1 tensor; ``True``/``1`` marks
            valid (non-padded) positions, ``False``/``0`` is ignored.
        eps: Numerical floor for the per-row token count, in case a
            row has zero valid tokens. The fallback returns the
            unweighted mean of zeros (i.e. zero) for that row.

    Returns:
        Pooled embeddings of shape ``(B, D)`` in the same dtype as
        the input embeddings.
    """
    if embeddings.ndim != 3:
        raise ValueError(
            f"embeddings must be (B, T, D), got shape={tuple(embeddings.shape)}"
        )
    if pad_mask.ndim != 2 or pad_mask.shape != embeddings.shape[:2]:
        raise ValueError(
            f"pad_mask shape {tuple(pad_mask.shape)} must match "
            f"(B, T) of embeddings ({tuple(embeddings.shape[:2])})"
        )

    mask = pad_mask.to(embeddings.dtype).unsqueeze(-1)  # (B, T, 1)
    masked_sum = (embeddings * mask).sum(dim=1)  # (B, D)
    counts = mask.sum(dim=1).clamp_min(eps)  # (B, 1)
    return masked_sum / counts


class Pi05ValueFunction(nn.Module):
    """Frozen pi0.5 backbone reference + trainable distributional VF head.

    The wrapper does **not** own the backbone — it holds a reference
    so ``state_dict()`` only saves the head parameters, keeping
    checkpoints small (~5 MB for the 3-layer head vs ~3 GB if the
    backbone slipped in). Frozen-ness is enforced at construction by
    setting ``requires_grad=False`` on every backbone parameter.

    Args:
        backbone: Any module that exposes pi0.5-style prefix
            embeddings (e.g. a vendored ``PI0Pytorch`` instance, or
            in tests, ``PaliGemmaWithExpertModel`` driven by hand).
            Pure reference; not registered as a submodule.
        input_dim: Width of the pooled prefix embedding (``2048`` for
            pi0.5's Gemma-2B encoder).
        hidden_dim: Hidden width of the VF head MLP.
        num_bins: Number of distributional value bins.
        num_layers: Total layers in the VF head MLP (>= 1).
        dropout: Dropout inside the head.
        bin_min, bin_max: Support of the bin grid.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        num_bins: int = DEFAULT_NUM_BINS,
        num_layers: int = 3,
        dropout: float = 0.1,
        bin_min: float = DEFAULT_BIN_MIN,
        bin_max: float = DEFAULT_BIN_MAX,
    ):
        super().__init__()
        # Keep backbone OUT of the submodule tree so it is neither
        # part of state_dict nor parameter iteration. The caller
        # remains free to keep training the LoRA adapters on it.
        object.__setattr__(self, "_backbone_ref", backbone)
        for p in backbone.parameters():
            p.requires_grad_(False)

        self.head = ValueFunctionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            num_layers=num_layers,
            dropout=dropout,
            bin_min=bin_min,
            bin_max=bin_max,
        )

    @property
    def backbone(self) -> nn.Module:
        """The backbone reference (not a submodule, not in state_dict)."""
        return self._backbone_ref  # type: ignore[attr-defined]

    @property
    def bin_centers(self) -> torch.Tensor:
        return self.head.bin_centers

    def forward(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Distributional VF logits ``(B, num_bins)`` from prefix embeddings."""
        pooled = pool_masked_mean(prefix_embs, prefix_pad_masks)
        return self.head(pooled)

    def predict_value(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Continuous expected value ``(B,)`` from prefix embeddings."""
        logits = self.forward(prefix_embs, prefix_pad_masks)
        return expected_value_from_logits(logits, self.bin_centers.to(logits.device))
