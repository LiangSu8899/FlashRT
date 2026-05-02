"""Classifier-Free Guidance for advantage-conditioned VLA inference.

Implements the CFG combination from π*0.6 (arXiv:2511.14759, Appendix
E). Each denoising step combines the conditioned (advantage = positive)
and unconditioned velocity predictions into a sharpened guided velocity:

    v_guided = v_uncond + beta * (v_cond - v_uncond)
             = (1 - beta) * v_uncond + beta * v_cond

This module provides the math and prompt-building helpers; the per-step
two-forward orchestration is the responsibility of each frontend, which
controls how the two forwards are scheduled (serial, dual-stream, or
fused batch=2).

When ``beta == 1.0`` the formula collapses to ``v_cond``, so frontends
should fall back to a single forward pass on the conditioned prompt.
"""

from __future__ import annotations

from .acp_tags import build_acp_tagged_task, build_unconditioned_task


class CFGSampler:
    """Stateless CFG configuration + prompt builder.

    Args:
        beta: Guidance strength. ``beta = 1.0`` disables CFG (single
            forward, conditioned prompt only). ``beta > 1.0`` sharpens
            the conditioned distribution at the cost of one extra
            forward per denoising step. The π*0.6 paper recommends
            ``beta in [1.5, 2.5]`` for deployment.
        advantage_positive: Whether the conditioned prompt uses the
            positive advantage tag. Always ``True`` for the standard
            "select for high-advantage actions" use case; exposed for
            symmetry / debugging.
    """

    def __init__(self, beta: float = 1.5, advantage_positive: bool = True):
        if beta < 1.0:
            raise ValueError(
                f"CFG beta must be >= 1.0 (1.0 disables CFG); got {beta}")
        self.beta = float(beta)
        self.advantage_positive = bool(advantage_positive)

    @property
    def is_active(self) -> bool:
        """``True`` when CFG actually changes inference (``beta > 1``)."""
        return self.beta > 1.0

    def conditioned_prompt(self, task: str | None) -> str:
        """Prompt for the conditioned forward (advantage tag appended)."""
        return build_acp_tagged_task(task, is_positive=self.advantage_positive)

    def unconditioned_prompt(self, task: str | None) -> str:
        """Prompt for the unconditioned forward (no advantage tag)."""
        return build_unconditioned_task(task)

    def combine(self, v_cond, v_uncond):
        """Compute the CFG-guided velocity given the two predictions.

        ``v_cond`` and ``v_uncond`` may be any framework's tensor type
        (torch.Tensor, numpy.ndarray, jax.Array). The combination is a
        plain affine ``(1 - beta) * v_uncond + beta * v_cond`` that any
        of these dispatch through their ``__add__`` / ``__mul__``.

        For ``beta == 1.0`` this returns ``v_cond`` unchanged so the
        unconditioned forward can be skipped entirely upstream.
        """
        if not self.is_active:
            return v_cond
        return v_uncond + self.beta * (v_cond - v_uncond)
