r"""ACP prompt hook for advantage-conditioned policy training.

For each sample in a training batch:

* with probability ``(1 - dropout)`` append ``"Advantage: positive"``
  or ``"Advantage: negative"`` to ``batch["task"]`` based on the
  ``acp_indicator`` column written by ``value_infer``;
* with probability ``dropout`` leave ``batch["task"]`` unchanged
  — these "unconditioned" samples are what classifier-free guidance
  later combines against at test time (Appendix E).

Default ``dropout = 0.30`` matches the openpi-compiler reference and
the π\*0.6 paper recommendation. The hook reuses the same
``build_acp_tagged_task`` helper as the inference path
(``flash_rt/core/rl/acp_tags.py``) so the prompt format is
byte-identical between train and serve.

Source: ported from ``openpi-compiler/RL/recap/acp_hook.py``
(2026-04-25, verbatim semantics; only the import path changes from
``recap.acp_tags`` to ``flash_rt.core.rl.acp_tags``).
"""

from __future__ import annotations

import random
from typing import Any

import torch

from flash_rt.core.rl.acp_tags import build_acp_tagged_task


def _extract_indicators(values: Any, batch_size: int) -> list[bool]:
    """Validate + decode a 1-D 0/1 ``int`` tensor of indicators.

    Refuses bool / floating tensors so that "True/False" or "0.0/1.0"
    inputs cannot silently shift the meaning of ``acp_indicator``.
    """
    if not isinstance(values, torch.Tensor):
        raise TypeError("ACP indicator must be a torch.Tensor.")
    if values.dtype == torch.bool or values.dtype.is_floating_point:
        raise TypeError(
            "ACP indicator must be integer 0/1, got non-integer tensor type."
        )
    if values.ndim != 1:
        raise TypeError(
            f"ACP indicator tensor must be 1D, got shape={tuple(values.shape)}."
        )
    if values.shape[0] != batch_size:
        raise ValueError(
            f"ACP batch size mismatch: expected {batch_size}, got {values.shape[0]}."
        )

    parsed = values.detach().cpu().tolist()
    if any(v not in (0, 1) for v in parsed):
        bad = [v for v in parsed if v not in (0, 1)][0]
        raise ValueError(f"ACP indicator must be 0 or 1, got {bad}.")
    return [v == 1 for v in parsed]


class ACPPromptHook:
    """Inject the advantage tag into a training batch's task prompts.

    Args:
        indicator_field: Key in the batch dict carrying the int 0/1
            tensor produced by ``value_infer``.
        dropout: Probability of skipping the tag injection per sample
            — those samples become the unconditioned distribution that
            classifier-free guidance combines against at inference.
            Default ``0.30`` matches openpi-compiler / paper.
        seed: Optional RNG seed for reproducible dropout patterns.
    """

    def __init__(
        self,
        indicator_field: str = "acp_indicator",
        dropout: float = 0.3,
        seed: int | None = None,
    ):
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        self.indicator_field = indicator_field
        self.dropout = float(dropout)
        self.rng = random.Random(0 if seed is None else seed)

    def _resolve_indicators(self, batch: dict[str, Any], batch_size: int) -> list[bool]:
        if self.indicator_field not in batch:
            raise KeyError(
                f"ACP indicator field '{self.indicator_field}' is missing from batch."
            )
        return _extract_indicators(batch[self.indicator_field], batch_size)

    def __call__(self, batch: dict[str, Any], step: int = 0) -> dict[str, Any]:
        """Mutate ``batch["task"]`` in place with ACP tags.

        Args:
            batch: Training batch dict. Must contain ``"task"`` (a
                ``list[str]``) and the indicator tensor under
                ``self.indicator_field``.
            step: Current training step. Currently unused; reserved
                for a future schedule (e.g. linear-warmup of dropout).

        Returns:
            The (mutated) batch — the input dict object itself, so
            callers can either ignore the return value or chain hooks.
        """
        if not isinstance(batch, dict):
            raise TypeError(f"ACP batch must be dict, got {type(batch).__name__}.")
        if "task" not in batch:
            raise KeyError("ACP requires 'task' in batch.")

        tasks = batch["task"]
        if not isinstance(tasks, list):
            raise TypeError(
                f"ACP batch['task'] must be list[str], got {type(tasks).__name__}."
            )

        indicators = self._resolve_indicators(batch, len(tasks))

        conditioned: list[str] = []
        for task, is_positive in zip(tasks, indicators, strict=True):
            if self.dropout > 0.0 and self.rng.random() < self.dropout:
                conditioned.append(task)
                continue
            conditioned.append(build_acp_tagged_task(task, is_positive=is_positive))
        batch["task"] = conditioned
        return batch


def build_acp_hook(
    enable: bool = True,
    *,
    indicator_field: str = "acp_indicator",
    dropout: float = 0.3,
    seed: int | None = None,
) -> ACPPromptHook | None:
    """Convenience constructor — returns ``None`` when ``enable=False``.

    Mirrors openpi-compiler's ``build_acp_hook``: the W7 train_policy
    driver checks ``hook is not None`` to decide whether to apply ACP
    each step, so flipping ``enable`` between True/False at the config
    level requires no other changes.
    """
    if not enable:
        return None
    return ACPPromptHook(
        indicator_field=indicator_field, dropout=dropout, seed=seed
    )
