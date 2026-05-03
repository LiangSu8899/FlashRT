r"""ACP prompt hook for advantage-conditioned policy training (JAX).

Mirrors the PyTorch ``training.rl.acp_hook.ACPPromptHook`` 1-for-1
on the prompt-mutation side. Both stacks consume the SAME shared
primitive (:func:`flash_rt.core.rl.acp_tags.build_acp_tagged_task`)
so the resulting ``batch["task"]`` strings are byte-identical given
the same ``(tasks, indicators, seed)`` triple.

Differences from the PyTorch hook:

* Indicators are validated as ``np.ndarray`` (PyTorch hook accepts
  ``torch.Tensor``). Both stacks check the same constraints: 1-D,
  integer dtype, values in {0, 1}, length matches ``len(tasks)``.
* The hook does not own a torch RNG; it uses ``random.Random(seed)``,
  the same as PyTorch (``training.rl.acp_hook.ACPPromptHook`` line 84).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from flash_rt.core.rl.acp_tags import build_acp_tagged_task


def _extract_indicators(values: Any, batch_size: int) -> list[bool]:
    """Validate + decode a 1-D 0/1 ``int`` indicator array.

    Refuses bool / floating arrays so that ``True/False`` or
    ``0.0/1.0`` inputs cannot silently shift the meaning of
    ``acp_indicator``. Mirrors the PyTorch
    :func:`training.rl.acp_hook._extract_indicators` 1-for-1.
    """
    arr = np.asarray(values)
    if arr.dtype == bool or np.issubdtype(arr.dtype, np.floating):
        raise TypeError(
            "ACP indicator must be integer 0/1, got non-integer array type."
        )
    if arr.ndim != 1:
        raise TypeError(
            f"ACP indicator array must be 1D, got shape={tuple(arr.shape)}."
        )
    if arr.shape[0] != batch_size:
        raise ValueError(
            f"ACP batch size mismatch: expected {batch_size}, got {arr.shape[0]}."
        )
    parsed = arr.tolist()
    if any(v not in (0, 1) for v in parsed):
        bad = [v for v in parsed if v not in (0, 1)][0]
        raise ValueError(f"ACP indicator must be 0 or 1, got {bad}.")
    return [v == 1 for v in parsed]


class JaxACPPromptHook:
    r"""Inject the advantage tag into a training batch's task prompts.

    Args mirror :class:`training.rl.acp_hook.ACPPromptHook` exactly so
    a parity test can construct equivalent hooks on both sides:

    Args:
        indicator_field: Key in the batch dict carrying the int 0/1
            ``np.ndarray`` produced by ``value_infer``.
        dropout: Probability of skipping the tag injection per sample.
            Default ``0.30`` matches the π\*0.6 paper / openpi-compiler.
        seed: RNG seed for reproducible dropout patterns.
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
                ``list[str]``) and the indicator array under
                ``self.indicator_field``.
            step: Current training step. Currently unused; reserved
                for a future schedule (e.g. linear-warmup of dropout).

        Returns:
            The (mutated) batch — the input dict object itself.
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


def build_jax_acp_hook(
    enable: bool = True,
    *,
    indicator_field: str = "acp_indicator",
    dropout: float = 0.3,
    seed: int | None = None,
) -> JaxACPPromptHook | None:
    """Convenience constructor — returns ``None`` when ``enable=False``."""
    if not enable:
        return None
    return JaxACPPromptHook(
        indicator_field=indicator_field, dropout=dropout, seed=seed,
    )
