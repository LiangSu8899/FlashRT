"""Shared ``calibrate()`` shim for frontends that do not yet implement
a native multi-sample calibration loop.

Used by Thor frontends (pi0_thor / pi05_thor / pi0fast / groot_thor),
their JAX counterparts, and groot_rtx — all of which currently rely on
the implicit "first infer triggers recalibration" mechanism. The shim
turns that into an explicit public API that matches pi0_rtx / pi05_rtx
in shape, so user code is portable across frontends.

Behaviour:
    N == 1 : run one forward via ``frontend.infer(obs)`` and discard the
             output. This fires whatever implicit calibration hook the
             frontend has (Thor's ``_recalibrate_with_real_data``
             auto-runs in the first infer).
    N >= 2 : raise NotImplementedError with a clear pointer to the RTX
             frontends which support multi-sample today.

When a frontend later grows a native multi-sample path it should define
its own ``calibrate`` method and stop calling this helper.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


_THOR_MULTI_SAMPLE_MESSAGE = (
    "Multi-sample (N>=2) dataset calibration is not yet implemented for "
    "this frontend. Today it is only supported on the RTX torch frontends "
    "(Pi0TorchFrontendRtx, Pi05TorchFrontendRtx). Either pass a single "
    "observation to fall back to the implicit recalibration path, or run "
    "calibration on the RTX frontend if your deployment environment "
    "permits. Thor multi-sample support is planned for a future release."
)


def implicit_calibrate(
    frontend: Any,
    observations: Iterable[Any],
    *,
    percentile: float = 99.9,
    max_samples: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """Shim: force a single implicit recalibration via ``infer(obs_list[0])``.

    Raises NotImplementedError for N >= 2 (Thor multi-sample path is TODO).
    """
    if isinstance(observations, dict):
        obs_list = [observations]
    elif isinstance(observations, list):
        obs_list = observations
    else:
        obs_list = list(observations)
    if max_samples is not None:
        obs_list = obs_list[:max_samples]
    n = len(obs_list)
    if n == 0:
        raise ValueError("observations must contain at least 1 sample")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")

    if n > 1:
        raise NotImplementedError(_THOR_MULTI_SAMPLE_MESSAGE)

    # Trigger implicit recalibration by running one inference and
    # discarding the result. The frontend's first-infer path will set
    # its ``_real_data_calibrated`` flag (or equivalent).
    frontend.infer(obs_list[0])
