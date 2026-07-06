"""Pi0.5 registered stage plans.

This module contains declarations only. Extra graph capture is provided by
modules such as ``flash_rt.subgraphs.pi05.context_action`` and must be enabled
before the pipeline captures graphs.
"""

from __future__ import annotations

from flash_rt.subgraphs.stage_plan import Stage, StagePlan, register_stage_plan


def full(**_) -> StagePlan:
    return StagePlan.full(graph="infer", stage="infer")


def context_action(**_) -> StagePlan:
    return StagePlan((
        Stage("context", graph="context"),
        Stage("action", graph="decode_only", after=("context",)),
    ), name="context_action", metadata={
        "granularity": "context/action",
        "context": "prompt_copy+vision_encoder+transformer_encoder",
        "action": "transformer_decoder",
    })


def context_rtc_prefix_action(*, prefix_len: int = 1, **_) -> StagePlan:
    prefix = int(prefix_len)
    if prefix < 0:
        raise ValueError("prefix_len must be >= 0")
    return StagePlan((
        Stage("context", graph="context"),
        Stage("action", graph="decode_rtc_prefix", after=("context",)),
    ), name="context_rtc_prefix_action", metadata={
        "granularity": "context/rtc_prefix_action",
        "context": "prompt_copy+vision_encoder+transformer_encoder",
        "action": "transformer_decoder with fixed previous-chunk prefix",
        "rtc_prefix_len": prefix,
    })


def context_rtc_vjp_guided_action(**_) -> StagePlan:
    return StagePlan((
        Stage("context", graph="context"),
        Stage("action", graph="decode_rtc_vjp_guided", after=("context",)),
    ), name="context_rtc_vjp_guided_action", metadata={
        "granularity": "context/rtc_vjp_guided_action",
        "context": "prompt_copy+vision_encoder+transformer_encoder",
        "action": "transformer_decoder with denoiser VJP guidance",
        "requires": "producer DenoiserVjpProvider",
    })


def vlash(*, lookahead_steps: int = 1, **_) -> StagePlan:
    """Context/action cut conditioned on an upper-runtime projected state."""
    steps = int(lookahead_steps)
    if steps < 0:
        raise ValueError("lookahead_steps must be >= 0")
    return StagePlan((
        Stage("context", graph="context"),
        Stage("action", graph="decode_only", after=("context",)),
    ), name="vlash", metadata={
        "granularity": "context/vlash_action",
        "context": "prompt_copy+vision_encoder+transformer_encoder",
        "action": "transformer_decoder",
        "state_projection": "start_state+sum(next_actions)",
        "lookahead_steps": steps,
    })


register_stage_plan("full", full, model="pi05", replace=True)
register_stage_plan("context_action", context_action, model="pi05",
                    replace=True)
register_stage_plan("context_rtc_prefix_action", context_rtc_prefix_action,
                    model="pi05", replace=True)
register_stage_plan("context_rtc_vjp_guided_action",
                    context_rtc_vjp_guided_action, model="pi05",
                    replace=True)
register_stage_plan("vlash", vlash, model="pi05", replace=True)


__all__ = [
    "full",
    "context_action",
    "context_rtc_prefix_action",
    "context_rtc_vjp_guided_action",
    "vlash",
]
