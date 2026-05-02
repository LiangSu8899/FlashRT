"""CPU-only unit tests for ``flash_rt.core.rl`` primitives.

Covers the ACP tag builder and the CFGSampler combine math. No CUDA or
model loading required.
"""

import numpy as np
import pytest

from flash_rt.core.rl import (
    ACP_NEGATIVE_TAG,
    ACP_POSITIVE_TAG,
    CFGSampler,
    build_acp_tagged_task,
    build_unconditioned_task,
)


def test_acp_tag_constants():
    assert ACP_POSITIVE_TAG == "Advantage: positive"
    assert ACP_NEGATIVE_TAG == "Advantage: negative"


def test_build_acp_tagged_task_positive():
    out = build_acp_tagged_task("pick up the cup", is_positive=True)
    assert out == "pick up the cup\nAdvantage: positive"


def test_build_acp_tagged_task_negative():
    out = build_acp_tagged_task("pick up the cup", is_positive=False)
    assert out == "pick up the cup\nAdvantage: negative"


@pytest.mark.parametrize("empty", [None, ""])
def test_build_acp_tagged_task_empty_falls_back_to_tag(empty):
    out = build_acp_tagged_task(empty, is_positive=True)
    assert out == "Advantage: positive"


def test_unconditioned_task_passthrough():
    assert build_unconditioned_task("fold the shirt") == "fold the shirt"
    assert build_unconditioned_task(None) == ""


def test_cfg_sampler_default_is_disabled():
    s = CFGSampler()
    assert s.beta == 1.5
    assert s.is_active is True


def test_cfg_sampler_beta_one_inactive():
    s = CFGSampler(beta=1.0)
    assert s.is_active is False


def test_cfg_sampler_rejects_beta_below_one():
    with pytest.raises(ValueError, match="beta must be"):
        CFGSampler(beta=0.5)


def test_cfg_sampler_prompt_pair():
    s = CFGSampler(beta=1.5)
    assert s.conditioned_prompt("fold") == "fold\nAdvantage: positive"
    assert s.unconditioned_prompt("fold") == "fold"


def test_cfg_sampler_combine_disabled_returns_cond():
    s = CFGSampler(beta=1.0)
    v_cond = np.array([1.0, 2.0, 3.0])
    v_uncond = np.array([0.0, 0.0, 0.0])
    out = s.combine(v_cond, v_uncond)
    np.testing.assert_array_equal(out, v_cond)


def test_cfg_sampler_combine_matches_formula():
    """v_guided = (1 - beta) * v_uncond + beta * v_cond."""
    s = CFGSampler(beta=2.0)
    v_cond = np.array([1.0, 2.0, 3.0])
    v_uncond = np.array([0.5, 0.5, 0.5])
    expected = -v_uncond + 2.0 * v_cond  # (1-2)*v_u + 2*v_c
    np.testing.assert_allclose(s.combine(v_cond, v_uncond), expected)


def test_cfg_sampler_combine_at_beta_1p5():
    s = CFGSampler(beta=1.5)
    v_cond = np.array([1.0, 0.0])
    v_uncond = np.array([0.0, 1.0])
    # 1.5 * v_cond + (1 - 1.5) * v_uncond = 1.5 * [1,0] - 0.5 * [0,1] = [1.5, -0.5]
    np.testing.assert_allclose(s.combine(v_cond, v_uncond), [1.5, -0.5])
