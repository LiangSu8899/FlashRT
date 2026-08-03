"""Contract tests for the quantize-on-adopt door and the moe_experts
family — everything checkable without a GPU: discovery is by shape
contract, refusals are named, and the envelope mirrors the kernel's own
checks instead of inventing walls.
"""

from __future__ import annotations

import pytest
import torch

from flash_rt.structures.impls.moe_experts.nvfp4_dynamic import (
    SUPPORT, check_experts)
from flash_rt.structures.quantize_on_adopt import (
    _is_moe_expert_bank, quantize_on_adopt)


class _Bank(torch.nn.Module):
    def __init__(self, e=4, i=32, h=64, dims=3):
        super().__init__()
        gu = torch.zeros(e, 2 * i, h) if dims == 3 else torch.zeros(e, h)
        dn = torch.zeros(e, h, i) if dims == 3 else torch.zeros(e, i)
        self.gate_up_proj = torch.nn.Parameter(gu)
        self.down_proj = torch.nn.Parameter(dn)
        self.act_fn = torch.nn.functional.silu


def test_discovery_accepts_the_shape_contract():
    assert _is_moe_expert_bank(_Bank())


def test_discovery_rejects_wrong_rank_and_mismatch():
    assert not _is_moe_expert_bank(_Bank(dims=2))
    bad = _Bank()
    bad.down_proj = torch.nn.Parameter(torch.zeros(4, 64, 48))  # 2I != gu
    assert not _is_moe_expert_bank(bad)
    assert not _is_moe_expert_bank(torch.nn.Linear(8, 8))


def test_discovery_requires_the_activation():
    bank = _Bank()
    del bank.act_fn
    assert not _is_moe_expert_bank(bank)


def test_check_experts_returns_dims_and_names_refusals():
    e, h, i = check_experts({"gate_up_proj": torch.zeros(4, 64, 128),
                             "down_proj": torch.zeros(4, 128, 32)})
    assert (e, h, i) == (4, 128, 32)
    with pytest.raises(ValueError, match="inconsistent"):
        check_experts({"gate_up_proj": torch.zeros(4, 64, 128),
                       "down_proj": torch.zeros(4, 128, 48)})
    with pytest.raises(ValueError, match="multiple of"):
        check_experts({"gate_up_proj": torch.zeros(4, 40, 120),
                       "down_proj": torch.zeros(4, 120, 20)})


def test_envelope_mirrors_kernel_not_invented():
    # the only contraction-dim wall is the kernel's own K%16; no upper
    # bounds — the adoption path serves whatever the checkpoint holds
    assert SUPPORT["K"] == {"min": 16, "multiple_of": 16}
    assert "max" not in SUPPORT["N"]


def test_unknown_format_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown quantize-on-adopt"):
        quantize_on_adopt(torch.nn.Linear(8, 8), "made_up")


def test_bankless_model_is_a_named_refusal():
    with pytest.raises(ValueError, match="no expert banks"):
        quantize_on_adopt(torch.nn.Sequential(torch.nn.Linear(8, 8)))
