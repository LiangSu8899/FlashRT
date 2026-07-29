from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flash_rt.structures.impls.attention_core import fa2_seqused


class _FakePackedKVAttention:
    def __init__(
        self,
        plan,
        q_shape,
        kv_heads,
        dtype,
        device,
        prefix_kv=None,
        scratch=None,
    ):
        del q_shape, kv_heads, dtype, device, prefix_kv
        self.plan = plan
        self._scratch = scratch or SimpleNamespace()


def _capture(head_dim: int):
    q = torch.randn(1, 4, 7, head_dim, dtype=torch.bfloat16)
    key = torch.randn(1, 2, 13, head_dim, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    return {
        "q": q,
        "keys": [key, key.clone()],
        "values": [value, value.clone()],
        "mask": None,
    }


@pytest.mark.parametrize("head_dim", [48, 64, 72, 80, 128, 256])
def test_attention_core_admits_production_logical_head_dims(
    monkeypatch, head_dim
):
    monkeypatch.setattr(
        fa2_seqused, "PackedKVAttention", _FakePackedKVAttention
    )
    bound = fa2_seqused.bind_attention_core([_capture(head_dim)])
    assert bound is not None
    modules, update = bound
    assert len(modules) == 1
    assert callable(update)


@pytest.mark.parametrize("head_dim", [7, 44, 264])
def test_attention_core_refuses_unaligned_or_oversized_head_dims(head_dim):
    assert fa2_seqused.bind_attention_core([_capture(head_dim)]) is None
