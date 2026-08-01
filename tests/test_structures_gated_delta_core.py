from __future__ import annotations

import torch
from torch import nn

from flash_rt.structures.adapters.transformers_gated_delta import (
    TransformersGatedDeltaAdapter,
)
from flash_rt.structures.catalog.gated_delta_core.reference import (
    gated_delta_core_ref,
)
from flash_rt.structures.registry import load


def test_gated_delta_reference_carries_state_across_calls():
    torch.manual_seed(4)
    q = torch.randn(1, 3, 2, 4, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = -torch.rand(1, 3, 2, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn_like(g))
    whole, state = gated_delta_core_ref(q, k, v, g, beta)
    first, state1 = gated_delta_core_ref(
        q[:, :2], k[:, :2], v[:, :2], g[:, :2], beta[:, :2])
    last, state2 = gated_delta_core_ref(
        q[:, 2:], k[:, 2:], v[:, 2:], g[:, 2:], beta[:, 2:], state1)
    # The public state boundary is BF16, so splitting the call introduces one
    # intentional state cast that a single sequence call does not.
    torch.testing.assert_close(
        torch.cat((first, last), dim=1), whole, rtol=5e-2, atol=1e-4)
    torch.testing.assert_close(state2, state, rtol=5e-2, atol=1e-4)


def test_gated_delta_catalog_declares_state_boundary():
    spec = load("gated_delta_core")
    inputs = {entry["name"] for entry in spec.boundary["inputs"]}
    outputs = {entry["name"] for entry in spec.boundary["outputs"]}
    assert {"q", "k", "v", "log_decay", "beta", "state"} <= inputs
    assert {"out", "final_state"} <= outputs


class _HostCore:
    def __call__(
        self, query, key, value, g, beta, *, initial_state=None,
        output_final_state=False, use_qk_l2norm_in_kernel=True, **kwargs,
    ):
        del kwargs
        out, state = gated_delta_core_ref(
            query, key, value, g, beta, initial_state,
            qk_l2norm=use_qk_l2norm_in_kernel)
        return out, state if output_final_state else None


class _FakeLinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_v_heads = 48
        self.head_k_dim = 128
        self.head_v_dim = 128
        self.recurrent_gated_delta_rule = _HostCore()
        self.chunk_gated_delta_rule = _HostCore()

    def forward(self, q, k, v, g, beta, state=None):
        return self.recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=state,
            output_final_state=True, use_qk_l2norm_in_kernel=True)


class _BoundCore(nn.Module):
    def forward(
        self, query, key, value, g, beta, state, *,
        output_final_state, use_qk_l2norm,
    ):
        out, state = gated_delta_core_ref(
            query, key, value, g, beta, state,
            qk_l2norm=use_qk_l2norm)
        return out, state if output_final_state else None


def test_transformers_gated_delta_adapter_routes_and_reverts(monkeypatch):
    host = nn.ModuleDict({"linear_attn": _FakeLinearAttention()}).eval()
    q = torch.randn(1, 1, 48, 128, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = -torch.rand(1, 1, 48, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn_like(g))
    initial_state = torch.zeros(1, 48, 128, 128, dtype=torch.bfloat16)
    expected = host.linear_attn(q, k, v, g, beta, initial_state)
    original = host.linear_attn.recurrent_gated_delta_rule

    monkeypatch.setitem(
        TransformersGatedDeltaAdapter.__call__.__globals__,
        "bind_gated_delta_core", lambda row: _BoundCore())
    result = TransformersGatedDeltaAdapter()(
        host, lambda: host.linear_attn(q, k, v, g, beta, initial_state))
    assert result is not None and result["observed"]
    got = host.linear_attn(q, k, v, g, beta, initial_state)
    torch.testing.assert_close(got[0], expected[0])
    torch.testing.assert_close(got[1], expected[1])
    result["revert"][0]()
    assert host.linear_attn.recurrent_gated_delta_rule is original


def test_transformers_gated_delta_adapter_rejects_other_head_shapes():
    host = nn.ModuleDict({"linear_attn": _FakeLinearAttention()}).eval()
    host.linear_attn.num_v_heads = 8
    assert TransformersGatedDeltaAdapter()(host, lambda: None) is None


def test_transformers_gated_delta_adapter_accepts_second_head_profile(
        monkeypatch):
    host = nn.ModuleDict({"linear_attn": _FakeLinearAttention()}).eval()
    host.linear_attn.num_v_heads = 32
    q = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = -torch.rand(1, 1, 32, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn_like(g))
    initial_state = torch.zeros(1, 32, 128, 128, dtype=torch.bfloat16)
    monkeypatch.setitem(
        TransformersGatedDeltaAdapter.__call__.__globals__,
        "bind_gated_delta_core", lambda row: _BoundCore())

    result = TransformersGatedDeltaAdapter()(
        host, lambda: host.linear_attn(q, k, v, g, beta, initial_state))

    assert result is not None and result["observed"]
    out, state = host.linear_attn(q, k, v, g, beta, initial_state)
    assert out.shape == q.shape
    assert state.shape == (1, 32, 128, 128)
