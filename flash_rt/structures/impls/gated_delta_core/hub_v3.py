"""Hub v3 executable forms for ``gated_delta_core``."""

from __future__ import annotations

import torch

from ...guard import PROCEED, GuardedSeam
from .. import hub_kernel


class HubV3GatedDeltaCore(GuardedSeam, torch.nn.Module):
    """Single-token H=32/48, D=128 recurrence with explicit state output."""

    def __init__(self, sample: torch.Tensor):
        super().__init__()
        if sample.dtype != torch.bfloat16:
            raise ValueError("gated_delta_core v3 requires BF16 Q/K/V")
        if sample.ndim != 4 or sample.shape[0] != 1 \
                or sample.shape[1] != 1 \
                or sample.shape[2] not in (32, 48) \
                or sample.shape[3] != 128:
            raise ValueError(
                "gated_delta_core v3 requires Q shape "
                "(1,1,H,128) with H=32 or H=48; the published "
                "sequence API has no explicit state output")
        if not sample.is_contiguous():
            raise ValueError("gated_delta_core v3 requires contiguous Q/K/V")
        self.heads = int(sample.shape[2])
        self._ops = hub_kernel("flashrt/gated-delta-attention", ">=3")
        self.register_buffer(
            "_state_out",
            torch.empty(
                1, self.heads, 128, 128,
                device=sample.device, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "_out",
            torch.empty(
                1, self.heads, 128,
                device=sample.device, dtype=torch.bfloat16),
            persistent=False,
        )
        self._frt_arm(
            dtypes=(torch.bfloat16,), device=sample.device, k=128,
            rows=self.heads)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor | None,
        *,
        output_final_state: bool,
        use_qk_l2norm: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        admitted = self._frt_admit(query)
        if admitted is not PROCEED:
            return admitted
        if query.ndim != 4 or query.shape[0] != 1 \
                or query.shape[1:] != (1, self.heads, 128):
            raise ValueError(
                "gated_delta_core v3 query shape moved after binding")
        if key.shape != query.shape or value.shape != query.shape:
            raise ValueError("gated_delta_core v3 Q/K/V shapes differ")
        if not (query.is_contiguous() and key.is_contiguous()
                and value.is_contiguous()):
            raise ValueError("gated_delta_core v3 requires contiguous Q/K/V")
        if log_decay.shape != query.shape[:3] \
                or beta.shape != log_decay.shape:
            raise ValueError("gated_delta_core v3 gating shapes differ")
        if log_decay.dtype != torch.bfloat16 \
                or beta.dtype != torch.bfloat16:
            raise ValueError(
                "gated_delta_core v3 requires BF16 log-decay and beta")
        if state is None or state.shape != (1, self.heads, 128, 128):
            raise ValueError("gated_delta_core v3 state shape differs")
        if state.dtype != torch.bfloat16 or not state.is_contiguous():
            raise ValueError(
                "gated_delta_core v3 requires contiguous BF16 state")
        # One custom op. The caller's state is read-only and the final state is
        # written into graph-stable storage for snapshot and rollback.
        out, state_out = self._ops.gated_delta_recurrent_inout_bf16(
            query[:, 0], key[:, 0], value[:, 0],
            log_decay[:, 0], beta[:, 0], state,
            use_qk_l2norm=use_qk_l2norm,
            state_out=self._state_out,
            out=self._out,
        )
        return out[:, None], state_out if output_final_state else None


def bind_gated_delta_core(sample: dict[str, torch.Tensor]):
    """Bind v3 decode recurrence and launch the observed real sample once."""
    core = HubV3GatedDeltaCore(sample["query"])
    with torch.no_grad():
        core(
            sample["query"], sample["key"], sample["value"],
            sample["g"], sample["beta"], sample.get("state"),
            output_final_state=bool(sample.get("output_final_state", True)),
            use_qk_l2norm=bool(sample.get("use_qk_l2norm", True)),
        )
    guard = core._frt_guard
    if guard is not None:
        guard.calls = 0
    return core
