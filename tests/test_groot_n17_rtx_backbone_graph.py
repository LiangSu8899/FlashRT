from __future__ import annotations

import torch

from flash_rt.frontends.torch.groot_n17_rtx import GrootN17TorchFrontendRtx
from flash_rt.frontends.torch.groot_n17_rtx_fp8 import (
    GrootN17TorchFrontendRtxFP8,
)


def test_fp8_infer_replays_backbone_before_action_graph(monkeypatch):
    frontend = object.__new__(GrootN17TorchFrontendRtxFP8)
    features = object()
    calls = []

    def run_backbone_graph(aux):
        calls.append(("backbone", aux))
        return features

    def action_infer(self, state, **kwargs):
        calls.append(("action", self._backbone_features, state, kwargs))
        return "actions"

    frontend.run_backbone_graph = run_backbone_graph
    monkeypatch.setattr(GrootN17TorchFrontendRtx, "infer", action_infer)

    state = torch.empty(1, 1, 132)
    aux = {"pixel_features": object()}
    result = frontend.infer(state, aux=aux)

    assert result == "actions"
    assert calls[0] == ("backbone", aux)
    assert calls[1][0:3] == ("action", features, state)
    assert calls[1][3]["use_dit_graph"] is True


def test_fp8_infer_without_aux_reuses_prompt_backbone(monkeypatch):
    frontend = object.__new__(GrootN17TorchFrontendRtxFP8)
    features = object()
    frontend._backbone_features = features

    def fail_if_replayed(aux):
        raise AssertionError("backbone graph should not replay without aux")

    def action_infer(self, state, **kwargs):
        assert self._backbone_features is features
        return "actions"

    frontend.run_backbone_graph = fail_if_replayed
    monkeypatch.setattr(GrootN17TorchFrontendRtx, "infer", action_infer)

    assert frontend.infer(torch.empty(1, 1, 132)) == "actions"


def test_run_backbone_graph_refreshes_persistent_inputs():
    frontend = object.__new__(GrootN17TorchFrontendRtxFP8)
    frontend.device = torch.device("cpu")
    frontend._S_vit = 2
    frontend.Se = 3
    frontend._kbb_vit_h = torch.zeros(2, 1024, dtype=torch.float16)
    frontend._kbb_llm_h = torch.zeros(3, 2048, dtype=torch.float16)
    frontend._kbb_vlsa_h = torch.ones(3, 2048, dtype=torch.float16)

    class Graph:
        replay_count = 0

        def replay(self):
            self.replay_count += 1

    graph = Graph()
    frontend._kbb_graph = graph
    aux = {
        "pixel_features": torch.full((1, 2, 1024), 2.0),
        "llm_input_embeds": torch.full((1, 3, 2048), 3.0),
    }

    result = frontend.run_backbone_graph(aux)

    assert graph.replay_count == 1
    assert torch.all(frontend._kbb_vit_h == 2)
    assert torch.all(frontend._kbb_llm_h == 3)
    assert result.data_ptr() == frontend._kbb_vlsa_h.data_ptr()
    assert result.shape == (1, 3, 2048)
