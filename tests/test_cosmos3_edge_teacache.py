"""TeaCache schedule and graph-capture contract tests."""

from __future__ import annotations

import pytest

try:
    from flash_rt.models.cosmos3_edge.pipeline_thor import CosmosEdgeThor
except Exception as exc:  # pragma: no cover - extension/build dependent
    pytest.skip(f"Cosmos3-Edge pipeline is not importable: {exc}", allow_module_level=True)


def _bare_engine(*, num_steps: int = 30) -> CosmosEdgeThor:
    engine = object.__new__(CosmosEdgeThor)
    engine.num_steps = num_steps
    engine.graph = None
    engine.compute_steps = None
    return engine


def test_teacache_schedule_is_canonical_and_validated_before_capture():
    engine = _bare_engine()
    engine.set_teacache([29, 3, 0, 3])
    assert engine.compute_steps == frozenset({0, 3, 29})

    engine.set_teacache(None)
    assert engine.compute_steps is None

    for invalid in ([], [1, 2], [0, 30], [-1, 0]):
        with pytest.raises(ValueError, match="invalid TeaCache compute schedule"):
            engine.set_teacache(invalid)


def test_teacache_schedule_is_frozen_after_capture():
    engine = _bare_engine()
    engine.set_teacache([0, 15, 29])
    before = engine.compute_steps
    engine.graph = object()

    with pytest.raises(RuntimeError, match="before graph capture"):
        engine.set_teacache([0, 29])
    assert engine.compute_steps == before


def test_teacache_run_loop_computes_selected_steps_and_advances_every_step():
    engine = _bare_engine(num_steps=5)
    engine.compute_steps = frozenset({0, 3})
    engine.latent = object()
    engine.velocity = object()
    computed: list[int] = []
    advanced: list[tuple[int, object]] = []

    class _UniPC:
        prev_m1 = object()

        @staticmethod
        def step(_latent, velocity, step):
            advanced.append((step, velocity))

    engine.unipc = _UniPC()

    def _forward(step, _latent):
        computed.append(step)
        return ("computed", step)

    engine.forward_step = _forward
    assert engine.run_loop() is engine.latent
    assert computed == [0, 3]
    assert [step for step, _ in advanced] == [0, 1, 2, 3, 4]
    assert advanced[1][1] is engine.velocity
    assert advanced[2][1] is engine.velocity
    assert advanced[4][1] is engine.velocity
