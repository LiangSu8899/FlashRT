import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from serving.higgs_audio_agent.server import build_app


class _Frontend:
    max_new_frames = 1024

    def __init__(self):
        self.calls = []

    def generate_stream(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return iter(())


def test_speech_api_preserves_greedy_default():
    frontend = _Frontend()
    client = TestClient(build_app(frontend, "higgs-test"))

    response = client.post("/v1/audio/speech", json={"input": "hello"})

    assert response.status_code == 200
    assert frontend.calls == [
        ("hello", {"system": None, "temperature": 0.0, "seed": None})
    ]


def test_speech_api_forwards_sampling_fallback():
    frontend = _Frontend()
    client = TestClient(build_app(frontend, "higgs-test"))

    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "temperature": 1.0, "seed": 1234},
    )

    assert response.status_code == 200
    assert frontend.calls == [
        ("hello", {"system": None, "temperature": 1.0, "seed": 1234})
    ]
