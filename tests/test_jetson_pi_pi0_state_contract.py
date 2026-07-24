import ctypes

import numpy as np
import pytest

from flash_rt.frontends.jetson_pi import pi0


def test_pi0_frontend_rejects_more_than_three_views():
    with pytest.raises(ValueError, match=r"num_views must be in \[1, 3\]"):
        pi0.Pi0JetsonPiFrontend(
            "/models/pi0.gguf",
            mmproj_path="/models/pi0-mmproj.gguf",
            num_views=4,
            action_steps=50,
            action_dim=32,
        )


def test_pi0_frontend_passes_policy_state_without_action_padding(monkeypatch):
    received_states = []

    @pi0._SetInputFn
    def set_input(_self, port, data, size, _stream):
        if port == pi0.PORT_STATE:
            count = size // ctypes.sizeof(ctypes.c_float)
            values = (ctypes.c_float * count).from_address(data)
            received_states.append(np.ctypeslib.as_array(values).copy())
        return 0

    @pi0._GetOutputFn
    def get_output(_self, port, out, capacity, written, _stream):
        assert port == pi0.PORT_ACTIONS
        assert capacity == 32 * ctypes.sizeof(ctypes.c_float)
        ctypes.memset(out, 0, capacity)
        written[0] = capacity
        return 0

    @pi0._StepFn
    def step(_self):
        return 0

    @pi0._LastErrorFn
    def last_error(_self):
        return b""

    frontend = object.__new__(pi0.Pi0JetsonPiFrontend)
    frontend.num_views = 1
    frontend.image_height = 1
    frontend.image_width = 1
    frontend.action_steps = 1
    frontend.action_dim = 32
    frontend._prompt = b"pick up the block"
    frontend._model = pi0.FrtModelRuntimeV1()
    frontend._model.self = ctypes.c_void_p(1)
    frontend._model.verbs = pi0._FrtVerbsV1(
        ctypes.sizeof(pi0._FrtVerbsV1), 0, set_input, get_output,
        pi0._PrepareFn(), step, last_error)
    frontend._callback_keepalive = (set_input, get_output, step, last_error)

    state = np.arange(8, dtype=np.float32)
    observation = {
        "images": [np.zeros((1, 1, 3), dtype=np.uint8)],
        "state": state,
    }

    frontend.infer(observation)
    monkeypatch.setattr(pi0, "_run_generic_stage", lambda _model, _name: 0)
    frontend.context(observation)

    assert len(received_states) == 2
    assert np.array_equal(received_states[0], state)
    assert np.array_equal(received_states[1], state)

    with pytest.raises(ValueError, match="at least one value"):
        frontend.infer({**observation, "state": np.empty(0, dtype=np.float32)})
    with pytest.raises(ValueError, match="must be finite"):
        frontend.infer({**observation, "state": np.array([np.nan], dtype=np.float32)})
    with pytest.raises(ValueError, match="must be finite"):
        frontend.context({
            **observation,
            "state": np.array([np.inf], dtype=np.float32),
        })
