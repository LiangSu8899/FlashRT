"""Jetson-PI generic GGUF LLM frontend — drives the Jetson-PI llama.cpp LLM
provider through the FlashRT ``frt_model_runtime_v1`` C ABI via ctypes.

This is the Python entry for plain text completion (Pi0 is a VLA and
goes through :mod:`flash_rt.frontends.jetson_pi.pi0`). One raw prompt in,
one generated text blob out per :meth:`generate` call. The caller is
responsible for applying the chat template; the engine only does raw
prompt -> text.

``flash_rt.load_model(framework="jetson_pi", config="llm", ...)`` returns a
:class:`LlmJetsonPiFrontend` directly (it is NOT wrapped in ``VLAModel`` —
LLMs are not VLA and do not take images).
"""

from __future__ import annotations

import ctypes
import json
import os

import numpy as np

from .pi0 import (  # reuse the ctypes mirrors + lib finder
    FrtModelRuntimeV1,
    _FRT_MODEL_RUNTIME_ABI_VERSION,
    _find_lib,
    _run_generic_stage,
)

# Port / stage indices (c_api.h: FRT_LLAMA_CPP_LLM_*)
PORT_PROMPT = 0
PORT_TEXT = 1
PORT_NEXT_TOKEN = 2
PORT_LOGITS = 3
PORT_IS_EOG = 4
PORT_TOKENS = 5
STAGE_INFER = 0
STAGE_RESET = 1
STAGE_PREFILL = 2
STAGE_DECODE = 3


class LlmJetsonPiFrontend:
    """Generic GGUF LLM completion frontend backed by the Jetson-PI provider."""

    def __init__(self, checkpoint, *, backend="cpu", n_ctx=0, n_threads=0,
                 temp=0.8, top_k=40, top_p=0.9, seed=1, max_tokens=512,
                 lib_path=None, **_unused):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.max_tokens = int(max_tokens)
        self._lib_path = _find_lib(lib_path, env_var="FLASHRT_LLM_LIB")
        self._lib = ctypes.CDLL(self._lib_path)

        self._lib.frt_model_runtime_open_v1.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self._lib.frt_model_runtime_open_v1.restype = ctypes.c_int

        config = {
            "model_family": "llm",
            "model_path": str(checkpoint),
            "backend": backend,
            "n_ctx": int(n_ctx),
            "n_threads": int(n_threads),
            "temp": float(temp),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "seed": int(seed),
            "max_tokens": int(max_tokens),
        }
        config_json = json.dumps(config, ensure_ascii=False).encode("utf-8")

        model_ptr = ctypes.c_void_p(0)
        rc = self._lib.frt_model_runtime_open_v1(
            config_json, ctypes.byref(model_ptr))
        if rc != 0 or not model_ptr.value:
            raise RuntimeError(
                f"frt_model_runtime_open_v1 failed for LLM (rc={rc})")

        self._model = FrtModelRuntimeV1.from_address(model_ptr.value)
        if self._model.abi_version != _FRT_MODEL_RUNTIME_ABI_VERSION:
            raise RuntimeError(
                f"frt_model_runtime_v1 abi_version={self._model.abi_version}, "
                f"expected {_FRT_MODEL_RUNTIME_ABI_VERSION}.")
        if self._model.struct_size < ctypes.sizeof(FrtModelRuntimeV1):
            raise RuntimeError(
                f"frt_model_runtime_v1 struct_size={self._model.struct_size} "
                f"< ctypes sizeof {ctypes.sizeof(FrtModelRuntimeV1)}.")
        self._model_ptr = model_ptr

    def generate(self, prompt):
        """Run one whole-prompt completion. Returns the generated text (str)."""
        if self._model is None:
            raise RuntimeError("LlmJetsonPiFrontend is closed")
        if isinstance(prompt, str):
            prompt_bytes = prompt.encode("utf-8")
        else:
            prompt_bytes = bytes(prompt)
        verbs = self._model.verbs
        self_ = self._model.self

        rc = verbs.set_input(self_, PORT_PROMPT, prompt_bytes,
                          len(prompt_bytes), -1)
        self._check(rc, "set_input prompt")
        rc = verbs.step(self_)
        self._check(rc, "step infer")

        written = ctypes.c_uint64(0)
        rc = verbs.get_output(self_, PORT_TEXT, None, 0,
                           ctypes.byref(written), -1)
        if rc != -5 or written.value == 0:
            self._check(rc, "query text output size")
            raise RuntimeError("query text output size returned zero bytes")
        cap = written.value
        out = (ctypes.c_char * cap)()
        written.value = 0
        rc = verbs.get_output(self_, PORT_TEXT, out, cap,
                           ctypes.byref(written), -1)
        self._check(rc, "get_output text")
        return out.raw[:written.value].decode("utf-8", errors="replace")

    def reset(self):
        """Clear the current KV-cache and sampler state."""
        if self._model is None:
            raise RuntimeError("LlmJetsonPiFrontend is closed")
        rc = _run_generic_stage(self._model, "reset")
        self._check(rc, "generic stage reset")

    def prefill(self, prompt=None, *, tokens=None):
        """Start a session from either raw ``prompt`` or int32 ``tokens``."""
        if self._model is None:
            raise RuntimeError("LlmJetsonPiFrontend is closed")
        if (prompt is None) == (tokens is None):
            raise ValueError("exactly one of prompt or tokens is required")
        verbs = self._model.verbs
        if tokens is not None:
            token_array = np.ascontiguousarray(tokens, dtype=np.int32)
            if token_array.ndim != 1 or token_array.size == 0:
                raise ValueError("tokens must be a non-empty 1-D int32 array")
            rc = verbs.set_input(
                self._model.self, PORT_TOKENS, token_array.ctypes.data,
                token_array.nbytes, -1)
            self._check(rc, "set_input tokens")
        else:
            if isinstance(prompt, str):
                prompt_bytes = prompt.encode("utf-8")
            else:
                prompt_bytes = bytes(prompt)
            rc = verbs.set_input(self._model.self, PORT_PROMPT, prompt_bytes,
                              len(prompt_bytes), -1)
            self._check(rc, "set_input prompt")
        rc = _run_generic_stage(self._model, "prefill")
        self._check(rc, "generic stage prefill")
        return self.get_logits()

    def decode(self):
        """Sample and decode one token from the current session."""
        if self._model is None:
            raise RuntimeError("LlmJetsonPiFrontend is closed")
        verbs = self._model.verbs
        rc = _run_generic_stage(self._model, "decode")
        self._check(rc, "generic stage decode")

        next_token = ctypes.c_int32()
        written = ctypes.c_uint64(0)
        rc = verbs.get_output(
            self._model.self, PORT_NEXT_TOKEN, ctypes.byref(next_token),
            ctypes.sizeof(next_token), ctypes.byref(written), -1)
        self._check(rc, "get_output next_token")
        if written.value != ctypes.sizeof(next_token):
            raise RuntimeError(
                f"get_output next_token wrote {written.value} bytes, expected "
                f"{ctypes.sizeof(next_token)}")

        is_eog = ctypes.c_int32()
        written = ctypes.c_uint64(0)
        rc = verbs.get_output(
            self._model.self, PORT_IS_EOG, ctypes.byref(is_eog),
            ctypes.sizeof(is_eog), ctypes.byref(written), -1)
        self._check(rc, "get_output is_eog")
        if written.value != ctypes.sizeof(is_eog):
            raise RuntimeError(
                f"get_output is_eog wrote {written.value} bytes, expected "
                f"{ctypes.sizeof(is_eog)}")

        written = ctypes.c_uint64(0)
        rc = verbs.get_output(self._model.self, PORT_TEXT, None, 0,
                           ctypes.byref(written), -1)
        if rc != -5 or written.value == 0:
            self._check(rc, "query text output size")
            raise RuntimeError("query text output size returned zero bytes")
        cap = written.value
        out = (ctypes.c_char * cap)()
        written.value = 0
        rc = verbs.get_output(self._model.self, PORT_TEXT, out, cap,
                           ctypes.byref(written), -1)
        self._check(rc, "get_output text")
        return {
            "token": int(next_token.value),
            "is_eog": bool(is_eog.value),
            "text": out.raw[:written.value].decode("utf-8", errors="replace"),
        }

    def get_logits(self):
        """Copy the current next-token logits into a NumPy float32 array."""
        if self._model is None:
            raise RuntimeError("LlmJetsonPiFrontend is closed")
        verbs = self._model.verbs
        required = ctypes.c_uint64(0)
        rc = verbs.get_output(
            self._model.self, PORT_LOGITS, None, 0,
            ctypes.byref(required), -1)
        if rc != -5 or required.value == 0:
            self._check(rc, "query logits size")
            raise RuntimeError("query logits size returned no bytes")
        if required.value % np.dtype(np.float32).itemsize != 0:
            raise RuntimeError(
                f"logits byte size {required.value} is not float32-aligned")
        logits = np.empty(
            required.value // np.dtype(np.float32).itemsize, dtype=np.float32)
        written = ctypes.c_uint64(0)
        rc = verbs.get_output(
            self._model.self, PORT_LOGITS, logits.ctypes.data,
            logits.nbytes, ctypes.byref(written), -1)
        self._check(rc, "get_output logits")
        if written.value != logits.nbytes:
            raise RuntimeError(
                f"get_output logits wrote {written.value} bytes, expected "
                f"{logits.nbytes}")
        return logits

    def infer(self, observation, debug=False):
        """VLAModel-predict-parity entry: observation['prompt'] -> {'text': ...}."""
        _ = debug
        prompt = observation.get("prompt") if isinstance(observation, dict) else None
        if prompt is None:
            raise ValueError("observation['prompt'] is required")
        return {"text": self.generate(prompt)}

    def close(self):
        if getattr(self, "_model", None) is not None:
            if self._model.release:
                self._model.release(self._model.owner)
            self._model = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _check(self, rc, what):
        if rc != 0:
            err = b""
            try:
                err = self._model.verbs.last_error(self._model.self) or b""
            except Exception:
                pass
            raise RuntimeError(
                f"{what} failed (rc={rc}): "
                f"{(err.decode(errors='replace') if err else 'no error')}")
