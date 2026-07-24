"""Jetson-PI Pi0 frontend — drives the Jetson-PI llama.cpp/GGML Pi0 provider
through the FlashRT ``frt_model_runtime_v1`` C ABI via ctypes.

It dlopens the SHARED provider library
(``libflashrt_cpp_llama_cpp_provider_c.so``)
built under ``FLASHRT_CPP_WITH_JETSON_PI``, opens a Pi0 runtime through
``frt_model_runtime_open_v1``, and drives one whole-model Pi0 infer per
``infer(observation)`` call.

The frontend intentionally does no action unnormalization / LIBERO slicing:
it returns the raw ``action_steps × action_dim`` action chunk the model
produces. Higher layers (or the caller) post-process as needed.

Memory: the Jetson-PI engine copies all inputs on ``set_input`` (see
``jetson_pi_engine.cpp``), so numpy arrays need not be kept alive past the
``infer`` call.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os

import numpy as np

# ---- frt_image_view ctypes mirror (matches runtime/include/flashrt/model_runtime.h) ----

FRT_RT_PIXEL_RGB8 = 0


class FrtImageView(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("pixel_format", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("bytes", ctypes.c_uint64),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("stride_bytes", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("timestamp_ns", ctypes.c_uint64),
    ]


# ---- frt_model_runtime_v1 + GENERIC_STAGE_PLAN_V1 ctypes mirrors -----------

_SetInputFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int)
_GetOutputFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_int)
_PrepareFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64)
_StepFn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_LastErrorFn = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p)
_RetainReleaseFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _FrtVerbsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("set_input", _SetInputFn),
        ("get_output", _GetOutputFn),
        ("prepare", _PrepareFn),
        ("step", _StepFn),
        ("last_error", _LastErrorFn),
    ]


class FrtModelRuntimeV1(ctypes.Structure):
    pass


_QueryExtensionFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(FrtModelRuntimeV1), ctypes.c_uint64,
    ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
_RunOpaqueFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)


class _GenericStageDescV1(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("executor_kind", ctypes.c_uint32),
        ("executor_ref", ctypes.c_uint32),
        ("n_after", ctypes.c_uint32),
        ("after", ctypes.POINTER(ctypes.c_uint32)),
    ]


class _GenericStagePlanV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("stages", ctypes.POINTER(_GenericStageDescV1)),
        ("n_stages", ctypes.c_uint64),
        ("stage_self", ctypes.c_void_p),
        ("run_opaque", _RunOpaqueFn),
    ]


FrtModelRuntimeV1._fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("exp", ctypes.c_void_p),
        ("ports", ctypes.c_void_p),
        ("n_ports", ctypes.c_uint64),
        ("stages", ctypes.c_void_p),
        ("n_stages", ctypes.c_uint64),
        ("self", ctypes.c_void_p),
        ("verbs", _FrtVerbsV1),
        ("owner", ctypes.c_void_p),
        ("retain", _RetainReleaseFn),
        ("release", _RetainReleaseFn),
        ("query_extension", _QueryExtensionFn),
]


# Port indices (c_api.h: FRT_LLAMA_CPP_PI0_PORT_*)
PORT_IMAGES = 0
PORT_PROMPT = 1
PORT_STATE = 2
PORT_ACTIONS = 3
STAGE_INFER = 0
STAGE_CONTEXT = 1
STAGE_ACTION = 2

_FRT_MODEL_RUNTIME_ABI_VERSION = 1
_FRT_EXT_GENERIC_STAGE_PLAN_V1 = 1
_F32_BYTES = np.dtype(np.float32).itemsize


def _generic_plan(model):
    extension = ctypes.c_void_p()
    rc = model.query_extension(
        ctypes.byref(model), _FRT_EXT_GENERIC_STAGE_PLAN_V1, 1,
        ctypes.byref(extension))
    if rc != 0 or not extension.value:
        raise RuntimeError(f"generic stage plan query failed (rc={rc})")
    plan = _GenericStagePlanV1.from_address(extension.value)
    if plan.abi_version < 1 or not plan.stages or not plan.run_opaque:
        raise RuntimeError("provider returned an invalid generic stage plan")
    return plan


def _run_generic_stage(model, name):
    plan = _generic_plan(model)
    encoded = name.encode("utf-8")
    for index in range(plan.n_stages):
        stage = plan.stages[index]
        if stage.name == encoded:
            return plan.run_opaque(plan.stage_self, stage.executor_ref)
    raise RuntimeError(f"generic stage {name!r} is not in the selected plan")


def _find_lib(lib_path, env_var="FLASHRT_PI0_LIB"):
    """Resolve the SHARED provider .so path.

    Priority: explicit ``lib_path`` kwarg, environment override, then the
    platform dynamic-loader search path. An explicit path is a hard contract.
    """
    if lib_path is not None:
        if not os.path.exists(lib_path):
            raise RuntimeError(
                f"lib_path does not exist: {lib_path}")
        return lib_path
    env = os.environ.get(env_var)
    if env and os.path.exists(env):
        return env
    return (ctypes.util.find_library("flashrt_cpp_llama_cpp_provider_c") or
            "libflashrt_cpp_llama_cpp_provider_c.so")


class Pi0JetsonPiFrontend:
    """Pi0 VLA frontend backed by the Jetson-PI llama.cpp/GGML provider."""

    def __init__(self, checkpoint, *, mmproj_path=None, backend="cpu",
                 num_views=2, image_height=224, image_width=224,
                 action_steps=None, action_dim=None,
                 lib_path=None, **_unused):
        if mmproj_path is None:
            raise ValueError("mmproj_path is required for the Jetson-PI Pi0 frontend")
        if action_steps is None or action_dim is None:
            raise ValueError(
                "action_steps and action_dim must be set explicitly (the "
                "verified pi0_base checkpoint is 50x32; use the selected "
                "checkpoint's model-specific shape)")
        self.num_views = int(num_views)
        if self.num_views < 1 or self.num_views > 3:
            raise ValueError("num_views must be in [1, 3] for Jetson-PI Pi0")
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.action_steps = int(action_steps)
        self.action_dim = int(action_dim)
        self._prompt = b""
        self._lib_path = _find_lib(lib_path)
        self._lib = ctypes.CDLL(self._lib_path)

        self._lib.frt_model_runtime_open_v1.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self._lib.frt_model_runtime_open_v1.restype = ctypes.c_int

        config = {
            "model_family": "pi0",
            "model_path": str(checkpoint),
            "mmproj_path": str(mmproj_path),
            "backend": backend,
            "n_views": self.num_views,
            "image_height": int(image_height),
            "image_width": int(image_width),
            "image_channels": 3,
            "action_steps": int(action_steps),
            "action_dim": int(action_dim),
        }
        # ensure_ascii=False: the C-side JSON parser (pi0_runtime.cpp) does
        # not handle \uXXXX escapes, only raw UTF-8 bytes.
        config_json = json.dumps(config, ensure_ascii=False).encode("utf-8")

        model_ptr = ctypes.c_void_p(0)
        rc = self._lib.frt_model_runtime_open_v1(
            config_json, ctypes.byref(model_ptr))
        if rc != 0 or not model_ptr.value:
            raise RuntimeError(
                f"frt_model_runtime_open_v1 failed for Pi0 (rc={rc})")

        self._model = FrtModelRuntimeV1.from_address(model_ptr.value)
        # ABI gate: refuse to drive a struct laid out for a different ABI
        # version (mirrors the check in runtime/bindings/runtime_pybind.cpp).
        if self._model.abi_version != _FRT_MODEL_RUNTIME_ABI_VERSION:
            raise RuntimeError(
                f"frt_model_runtime_v1 abi_version={self._model.abi_version}, "
                f"expected {_FRT_MODEL_RUNTIME_ABI_VERSION}; the provider "
                f".so was built against a different FlashRT runtime ABI.")
        if self._model.struct_size < ctypes.sizeof(FrtModelRuntimeV1):
            raise RuntimeError(
                f"frt_model_runtime_v1 struct_size={self._model.struct_size} "
                f"< ctypes sizeof {ctypes.sizeof(FrtModelRuntimeV1)}; the "
                f"provider .so is older than the Python frontend expects.")
        self._model_ptr = model_ptr  # keep the uintptr for sanity

    # -- VLAModel.predict contract -------------------------------------------

    def set_prompt(self, prompt_text):
        # Note: this signature does NOT accept `state` — VLAModel.predict
        # detects that and routes state through observation["state"] instead.
        if isinstance(prompt_text, bytes):
            self._prompt = prompt_text
        else:
            self._prompt = (prompt_text or "").encode("utf-8")

    def infer(self, observation, debug=False):
        if self._model is None:
            raise RuntimeError("Pi0JetsonPiFrontend is closed")
        _ = debug  # accepted for VLAModel.predict signature parity; unused
        # Collect images: predict() passes obs with 'images' list + 'image'/
        # 'wrist_image' legacy keys. Prefer the explicit list.
        if "images" in observation:
            images = list(observation["images"])
        else:
            images = [observation["image"]]
            if "wrist_image" in observation:
                images.append(observation["wrist_image"])
        if len(images) != self.num_views:
            raise ValueError(
                f"expected {self.num_views} images, got {len(images)}")
        views = self._make_image_views(images)

        state = observation.get("state")
        if state is None:
            raise ValueError("observation['state'] is required for the Jetson-PI Pi0 frontend")
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size == 0:
            raise ValueError("observation['state'] must contain at least one value")
        if not np.all(np.isfinite(state)):
            raise ValueError("observation['state'] values must be finite")
        state = np.ascontiguousarray(state)

        verbs = self._model.verbs
        self_ = self._model.self

        rc = verbs.set_input(self_, PORT_IMAGES, ctypes.cast(views, ctypes.c_void_p),
                          ctypes.sizeof(FrtImageView) * len(images), -1)
        self._check(rc, "set_input images")
        rc = verbs.set_input(self_, PORT_PROMPT, self._prompt,
                          len(self._prompt), -1)
        self._check(rc, "set_input prompt")
        rc = verbs.set_input(self_, PORT_STATE, state.ctypes.data,
                          state.nbytes, -1)
        self._check(rc, "set_input state")

        rc = verbs.step(self_)
        self._check(rc, "step infer")

        capacity = self.action_steps * self.action_dim * _F32_BYTES
        out = (ctypes.c_char * capacity)()
        written = ctypes.c_uint64(0)
        rc = verbs.get_output(self_, PORT_ACTIONS, out, capacity,
                           ctypes.byref(written), -1)
        self._check(rc, "get_output actions")
        need = capacity
        if written.value != need:
            raise RuntimeError(
                f"get_output wrote {written.value} bytes, expected {need}")
        actions = np.frombuffer(out, dtype=np.float32,
                                count=self.action_steps * self.action_dim
                                ).reshape(self.action_steps, self.action_dim).copy()
        return {"actions": actions}

    def context(self, observation):
        """Run the Pi0 context stage and retain provider-private encoded state."""
        if self._model is None:
            raise RuntimeError("Pi0JetsonPiFrontend is closed")
        images = list(observation["images"]) if "images" in observation else [
            observation["image"]] + (
                [observation["wrist_image"]]
                if "wrist_image" in observation else [])
        if len(images) != self.num_views:
            raise ValueError(
                f"expected {self.num_views} images, got {len(images)}")
        state = observation.get("state")
        if state is None:
            raise ValueError(
                "observation['state'] is required for the Jetson-PI Pi0 frontend")
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size == 0:
            raise ValueError("observation['state'] must contain at least one value")
        if not np.all(np.isfinite(state)):
            raise ValueError("observation['state'] values must be finite")
        state = np.ascontiguousarray(state)
        views = self._make_image_views(images)
        verbs = self._model.verbs
        self_ = self._model.self
        self._check(verbs.set_input(
            self_, PORT_IMAGES, ctypes.cast(views, ctypes.c_void_p),
            ctypes.sizeof(FrtImageView) * len(images), -1),
            "set_input images")
        self._check(verbs.set_input(
            self_, PORT_PROMPT, self._prompt, len(self._prompt), -1),
            "set_input prompt")
        self._check(verbs.set_input(
            self_, PORT_STATE, state.ctypes.data,
            state.nbytes, -1), "set_input state")
        self._check(_run_generic_stage(self._model, "context"),
                    "generic stage context")

    def action(self):
        """Consume the pending Pi0 context and return one action chunk."""
        if self._model is None:
            raise RuntimeError("Pi0JetsonPiFrontend is closed")
        verbs = self._model.verbs
        self._check(_run_generic_stage(self._model, "action"),
                    "generic stage action")
        capacity = self.action_steps * self.action_dim * _F32_BYTES
        out = (ctypes.c_char * capacity)()
        written = ctypes.c_uint64(0)
        self._check(verbs.get_output(
            self._model.self, PORT_ACTIONS, out, capacity,
            ctypes.byref(written), -1), "get_output actions")
        if written.value != capacity:
            raise RuntimeError(
                f"get_output wrote {written.value} bytes, expected {capacity}")
        return np.frombuffer(
            out, dtype=np.float32,
            count=self.action_steps * self.action_dim).reshape(
                self.action_steps, self.action_dim).copy()

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

    # -- helpers --------------------------------------------------------------

    def _make_image_views(self, images):
        views = (FrtImageView * len(images))()
        for i, im in enumerate(images):
            arr = np.ascontiguousarray(im, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(
                    f"image {i} must be HxWx3 uint8, got shape {arr.shape}")
            if arr.shape[0] != self.image_height or arr.shape[1] != self.image_width:
                raise ValueError(
                    f"image {i} must be {self.image_height}x{self.image_width}, "
                    f"got {arr.shape[0]}x{arr.shape[1]}")
            views[i].struct_size = ctypes.sizeof(FrtImageView)
            views[i].pixel_format = FRT_RT_PIXEL_RGB8
            views[i].data = ctypes.c_void_p(arr.ctypes.data)
            views[i].bytes = arr.nbytes
            views[i].width = int(arr.shape[1])
            views[i].height = int(arr.shape[0])
            views[i].stride_bytes = int(arr.strides[0])
            views[i].reserved = 0
            views[i].timestamp_ns = 0
            # Keep the array alive for the duration of the infer call by
            # stashing it on the views object (engine copies on set_input).
            setattr(views, f"_keepalive_{i}", arr)
        return views

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
