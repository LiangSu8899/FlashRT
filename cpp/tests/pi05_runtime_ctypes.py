"""ctypes declarations shared by Pi0.5 model-runtime integration gates."""

from __future__ import annotations

import ctypes


FRT_PI05_DTYPE_BFLOAT16 = 1
FRT_PI05_DTYPE_FLOAT16 = 2
FRT_PI05_DTYPE_FLOAT32 = 3

PromptLengthUpdateFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64)


class Pi05RuntimeConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("num_views", ctypes.c_int),
        ("chunk", ctypes.c_int),
        ("model_action_dim", ctypes.c_int),
        ("robot_action_dim", ctypes.c_int),
        ("action_mean", ctypes.POINTER(ctypes.c_float)),
        ("n_action_mean", ctypes.c_uint64),
        ("action_stddev", ctypes.POINTER(ctypes.c_float)),
        ("n_action_stddev", ctypes.c_uint64),
        ("graph_name", ctypes.c_char_p),
        ("image_buffer_name", ctypes.c_char_p),
        ("action_buffer_name", ctypes.c_char_p),
        ("image_dtype", ctypes.c_int),
        ("action_dtype", ctypes.c_int),
        ("max_frame_width", ctypes.c_int),
        ("max_frame_height", ctypes.c_int),
        ("prompt_tokenizer_model_path", ctypes.c_char_p),
        ("prompt_embedding_table_data", ctypes.c_void_p),
        ("prompt_embedding_table_bytes", ctypes.c_uint64),
        ("prompt_embedding_table_dtype", ctypes.c_int),
        ("prompt_embedding_vocab_size", ctypes.c_uint64),
        ("prompt_embedding_hidden_dim", ctypes.c_uint64),
        ("prompt_embedding_data", ctypes.c_void_p),
        ("prompt_embedding_bytes", ctypes.c_uint64),
        ("prompt_embedding_dtype", ctypes.c_int),
        ("max_prompt_tokens", ctypes.c_uint64),
        ("prompt_embedding_scale", ctypes.c_float),
        ("state_q01", ctypes.POINTER(ctypes.c_float)),
        ("n_state_q01", ctypes.c_uint64),
        ("state_q99", ctypes.POINTER(ctypes.c_float)),
        ("n_state_q99", ctypes.c_uint64),
        ("prompt_length_update", PromptLengthUpdateFn),
        ("prompt_length_update_user", ctypes.c_void_p),
        ("prompt_embedding_on_device", ctypes.c_int),
    ]


def load_pi05_library(path):
    lib = ctypes.CDLL(str(path))
    lib.frt_pi05_model_runtime_create_over.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Pi05RuntimeConfig),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.frt_pi05_model_runtime_create_over.restype = ctypes.c_int
    return lib


def native_overlay(lib, config, *, keepalive=()):
    """Return the atomic producer-declaration -> C++ runtime transformer."""
    def apply(declaration_ptr: int) -> int:
        model_ptr = ctypes.c_void_p()
        rc = lib.frt_pi05_model_runtime_create_over(
            ctypes.c_void_p(declaration_ptr), ctypes.byref(config),
            ctypes.byref(model_ptr))
        if rc != 0:
            raise RuntimeError(
                f"frt_pi05_model_runtime_create_over failed rc={rc}")
        return int(model_ptr.value or 0)

    apply._keepalive = tuple(keepalive)
    return apply
