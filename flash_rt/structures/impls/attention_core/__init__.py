from .fa2_seqused import (DenseAttention, PackedKVAttention,
                          bind_attention_core,
                          bind_dense_attention, plan_packed_kv)
from .two_way_fa2 import FactoredTwoWayAttention, bind_two_way_attention


def bind_dense_attention_best(captures):
    """Dense attention across the variant family: FA2, then FA4.

    One structure, parallel executable forms, and no second hardware
    table: each variant's kernel package declares its own archs, so the
    device split *is* the refusal machinery. FA2 (BF16, numerics
    preserving) binds wherever its runtime executes — every current
    receipt keeps its exact form. Where FA2's package refuses the
    device (the SM100 family: Thor and kin), the FA4 FP8 form takes the
    seam, which on those devices is the production hot path. Both are
    judged by the same downstream gates; a device neither serves keeps
    the host's own attention.
    """
    # a package can refuse a device two ways: its arch declaration
    # (ValueError from the loader's metadata check) or the kernels
    # library finding no build variant for the host at all (OSError) —
    # both mean "this variant does not serve here", not an error
    fa2_refusal = None
    try:
        core = bind_dense_attention(captures)
        if core is not None:
            return core
    except (ValueError, RuntimeError, OSError) as refusal:
        fa2_refusal = refusal
    from . import fa4_fp8
    try:
        return fa4_fp8.bind_dense_attention(captures)
    except (ValueError, RuntimeError, OSError) as refusal:
        if fa2_refusal is not None:
            raise ValueError(
                f"attention_core: no variant serves this device — "
                f"fa2: {fa2_refusal}; fa4: {refusal}") from refusal
        if isinstance(refusal, OSError):
            raise ValueError(
                f"attention_core fa4: no build variant for this host — "
                f"{refusal}") from refusal
        raise


__all__ = ["DenseAttention", "PackedKVAttention",
           "bind_attention_core", "bind_dense_attention",
           "bind_dense_attention_best", "plan_packed_kv",
           "FactoredTwoWayAttention", "bind_two_way_attention"]
