from .fa2_seqused import (DenseAttention, PackedKVAttention,
                          bind_attention_core,
                          bind_dense_attention, plan_packed_kv)
from .two_way_fa2 import FactoredTwoWayAttention, bind_two_way_attention


def bind_dense_attention_best(captures):
    """Dense attention across the variant family, precision-descending.

    One structure, parallel executable forms, and no second hardware
    table: each variant's kernel package declares its own archs (or
    ships build variants only for the hosts it serves), so the device
    split *is* the refusal machinery. The order is precision first:
    FA2 (BF16, numerics preserving) binds wherever its runtime
    executes — every current receipt keeps its exact form. Then the
    FA4 CuTe DSL form (BF16, the SM100-family hot path), the
    allocation-free masked MHA (BF16/FP16, batch-of-one sites), and
    last the FP8 FA4 form. All are judged by the same downstream
    gates; a device none serves keeps the host's own attention.
    """
    # a package can refuse a device three ways: its arch declaration
    # (ValueError from the loader's metadata check), the kernels
    # library finding no build variant for the host (OSError), or a
    # bind smoke the runtime cannot execute (RuntimeError) — each
    # means "not this variant here", never an error for the family
    def _fa2(caps):
        return bind_dense_attention(caps)

    def _fa4_cute(caps):
        from . import fa4_cute
        return fa4_cute.bind_dense_attention(caps)

    def _masked_mha(caps):
        from . import masked_mha
        return masked_mha.bind_dense_attention(caps)

    def _fa4_fp8(caps):
        from . import fa4_fp8
        return fa4_fp8.bind_dense_attention(caps)

    refusals, declined = [], 0
    for name, binder in (("fa2", _fa2), ("fa4_cute", _fa4_cute),
                         ("masked_mha", _masked_mha),
                         ("fa4_fp8", _fa4_fp8)):
        try:
            core = binder(captures)
        except (ValueError, RuntimeError, OSError) as refusal:
            refusals.append(f"{name}: {str(refusal)[:80]}")
            continue
        if core is not None:
            return core
        declined += 1
    if declined:
        # at least one variant executed its qualification and declined
        # the shape form — a site-level refusal the adapter records,
        # same contract as a single binder returning None
        return None
    raise ValueError(
        "attention_core: no variant serves this device — "
        + "; ".join(refusals))


__all__ = ["DenseAttention", "PackedKVAttention",
           "bind_attention_core", "bind_dense_attention",
           "bind_dense_attention_best", "plan_packed_kv",
           "FactoredTwoWayAttention", "bind_two_way_attention"]
# variant modules (fa4_fp8, fa4_cute, masked_mha) import lazily inside
# the family binder: loading one must not require the others' runtimes
