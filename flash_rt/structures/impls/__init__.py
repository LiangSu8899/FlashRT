"""Structure implementations.

``hub_kernel`` is the shared, process-wide hub loader: two impls that
depend on the same kernel repo must share one loaded module — a second
``kernels.get_kernel`` import of the same repo re-registers its fake
ops and torch.library raises.
"""

from functools import lru_cache


@lru_cache(maxsize=None)
def hub_kernel(repo: str, version: str):
    from kernels import get_kernel

    return get_kernel(repo, version=version)
