"""Structure implementations.

``hub_kernel`` is the shared, process-wide hub loader: two impls that
depend on the same kernel repo must share one loaded module — a second
``kernels.get_kernel`` import of the same repo re-registers its fake
ops and torch.library raises.

The loader also checks the package's own hardware declaration. A Hub
kernel package ships ``metadata.json`` with the CUDA archs it was built
for; that file is maintained on the kernels side and is the single
source of truth for hardware support — this layer reads it, it does not
keep a second table. A device outside the declared archs gets a clean
refusal here, before the kernel produces an unrelated-looking runtime
error; the refusal is caught by the binder and recorded in the plan
notes like any other. A package without metadata is loaded as before —
absence of a declaration is not evidence of incompatibility.
"""

import json
import pathlib
from functools import lru_cache


def _device_cc() -> tuple[int, int] | None:
    """Compute capability of the current CUDA device, or ``None``."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


def _declared_archs(module) -> list[str] | None:
    """The package's own ``backend.archs`` declaration, if it ships one."""
    try:
        meta = pathlib.Path(module.__file__).parent / "metadata.json"
        if not meta.is_file():
            return None
        archs = json.loads(meta.read_text()).get("backend", {}).get("archs")
        return list(archs) if archs else None
    except (OSError, ValueError, AttributeError):
        return None


def _check_arch(repo: str, module) -> None:
    archs = _declared_archs(module)
    if archs is None:
        return
    cc = _device_cc()
    if cc is None:
        # no CUDA device: binding fails later at weight transfer anyway;
        # the arch check has nothing truthful to say here
        return
    want = f"{cc[0]}.{cc[1]}"
    # "12.0a" (arch-specific build) serves a 12.0 device
    if any(a == want or a.rstrip("a") == want for a in archs):
        return
    raise ValueError(
        f"refused: kernel package {repo!r} declares archs {archs}, "
        f"device is sm {want}")


#: modules cached independently of the arch check: ``get_kernel`` must
#: run at most once per repo even when the check refuses (a second load
#: re-registers the package's fake ops and torch.library raises — the
#: refusal path must not manufacture that error on retry)
_LOADED: dict[tuple[str, str], object] = {}


@lru_cache(maxsize=None)
def hub_kernel(repo: str, version: str):
    from kernels import get_kernel

    key = (repo, version)
    if key not in _LOADED:
        _LOADED[key] = get_kernel(repo, version=version)
    module = _LOADED[key]
    _check_arch(repo, module)
    return module
