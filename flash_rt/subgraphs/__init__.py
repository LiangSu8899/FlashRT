"""Subgraph cut registration and producer capture hooks.

This package is intentionally outside ``flash_rt.runtime``. Runtime owns the
stable ABI; subgraphs are producer/customer setup policy.
"""

from .capture import (
    apply_frontend_capture_hooks,
    capture_graph,
    export_graph_records,
    materialize_captured_graphs,
    register_captured_graph,
    register_capture_hook,
    register_frontend_capture_hook,
    register_export_graph,
    run_capture_hooks,
)
from .stage_plan import (
    Stage,
    StagePlan,
    StagePlanFactory,
    list_stage_plans,
    register_stage_plan,
    resolve_stage_plan,
)

__all__ = [
    "Stage",
    "StagePlan",
    "StagePlanFactory",
    "apply_frontend_capture_hooks",
    "capture_graph",
    "export_graph_records",
    "list_stage_plans",
    "materialize_captured_graphs",
    "register_captured_graph",
    "register_capture_hook",
    "register_export_graph",
    "register_frontend_capture_hook",
    "register_stage_plan",
    "resolve_stage_plan",
    "run_capture_hooks",
]
