"""FlashRT structures — verified, composable model sub-blocks.

A structure is a versioned specification of one model region: boundary
tensors, framework-neutral weight slots, a plain reference implementation
used as ground truth, and qualification gates. This package hosts the
structure catalog and its registry. Implementations, host adapters, and
the qualification harness build on top of these specifications.
"""

from flash_rt.structures.registry import StructureSpec, list_structures, load

__all__ = ["StructureSpec", "list_structures", "load"]
