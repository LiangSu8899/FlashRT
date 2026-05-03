"""FlashRT — Action post-processing utilities."""

import numpy as np

LIBERO_ACTION_DIM = 7


def unnormalize_actions(actions, norm_stats):
    """Unnormalize actions using q01/q99 statistics (pure numpy)."""
    q01 = np.array(norm_stats["actions"]["q01"], dtype=np.float32)
    q99 = np.array(norm_stats["actions"]["q99"], dtype=np.float32)
    dim = min(actions.shape[-1], len(q01))
    clipped = np.clip(actions, -1.0, 1.0)
    unnorm = clipped.copy()
    unnorm[..., :dim] = (
        (clipped[..., :dim] + 1.0) / 2.0 * (q99[:dim] - q01[:dim] + 1e-6)
        + q01[:dim]
    )
    return unnorm
