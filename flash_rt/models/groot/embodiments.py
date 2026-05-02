"""GROOT N1.6 embodiment tag -> slot index, shared between Thor and RTX.

Kept in one place so groot_thor and groot_rtx frontends cannot drift
independently. Any caller that needs to map a tag to its per-embodiment
MLP slot imports from here.

The GR00T-N1.6-3B checkpoint has 32 embodiment slots in each per-embodiment
MLP weight tensor (state_encoder / action_encoder / action_decoder). Only
a subset of those slots actually contain trained weights — the rest are
at initialization std ~0.02 and will produce noise-like outputs regardless
of input. The ``TRAINED_EMBODIMENT_IDS`` set below records the empirically
verified trained slots (measured on ``action_encoder.W1.W`` std > 0.025).

If you ask the model to run on an untrained embodiment, you get garbage
outputs without any crash or obvious sign the model isn't actually
processing your inputs — the ``new_embodiment`` default is a placeholder
slot explicitly meant for fine-tuning, so running a demo against it with
the base checkpoint is expected to look noise-like.
"""

EMBODIMENT_TAG_TO_INDEX: dict[str, int] = {
    "oxe_google": 0,
    "oxe_widowx": 1,
    "libero_panda": 2,
    "unitree_g1": 8,
    "new_embodiment": 10,
    "robocasa_panda_omron": 13,
    "oxe_droid": 16,
    "gr1": 20,
    "behavior_r1_pro": 24,
}

# Slots with non-initialization weight std in GR00T-N1.6-3B (measured on
# action_encoder.W1.W, threshold std > 0.025). Outside this set the
# per-embodiment MLPs are at init (~0.02) and outputs will be noise.
#
#   7   — (no public tag)
#   13  — robocasa_panda_omron
#   17  — (no public tag)
#   20  — gr1
#   23  — (no public tag)
#   24  — behavior_r1_pro
#   25  — (no public tag)
#   26  — (no public tag)
TRAINED_EMBODIMENT_IDS: frozenset[int] = frozenset({7, 13, 17, 20, 23, 24, 25, 26})

# Public trained tags that users can reasonably select without knowing
# the internal slot index. Other trained slots exist (7, 17, 23, 25, 26)
# but aren't exposed in the embodiment_id.json of the 3B checkpoint.
PUBLIC_TRAINED_TAGS: tuple[str, ...] = (
    "robocasa_panda_omron",   # 13 — 3-view tabletop arm
    "gr1",                    # 20 — 1-view humanoid
    "behavior_r1_pro",        # 24 — 3-view humanoid
)


def is_embodiment_trained(tag_or_id) -> bool:
    """Return True if the given embodiment slot has trained weights.

    Accepts either the string tag (``"gr1"``) or the int slot id (20).
    Unknown tags return False.
    """
    if isinstance(tag_or_id, str):
        eid = EMBODIMENT_TAG_TO_INDEX.get(tag_or_id)
        if eid is None:
            return False
    else:
        eid = int(tag_or_id)
    return eid in TRAINED_EMBODIMENT_IDS
