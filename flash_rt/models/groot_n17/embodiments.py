"""GR00T N1.7 embodiment-tag → slot mapping.

Source of truth: ``embodiment_id.json`` shipped with the
``nvidia/GR00T-N1.7-3B`` checkpoint (52 tags, 30 slots).

Multi-view configuration is per-embodiment per
``Isaac-GR00T/gr00t/configs/data/embodiment_configs.py``: each tag has a
``modality_keys["video"]`` list whose length is the number of camera views
the tag was trained with.
"""

from __future__ import annotations


EMBODIMENT_TAG_TO_INDEX: dict[str, int] = {
    "simpler_env_google": 0,
    "simpler_env_widowx": 1,
    "libero_sim": 2,
    "droid_sim": 3,
    "robocasa_panda_omron": 13,
    "gr1_unified": 20,
    "oxe_droid_relative_eef_relative_joint": 24,
    "unitree_g1_full_body_with_waist_height_nav_cmd": 25,
    "real_g1_relative_eef_relative_joints": 25,
    "real_r1_pro_sharpa_relative_eef": 26,
    "agibot": 26,
    "xdof_relative_eef_relative_joint": 27,
    "new_embodiment": 10,
}

EMBODIMENT_NUM_VIEWS: dict[str, int] = {
    "simpler_env_google": 1,
    "simpler_env_widowx": 1,
    "unitree_g1_full_body_with_waist_height_nav_cmd": 1,
    "libero_sim": 2,
    "gr1_unified": 2,
    "oxe_droid_relative_eef_relative_joint": 2,
}
