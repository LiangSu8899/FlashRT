"""FlashVLA -- Pi0 model pipelines.

Per the unified pipeline_<hw>.py contract:
    pipeline_thor.py  - Thor SM110 decoder forward fns (Pi0 standard
                        RMSNorm, action_time_mlp + state_proj,
                        S_dec = Sa + 1)
    pipeline_rtx.py   - RTX SM120/SM89 Pi0Pipeline class (added in
                        S2 Phase 5)
"""
