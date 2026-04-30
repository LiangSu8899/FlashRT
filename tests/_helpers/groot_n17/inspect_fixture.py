"""Print shape + dtype for every key in a Phase 1 fixture."""
import sys
import torch

f = sys.argv[1] if len(sys.argv) > 1 else \
    "/work/tests/fixtures/gr00t_n17_ref_oxe_droid_relative_eef_relative_joint_2v_traj1_step0_seed0.pt"
d = torch.load(f, map_location="cpu", weights_only=False)
print(f"==> {f}")
for top in ("meta", "inputs", "actions", "activations"):
    print(f"\n--- {top} ---")
    v = d.get(top, None)
    if v is None:
        continue
    if top == "meta":
        for k, val in v.items():
            print(f"  {k}: {val}")
        continue
    if top == "inputs":
        for k, val in v.items():
            if hasattr(val, "shape"):
                print(f"  {k}: {tuple(val.shape)} {val.dtype}")
            elif isinstance(val, dict):
                print(f"  {k}: dict[{len(val)}]")
                for kk, vv in val.items():
                    if hasattr(vv, "shape"):
                        print(f"    {kk}: {tuple(vv.shape)} {vv.dtype}")
                    else:
                        s = str(vv)[:80]
                        print(f"    {kk}: {type(vv).__name__} {s}")
            else:
                s = str(val)[:80]
                print(f"  {k}: {type(val).__name__} {s}")
        continue
    # actions / activations: dict of tensors
    for k in sorted(v.keys()):
        t = v[k]
        if hasattr(t, "shape"):
            print(f"  {k}: {tuple(t.shape)} {t.dtype}")
        else:
            print(f"  {k}: {type(t).__name__}")
