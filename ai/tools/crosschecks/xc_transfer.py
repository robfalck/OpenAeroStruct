import sys, importlib, numpy as np

sys.path.insert(0, "ai/tools")
from om3_crosscheck import om3_oas, crosscheck, random_inputs, report

om3_oas()
from openaerostruct.utils.testing import get_default_surfaces

fails = 0
S = dict(get_default_surfaces()[0])
mesh = S["mesh"]

for mod, cls, ov in (
    ("compute_transformation_matrix", "ComputeTransformationMatrix", {}),
    ("displacement_transfer", "DisplacementTransfer", {"mesh": mesh}),
    ("load_transfer", "LoadTransfer", {}),
):
    n = getattr(importlib.import_module(f"openaerostruct.transfer.{mod}"), cls)(surface=S)
    o = getattr(importlib.import_module(f"oas_om3.transfer.{mod}"), cls)(surface=dict(S))
    fails += report(cls, crosscheck(o, n, random_inputs(n, ov, seed=7), rtol=1e-9, atol=1e-12))

print("FAILURES", fails)
