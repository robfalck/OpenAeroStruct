# NOTE: since TD-101, VortexMesh takes alpha in deg while OM3 takes rad, so the ground-effect
# VortexMesh rows here are expected to differ; ground effect is verified end to end by xc_aeropoint.py.
import sys, numpy as np
sys.path.insert(0, "ai/tools")
from om3_crosscheck import om3_oas, crosscheck, random_inputs, report
om3_oas()
from oas_om3.aerodynamics.vortex_mesh import VortexMesh as O
from openaerostruct.aerodynamics.vortex_mesh import VortexMesh as N
from oas_om3.aerodynamics.eval_mtx import EvalVelMtx as OE
from openaerostruct.aerodynamics.eval_mtx import EvalVelMtx as NE
from openaerostruct.meshing.mesh_generator import generate_mesh
fails = 0
for sym, ground, right in ((True, False, False), (False, False, False), (True, True, False), (True, True, True), (True, False, True)):
    m1, _ = generate_mesh({"num_y": 7, "num_x": 3, "wing_type": "CRM", "symmetry": sym, "num_twist_cp": 3})
    if right:
        m1 = m1[:, ::-1, :] * np.array([1, -1, 1])
    m2 = generate_mesh({"num_y": 5, "num_x": 2, "wing_type": "rect", "symmetry": sym}) + np.array([30.0, 0, 2.0])
    S = [{"name": "wing", "mesh": m1, "symmetry": sym, "groundplane": ground}, {"name": "tail", "mesh": m2, "symmetry": sym}]
    n = N(surfaces=S)
    ins = {f"{s['name']}_def_mesh": s["mesh"] + 0.01 for s in S}
    if ground:
        ins.update({"alpha": 0.07, "height_agl": 9.0})
    tag = f"sym={sym} ground={ground} right={right}"
    fails += report(f"VortexMesh {tag}", crosscheck(O(surfaces=S), n, ins, rtol=1e-7, atol=1e-8))
    if ground:
        nev = sum((s["mesh"].shape[0] - 1) * (s["mesh"].shape[1] - 1) for s in S)
        kw = dict(surfaces=S, num_eval_points=nev, eval_name="coll_pts")
        ne = NE(**kw)
        ins = random_inputs(ne, {"alpha": 3.0}, seed=7)
        for k in ins:
            if k.endswith("_vectors"):
                ins[k] = ins[k] * 5 - 2.5
        fails += report(f"EvalVelMtx {tag}", crosscheck(OE(**kw), ne, ins, rtol=1e-7, atol=1e-9))
print("FAILURES", fails)
