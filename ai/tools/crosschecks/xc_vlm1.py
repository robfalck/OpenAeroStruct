import sys, numpy as np
sys.path.insert(0, "ai/tools")
from om3_crosscheck import om3_oas, crosscheck, random_inputs, report
om3_oas()
import oas_om3.aerodynamics.get_vectors as o_gv, oas_om3.aerodynamics.collocation_points as o_cp, oas_om3.aerodynamics.eval_mtx as o_em
import openaerostruct.aerodynamics.get_vectors as n_gv, openaerostruct.aerodynamics.collocation_points as n_cp, openaerostruct.aerodynamics.eval_mtx as n_em
from openaerostruct.meshing.mesh_generator import generate_mesh
fails = 0
def surfs(sym, right=False):
    m1, _ = generate_mesh({"num_y": 7, "num_x": 3, "wing_type": "CRM", "symmetry": sym, "num_twist_cp": 3})
    m2 = generate_mesh({"num_y": 5, "num_x": 2, "wing_type": "rect", "symmetry": sym})
    m2 = m2 + np.array([30.0, 0, 2.0])
    if right:
        m1 = m1[:, ::-1, :] * np.array([1, -1, 1])
    return [{"name": "wing", "mesh": m1, "symmetry": sym}, {"name": "tail", "mesh": m2, "symmetry": sym}]
for sym, right in ((True, False), (False, False), (True, True)):
    S = surfs(sym, right)
    nev = sum((s["mesh"].shape[0] - 1) * (s["mesh"].shape[1] - 1) for s in S)
    tag = f"sym={sym} right={right}"
    fails += report(f"CollocationPoints {tag}", crosscheck(o_cp.CollocationPoints(surfaces=S), n_cp.CollocationPoints(surfaces=S),
        random_inputs(n_cp.CollocationPoints(surfaces=S), {f"{s['name']}_def_mesh": s["mesh"] for s in S})))
    kw = dict(surfaces=S, num_eval_points=nev, eval_name="coll_pts")
    fails += report(f"GetVectors {tag}", crosscheck(o_gv.GetVectors(**kw), n_gv.GetVectors(**kw), random_inputs(n_gv.GetVectors(**kw), seed=2)))
    new = n_em.EvalVelMtx(**kw)
    ins = random_inputs(new, {"alpha": 3.0}, seed=3)
    for k in ins:
        if k.endswith("_vectors"):
            ins[k] = ins[k] * 5 - 2.5
    fails += report(f"EvalVelMtx {tag}", crosscheck(o_em.EvalVelMtx(**kw), new, ins, rtol=1e-7, atol=1e-9))
print("FAILURES", fails)
