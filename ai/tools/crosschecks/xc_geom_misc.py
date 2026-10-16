import sys, importlib, numpy as np

sys.path.insert(0, "ai/tools")
from om3_crosscheck import om3_oas, crosscheck, random_inputs, report

om3_oas()
from oas_om3.utils.testing import get_three_section_surface
from oas_om3.geometry.utils import build_section_dicts
from openaerostruct.meshing.mesh_generator import generate_mesh

fails = 0
for sym in (True, False):
    mesh, _ = generate_mesh({"num_y": 7, "num_x": 3, "wing_type": "CRM", "symmetry": sym, "num_twist_cp": 3})
    S = {"name": "wing", "symmetry": sym, "mesh": mesh}
    n = importlib.import_module("openaerostruct.geometry.radius_comp").RadiusComp(surface=S)
    o = importlib.import_module("oas_om3.geometry.radius_comp").RadiusComp(surface=dict(S))
    ins = random_inputs(n, {"mesh": mesh + 0.01 * np.random.default_rng(1).random(mesh.shape)}, seed=2)
    fails += report(f"RadiusComp sym={sym}", crosscheck(o, n, ins, rtol=1e-9, atol=1e-12))
    for ny in (5, 8, 16):
        S = {"name": "wing", "symmetry": sym, "mesh": np.zeros((1, ny, 3))}
        n = importlib.import_module("openaerostruct.geometry.monotonic_constraint").MonotonicConstraint(
            var_name="x", surface=S
        )
        o = importlib.import_module("oas_om3.geometry.monotonic_constraint").MonotonicConstraint(
            var_name="x", surface=dict(S)
        )
        fails += report(
            f"MonotonicConstraint sym={sym} ny={ny}",
            crosscheck(o, n, random_inputs(n, {}, seed=3), rtol=1e-9, atol=1e-12),
        )

for sym in (True, False):
    surf, _ = get_three_section_surface(sym=sym)
    for ny in ([11, 11, 11], [5, 7, 9]):
        surf["ny"] = ny
        surf["t_over_c_cp"] = [np.array([0.15])] * 3
        secs = build_section_dicts(surf)
        if len(set(ny)) > 1:
            # OM3 sizes every section's t_over_c from the last section's ny (a bug); compare without t_over_c
            secs = [{k: v for k, v in sec.items() if k != "t_over_c_cp"} for sec in secs]
        for shift in (True, False):
            n = importlib.import_module("openaerostruct.geometry.geometry_unification").GeomMultiUnification(
                sections=secs, surface_name=surf["name"], shift_uni_mesh=shift
            )
            o = importlib.import_module("oas_om3.geometry.geometry_unification").GeomMultiUnification(
                sections=secs, surface_name=surf["name"], shift_uni_mesh=shift
            )
            ins = random_inputs(n, {}, seed=4)
            fails += report(
                f"GeomMultiUnification sym={sym} ny={ny} shift={shift}", crosscheck(o, n, ins, rtol=1e-9, atol=1e-12)
            )
        for dc in ([np.ones(3)] * 3, [np.array([1, 0, 1]), np.array([0, 1, 1])]):
            n = importlib.import_module("openaerostruct.geometry.geometry_multi_join").GeomMultiJoin(
                sections=secs, dim_constr=dc
            )
            o = importlib.import_module("oas_om3.geometry.geometry_multi_join").GeomMultiJoin(
                sections=secs, dim_constr=dc
            )
            fails += report(
                f"GeomMultiJoin sym={sym} ny={ny} dc={len(dc)}",
                crosscheck(o, n, random_inputs(n, {}, seed=5), rtol=1e-9, atol=1e-12),
            )
print("FAILURES", fails)
