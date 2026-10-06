import sys, importlib, numpy as np
sys.path.insert(0, "ai/tools")
from om3_crosscheck import om3_oas, crosscheck, random_inputs, report
om3_oas()
from openaerostruct.meshing.mesh_generator import generate_mesh
def surfs(sym):
    m1, _ = generate_mesh({"num_y": 7, "num_x": 3, "wing_type": "CRM", "symmetry": sym, "num_twist_cp": 3})
    m2 = generate_mesh({"num_y": 5, "num_x": 2, "wing_type": "rect", "symmetry": sym}) + np.array([30.0, 0, 2.0])
    return [{"name": "wing", "mesh": m1, "symmetry": sym}, {"name": "tail", "mesh": m2, "symmetry": sym}]
fails = 0
for sym in (True, False):
    S = surfs(sym)
    nev = sum((s["mesh"].shape[0] - 1) * (s["mesh"].shape[1] - 1) for s in S)
    cases = [
        ("convert_velocity", "ConvertVelocity", dict(surfaces=S), {"alpha": 3.0, "beta": 1.0}),
        ("convert_velocity", "ConvertVelocity", dict(surfaces=S, rotational=True), {"alpha": 3.0, "beta": 1.0}),
        ("mtx_rhs", "VLMMtxRHSComp", dict(surfaces=S), {}),
        ("horseshoe_circulations", "HorseshoeCirculations", dict(surfaces=S), {}),
        ("eval_velocities", "EvalVelocities", dict(surfaces=S, eval_name="force_pts", num_eval_points=nev), {}),
        ("rotational_velocity", "RotationalVelocity", dict(surfaces=S), {}),
        ("mesh_point_forces", "MeshPointForces", dict(surfaces=S), {}),
        ("panel_forces", "PanelForces", dict(surfaces=S), {}),
        ("panel_forces_surf", "PanelForcesSurf", dict(surfaces=S), {}),
    ]
    for mod, cls, kw, over in cases:
        old = getattr(importlib.import_module(f"oas_om3.aerodynamics.{mod}"), cls)
        new = getattr(importlib.import_module(f"openaerostruct.aerodynamics.{mod}"), cls)
        n = new(**kw)
        fails += report(f"{cls} sym={sym} {sorted(set(kw) - {'surfaces'})}", crosscheck(old(**kw), n, random_inputs(n, over, seed=4), rtol=1e-10, atol=1e-10))
    # SolveMatrix: well-conditioned mtx
    from oas_om3.aerodynamics.solve_matrix import SolveMatrix as O
    from openaerostruct.aerodynamics.solve_matrix import SolveMatrix as N
    rng = np.random.default_rng(5)
    mtx = rng.random((nev, nev)) + nev * np.eye(nev)
    fails += report(f"SolveMatrix sym={sym}", crosscheck(O(surfaces=S), N(surfaces=S), {"mtx": mtx, "rhs": rng.random(nev)}, rtol=1e-10, atol=1e-10))
print("FAILURES", fails)
