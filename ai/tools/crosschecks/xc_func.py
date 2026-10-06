import sys, importlib, numpy as np
sys.path.insert(0, "ai/tools")
from om3_crosscheck import om3_oas, crosscheck, random_inputs, report
om3_oas()
from openaerostruct.meshing.mesh_generator import generate_mesh
fails = 0
for sym in (True, False):
    m1, _ = generate_mesh({"num_y": 7, "num_x": 3, "wing_type": "CRM", "symmetry": sym, "num_twist_cp": 3})
    m2 = generate_mesh({"num_y": 5, "num_x": 2, "wing_type": "rect", "symmetry": sym}) + np.array([30.0, 0, 2.0])
    base = dict(symmetry=sym, CL0=0.05, CD0=0.015, k_lam=0.05, c_max_t=0.303, t_over_c_cp=np.array([0.15]))
    W = dict(base, name="wing", mesh=m1, with_viscous=True, with_wave=True)
    T = dict(base, name="tail", mesh=m2, with_viscous=False, with_wave=False)
    ny = m1.shape[1]
    for klam in (0.05, 0.0, 1.0):
        Wk = dict(W, k_lam=klam)
        mach = {"Mach_number": 0.84, "re": 1e6, "S_ref": 100.0, "t_over_c": np.full(ny - 1, 0.12) + 0.01 * np.arange(ny - 1)}
        for mod, cls in (("aerodynamics.viscous_drag", "ViscousDrag"),):
            n = getattr(importlib.import_module("openaerostruct." + mod), cls)(surface=Wk)
            o = getattr(importlib.import_module("oas_om3." + mod), cls)(surface=Wk)
            ins = random_inputs(n, mach, seed=11)
            ins["lengths"] = ins["lengths"] * 3 + 2; ins["widths"] = ins["widths"] * 2 + 1; ins["lengths_spanwise"] = ins["widths"] * 1.1
            fails += report(f"{cls} sym={sym} k_lam={klam}", crosscheck(o, n, ins, rtol=1e-9, atol=1e-12))
    cases = [
        ("aerodynamics.wave_drag", "WaveDrag", dict(surface=W), {"Mach_number": 0.95, "CL": 0.6}),
        ("aerodynamics.wave_drag", "WaveDrag", dict(surface=T), {"Mach_number": 0.95, "CL": 0.6}),
        ("aerodynamics.lift_coeff_2D", "LiftCoeff2D", dict(surface=W), {"alpha": 3.0}),
        ("aerodynamics.lift_drag", "LiftDrag", dict(surface=W), {"alpha": 3.0, "beta": 1.0}),
        ("aerodynamics.coeffs", "Coeffs", dict(), {}),
        ("aerodynamics.total_lift", "TotalLift", dict(surface=W), {}),
        ("aerodynamics.total_drag", "TotalDrag", dict(surface=W), {}),
        ("functionals.sum_areas", "SumAreas", dict(surfaces=[W, T]), {}),
        ("functionals.total_lift_drag", "TotalLiftDrag", dict(surfaces=[W, T]), {}),
        ("functionals.moment_coefficient", "MomentCoefficient", dict(surfaces=[W, T]), {}),
    ]
    for mod, cls, kw, over in cases:
        n = getattr(importlib.import_module("openaerostruct." + mod), cls)(**kw)
        o = getattr(importlib.import_module("oas_om3." + mod), cls)(**kw)
        ins = random_inputs(n, over, seed=12)
        if cls == "WaveDrag":
            ins["widths"] = ins["widths"] + 1; ins["lengths_spanwise"] = ins["widths"] * 1.2; ins["chords"] = ins["chords"] + 3; ins["t_over_c"] = ins["t_over_c"] * 0.1
        name = kw.get("surface", {}).get("name", "") if "surface" in kw else ""
        fails += report(f"{cls} sym={sym} {name}", crosscheck(o, n, ins, rtol=1e-9, atol=1e-12))
print("FAILURES", fails)
