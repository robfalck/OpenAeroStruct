import sys, importlib, numpy as np

sys.path.insert(0, "ai/tools")
from om3_crosscheck import om3_oas, crosscheck, random_inputs, report

om3_oas()
from openaerostruct.meshing.mesh_generator import generate_mesh

fails = 0
for alt in (0.0, 5000.0, 35000.0, 60000.0):
    n = importlib.import_module("openaerostruct.common.atmos_comp").AtmosComp()
    o = importlib.import_module("oas_om3.common.atmos_comp").AtmosComp()
    fails += report(
        f"AtmosComp alt={alt}",
        crosscheck(o, n, random_inputs(n, {"altitude": alt, "Mach_number": 0.7}, seed=3), rtol=1e-9, atol=1e-12),
    )
n = importlib.import_module("openaerostruct.common.reynolds_comp").ReynoldsComp()
o = importlib.import_module("oas_om3.common.reynolds_comp").ReynoldsComp()
fails += report(
    "ReynoldsComp",
    crosscheck(o, n, random_inputs(n, {"rho": 2e-3, "mu": 3e-7, "v": 800.0}, seed=4), rtol=1e-9, atol=1e-12),
)
for sym in (True, False):
    m1, _ = generate_mesh({"num_y": 7, "num_x": 3, "wing_type": "CRM", "symmetry": sym, "num_twist_cp": 3})
    m2 = generate_mesh({"num_y": 5, "num_x": 2, "wing_type": "rect", "symmetry": sym}) + np.array([30.0, 0, 2.0])
    base = dict(symmetry=sym, CL0=0.05, CD0=0.015, k_lam=0.05, c_max_t=0.303, t_over_c_cp=np.array([0.15]))
    W = dict(base, name="wing", mesh=m1, with_viscous=True, with_wave=False, fem_model_type="tube", S_ref_type="wetted")
    T = dict(
        base, name="tail", mesh=m2, with_viscous=False, with_wave=False, fem_model_type="tube", S_ref_type="wetted"
    )
    for mod, cls in (
        ("functionals.breguet_range", "BreguetRange"),
        ("functionals.center_of_gravity", "CenterOfGravity"),
        ("functionals.equilibrium", "Equilibrium"),
    ):
        surfs = [dict(W, structural_mass=1.0e3) if False else W, T]
        try:
            n = getattr(importlib.import_module("openaerostruct." + mod), cls)(surfaces=surfs)
            o = getattr(importlib.import_module("oas_om3." + mod), cls)(surfaces=surfs)
        except Exception as e:
            print(cls, "construct failed", repr(e))
            fails += 1
            continue
        ins = random_inputs(n, {}, seed=5)
        fails += report(f"{cls} sym={sym}", crosscheck(o, n, ins, rtol=1e-9, atol=1e-12))
print("FAILURES", fails)
