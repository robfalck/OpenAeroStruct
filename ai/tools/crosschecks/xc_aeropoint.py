import sys, numpy as np
sys.path.insert(0, "ai/tools")
import openmdao.api as om3, om4.api as om
from om3_crosscheck import om3_oas
om3_oas()
from oas_om3.geometry.geometry_group import Geometry as G3
from oas_om3.aerodynamics.aero_groups import AeroPoint as A3
from openaerostruct.geometry.geometry_group import Geometry as G4
from openaerostruct.aerodynamics.aero_groups import AeroPoint as A4
from openaerostruct.meshing.mesh_generator import generate_mesh

FLOW = {"v": (248.136, "m/s"), "alpha": (5.0, "deg"), "Mach_number": (0.84, None), "re": (1e6, "1/m"), "rho": (0.38, "kg/m**3"), "cg": (np.array([1.0, 0.0, 0.5]), "m")}

def build(sym, ground, tail):
    mesh, tw = generate_mesh({"num_y": 7, "num_x": 3, "wing_type": "CRM", "symmetry": sym, "num_twist_cp": 3})
    w = {"name": "wing", "symmetry": sym, "S_ref_type": "wetted", "twist_cp": tw, "chord_cp": np.array([1.0, 0.9, 1.1]), "sweep": 2.0,
         "mesh": mesh, "CL0": 0.0, "CD0": 0.015, "k_lam": 0.05, "t_over_c_cp": np.array([0.15, 0.12]), "c_max_t": 0.303,
         "with_viscous": True, "with_wave": True, "groundplane": ground}
    S = [w]
    if tail:
        m2 = generate_mesh({"num_y": 5, "num_x": 2, "wing_type": "rect", "symmetry": sym}) + np.array([40.0, 0, 3.0])
        S.append({"name": "tail", "symmetry": sym, "S_ref_type": "wetted", "mesh": m2, "twist_cp": np.zeros(2), "CL0": 0.0, "CD0": 0.0, "k_lam": 0.05,
                  "t_over_c_cp": np.array([0.12]), "c_max_t": 0.303, "with_viscous": True, "with_wave": False, "groundplane": ground})
    return S

def conns(S, add):
    for s in S:
        n = s["name"]
        add(n + ".mesh", "aero_point_0." + n + ".def_mesh")
        add(n + ".mesh", "aero_point_0.aero_states." + n + "_def_mesh")
        add(n + ".t_over_c", "aero_point_0." + n + "_perf.t_over_c")

for sym, ground, tail in ((True, False, False), (True, False, True), (False, False, True), (True, True, True)):
    S = build(sym, ground, tail)
    flow = dict(FLOW)
    if ground:
        flow["height_agl"] = (8.0, "m")
    # OM3
    p3 = om3.Problem(reports=False)
    ivc = om3.IndepVarComp()
    for k, (v, u) in flow.items():
        ivc.add_output(k, val=v, units=u)
    p3.model.add_subsystem("prob_vars", ivc, promotes=["*"])
    for s in S:
        p3.model.add_subsystem(s["name"], G3(surface=dict(s)))
    p3.model.add_subsystem("aero_point_0", A3(surfaces=[dict(s) for s in S]))
    for k in flow:
        p3.model.connect(k, "aero_point_0." + k)
    conns(S, p3.model.connect)
    p3.setup()
    # om4
    cl = []
    conns(S, lambda a, b: cl.append(om.Connection(src=a, tgt=b)))
    subs = {s["name"]: G4(surface=dict(s)) for s in S}
    subs["aero_point_0"] = om.Subsystem(A4(surfaces=[dict(s) for s in S]), promotes_inputs=list(flow))
    p4 = om.Problem(model=om.Group(subsystems=subs, connections=cl))
    for k, (v, u) in flow.items():
        p4.set_val(k, v, units=u)
    p3.run_model(); p4.run_model()
    of = ["aero_point_0.CL", "aero_point_0.CD", "aero_point_0.CM"]
    vd = max(np.max(abs(np.ravel(p3.get_val(o)) - np.ravel(p4.get_val(o)))) for o in of)
    wrt4 = ["alpha", "v", "wing.twist_cp", "wing.chord_cp", "wing.sweep", "wing.t_over_c_cp"]
    wrt3 = ["alpha", "v", "wing.twist_cp", "wing.chord_cp", "wing.sweep", "wing.t_over_c_cp"]
    J3 = p3.compute_totals(of=of, wrt=wrt3, return_format="array")
    out = []
    for mode in ("fwd", "rev"):
        J4 = p4.compute_totals(of=of, wrt=wrt4, mode=mode).todense()
        out.append(f"{mode} {np.max(abs(J3 - J4)) / np.max(abs(J3)):.1e}")
    print(f"sym={sym!s:5} ground={ground!s:5} tail={tail!s:5} CL={p4.get_val('aero_point_0.CL')[0]:.6f}  values diff {vd:.1e}  totals rel diff: {', '.join(out)}")

