import sys, numpy as np
sys.path.insert(0, "ai/tools")
import openmdao.api as om3, om4.api as om
from om3_crosscheck import om3_oas
om3_oas()
from oas_om3.aerodynamics.states import VLMStates as O
from openaerostruct.aerodynamics.states import VLMStates as N
from openaerostruct.meshing.mesh_generator import generate_mesh
for sym in (True, False):
    m1, _ = generate_mesh({"num_y": 7, "num_x": 3, "wing_type": "CRM", "symmetry": sym, "num_twist_cp": 3})
    m2 = generate_mesh({"num_y": 5, "num_x": 2, "wing_type": "rect", "symmetry": sym}) + np.array([30.0, 0, 2.0])
    S = [{"name": "wing", "mesh": m1, "symmetry": sym}, {"name": "tail", "mesh": m2, "symmetry": sym}]
    p3 = om3.Problem(reports=False); p3.model.add_subsystem("s", O(surfaces=S), promotes=["*"]); p3.model.set_input_defaults("alpha", 5.0, units="deg"); p3.setup()
    p4 = om.Problem(model=om.Group(subsystems={"s": om.Subsystem(N(surfaces=S), promotes_inputs=["*"], promotes_outputs=["*"])}))
    vals = {"v": 248.0, "alpha": 5.0, "beta": 1.0, "rho": 0.38, "wing_normals": None}
    for s in S:
        nx, ny = s["mesh"].shape[:2]
        # unit normals from the mesh (as VLMGeometry would supply)
        nrm = np.zeros((nx - 1, ny - 1, 3)); nrm[..., 2] = 1.0
        vals[s["name"] + "_def_mesh"] = s["mesh"]
        vals[s["name"] + "_normals"] = nrm
    vals.pop("wing_normals") if vals["wing_normals"] is None else None
    for k, v in vals.items():
        p3.set_val(k, v); p4.set_val(k, v)
    p3.run_model(); p4.run_model()
    of = ["circulations", "wing_sec_forces", "tail_sec_forces"]
    wrt = ["alpha", "v", "wing_def_mesh", "tail_def_mesh"]
    d = max(np.max(abs(np.ravel(p3.get_val(o)) - np.ravel(p4.get_val(o)))) for o in of)
    J3 = p3.compute_totals(of=of, wrt=wrt, return_format="array")
    for mode in ("fwd", "rev"):
        J4 = p4.compute_totals(of=of, wrt=wrt, mode=mode).todense()
        print(f"sym={sym} {mode}: values {d:.2e}, totals rel {np.max(abs(J3 - J4)) / np.max(abs(J3)):.2e}")
