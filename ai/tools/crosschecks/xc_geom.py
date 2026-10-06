import sys, numpy as np
sys.path.insert(0, "ai/tools")
import openmdao.api as om3, om4.api as om
from om3_crosscheck import om3_oas
om3_oas()
from oas_om3.geometry.geometry_group import Geometry as G3
from openaerostruct.geometry.geometry_group import Geometry as G4
from openaerostruct.meshing.mesh_generator import generate_mesh
for sym in (True, False):
    mesh, tw = generate_mesh({"num_y": 9, "num_x": 3, "wing_type": "CRM", "symmetry": sym, "num_twist_cp": 5})
    surf = {"name": "wing", "mesh": mesh, "symmetry": sym, "twist_cp": tw, "chord_cp": np.array([1.0, 0.9, 1.1]),
            "t_over_c_cp": np.array([0.15, 0.1]), "sweep": 5.0, "taper": 0.7, "xshear_cp": np.zeros(2)}
    p3 = om3.Problem(reports=False); p3.model.add_subsystem("wing", G3(surface=dict(surf))); p3.setup()
    p4 = om.Problem(model=om.Group(subsystems={"wing": G4(surface=dict(surf))}))
    vals = {"wing.twist_cp": tw + 1.0, "wing.chord_cp": np.array([1.1, 0.8, 1.2]), "wing.sweep": 9.0,
            "wing.t_over_c_cp": np.array([0.12, 0.09]), "wing.xshear_cp": np.array([0.0, 0.5])}
    for k, v in vals.items():
        p3.set_val(k, v); p4.set_val(k, v)
    p3.run_model(); p4.run_model()
    of = ["wing.mesh", "wing.t_over_c"]
    J3 = p3.compute_totals(of=of, wrt=list(vals), return_format="array")
    J4 = p4.compute_totals(of=of, wrt=list(vals)).todense()
    print(f"sym={sym}: mesh {np.max(abs(p3.get_val('wing.mesh') - p4.get_val('wing.mesh'))):.1e}  "
          f"t/c {np.max(abs(p3.get_val('wing.t_over_c').ravel() - p4.get_val('wing.t_over_c'))):.1e}  totals {np.max(abs(J3 - J4)):.1e}")
