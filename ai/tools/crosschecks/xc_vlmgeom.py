import sys, numpy as np
sys.path.insert(0, "ai/tools")
from om3_crosscheck import om3_oas, crosscheck, random_inputs, report
om3_oas()
from oas_om3.aerodynamics.geometry import VLMGeometry as Old
from openaerostruct.aerodynamics.geometry import VLMGeometry as New
from openaerostruct.meshing.mesh_generator import generate_mesh
fails = 0
for sym in (True, False):
    for sref in ("wetted", "projected"):
        mesh, _ = generate_mesh({"num_y": 7, "num_x": 3, "wing_type": "CRM", "symmetry": sym, "num_twist_cp": 3})
        surf = {"name": "wing", "mesh": mesh, "symmetry": sym, "S_ref_type": sref}
        new = New(surface=surf)
        m = mesh + 0.05 * np.random.default_rng(1).random(mesh.shape)
        fails += report(f"VLMGeometry sym={sym} {sref}", crosscheck(Old(surface=surf), new, {"def_mesh": m}))
print("FAILURES", fails)
