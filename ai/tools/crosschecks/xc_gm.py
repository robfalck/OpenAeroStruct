import sys, numpy as np
sys.path.insert(0, "ai/tools")
import openmdao.api as om3, om4.api as om
from om3_crosscheck import load_om3_module
from openaerostruct.geometry.geometry_mesh import GeometryMesh
from openaerostruct.meshing.mesh_generator import generate_mesh
# OM3 GeometryMesh imports the (now om4) transformations by package path, so load both OM3 modules
old_t = load_om3_module("openaerostruct/geometry/geometry_mesh_transformations.py")
sys.modules["openaerostruct.geometry.geometry_mesh_transformations_om3"] = old_t
import subprocess, tempfile, importlib.util, pathlib
src = subprocess.run(["git", "show", "main:openaerostruct/geometry/geometry_mesh.py"], capture_output=True, text=True).stdout
src = src.replace("openaerostruct.geometry.geometry_mesh_transformations", "openaerostruct.geometry.geometry_mesh_transformations_om3")
tmp = pathlib.Path(tempfile.mkdtemp()) / "gm_om3.py"; tmp.write_text(src)
spec = importlib.util.spec_from_file_location("gm_om3", tmp); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
for sym in (True, False):
    mesh, tw = generate_mesh({"num_y": 7, "num_x": 3, "wing_type": "CRM", "symmetry": sym, "num_twist_cp": 3})
    ny = mesh.shape[1]
    surf = {"name": "wing", "mesh": mesh, "symmetry": sym, "twist_cp": tw, "chord_cp": np.ones(2), "sweep": 5.0,
            "span": 40.0, "dihedral": 3.0, "taper": 0.6, "xshear_cp": np.zeros(2), "zshear_cp": np.zeros(2)}
    vals = {"twist": np.linspace(-3, 2, ny), "chord": np.linspace(1.2, 0.8, ny), "sweep": 10.0, "span": 45.0,
            "dihedral": 4.0, "taper": 0.5, "xshear": np.linspace(0, 1, ny), "zshear": np.linspace(0, 0.3, ny)}
    p3 = om3.Problem(reports=False); p3.model.add_subsystem("g", old.GeometryMesh(surface=surf), promotes_inputs=["*"]); p3.setup()
    p4 = om.Problem(model=om.Group(subsystems={"g": om.Subsystem(GeometryMesh(surface=surf), promotes_inputs=["*"])}))
    for k, v in vals.items():
        p3.set_val(k, v); p4.set_val(k, v)
    p3.run_model(); p4.run_model()
    wrt = list(vals)
    t3 = p3.compute_totals(of=["g.mesh"], wrt=wrt)
    J4 = p4.compute_totals(of=["g.mesh"], wrt=wrt).todense()
    J3 = np.hstack([np.asarray(t3["g.mesh", w]).reshape(mesh.size, -1) for w in wrt])
    print(f"sym={sym}: mesh diff {np.max(abs(p3.get_val('g.mesh') - p4.get_val('g.mesh'))):.2e}, totals diff {np.max(abs(J3 - J4)):.2e}")
