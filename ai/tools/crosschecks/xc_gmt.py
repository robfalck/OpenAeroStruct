import sys, numpy as np
sys.path.insert(0, "ai/tools")
from om3_crosscheck import load_om3_module, crosscheck
from openaerostruct.meshing.mesh_generator import generate_mesh
import openaerostruct.geometry.geometry_mesh_transformations as new
old = load_om3_module("openaerostruct/geometry/geometry_mesh_transformations.py")
rng = np.random.default_rng(0)
fails = 0
for sym in (True, False):
    mesh, _ = generate_mesh({"num_y": 7, "num_x": 3, "wing_type": "CRM", "symmetry": sym, "num_twist_cp": 3})
    mesh = mesh + 0.01 * rng.random(mesh.shape)
    ms = mesh.shape; ny = ms[1]
    cases = {
        "Taper": (dict(val=0.7, mesh=mesh, symmetry=sym, ref_axis_pos=0.3), {"taper": 0.6}),
        "ScaleX": (dict(val=np.ones(ny), mesh_shape=ms, ref_axis_pos=0.3), {"chord": rng.random(ny) + 0.5, "in_mesh": mesh}),
        "Sweep": (dict(val=5.0, mesh_shape=ms, symmetry=sym), {"sweep": 12.0, "in_mesh": mesh}),
        "Dihedral": (dict(val=5.0, mesh_shape=ms, symmetry=sym), {"dihedral": 7.0, "in_mesh": mesh}),
        "ShearX": (dict(val=np.zeros(ny), mesh_shape=ms), {"xshear": rng.random(ny), "in_mesh": mesh}),
        "ShearY": (dict(val=np.zeros(ny), mesh_shape=ms), {"yshear": rng.random(ny), "in_mesh": mesh}),
        "ShearZ": (dict(val=np.zeros(ny), mesh_shape=ms), {"zshear": rng.random(ny), "in_mesh": mesh}),
        "Stretch": (dict(val=30.0, mesh_shape=ms, symmetry=sym, ref_axis_pos=0.3), {"span": 40.0, "in_mesh": mesh}),
        "Rotate": (dict(val=np.zeros(ny), mesh_shape=ms, symmetry=sym, ref_axis_pos=0.3), {"twist": rng.random(ny) * 5, "in_mesh": mesh}),
        "Rotate_nox": (dict(val=np.zeros(ny), mesh_shape=ms, symmetry=sym, rotate_x=False), {"twist": rng.random(ny) * 5, "in_mesh": mesh}),
    }
    for name, (opts, ins) in cases.items():
        cname = name.split("_")[0]
        probs = crosscheck(getattr(old, cname)(**opts), getattr(new, cname)(**opts), ins)
        fails += bool(probs)
        print(f"sym={sym!s:5} {name:10} {'OK' if not probs else probs}")
print("FAILURES", fails)
