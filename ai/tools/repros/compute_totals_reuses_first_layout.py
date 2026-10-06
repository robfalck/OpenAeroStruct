import numpy as np, om4.api as om
p = om.Problem(model=om.Group(subsystems={"c": om.ExecComp(exprs=["y = 2*x + 3*z"], inputs={"x": om.InputVar(val=1.0), "z": om.InputVar(val=1.0)})}))
p.run_model()
print("wrt=[x, z]:", p.compute_totals(of=["c.y"], wrt=["c.x", "c.z"]).todense())
print("wrt=[z]   :", p.compute_totals(of=["c.y"], wrt=["c.z"]).todense(), "  (expected [[3.]])")
print("wrt=[x]   :", p.compute_totals(of=["c.y"], wrt=["c.x"]).todense(), "  (expected [[2.]])")
