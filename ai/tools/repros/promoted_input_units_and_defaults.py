import numpy as np, om4.api as om
from typing import Any
from pydantic import model_validator

class Echo(om.ExplicitComponent):
    """y = a, with `a` declared in `unit`."""
    unit: str
    val: float = 1.0
    @model_validator(mode="before")
    @classmethod
    def _b(cls, data: Any):
        data["inputs"] = {"a": om.InputVar(val=np.array([data.get("val", 1.0)]), units=data["unit"])}
        data["outputs"] = {"y": om.OutputVar(val=np.zeros(1), units=data["unit"])}
        data["partials"] = [om.PartialsSpec(of="y", wrt="a", rows=[0], cols=[0], val=[1.0])]
        return data
    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["y"] = inputs["a"]

def model(**kw):
    return om.Problem(model=om.Group(subsystems={
        "c_deg": om.Subsystem(Echo(unit="deg", val=0.0), promotes_inputs=["a"]),
        "c_rad": om.Subsystem(Echo(unit="rad", val=1.0), promotes_inputs=["a"])}, **kw))

try:
    p = model()
    p.set_val("a", 5.0)
    p.run_model()
    print("A) no InputDefault, set_val('a', 5.0):  c_deg.y =", p.get_val("c_deg.y"), "deg;  c_rad.y =", p.get_val("c_rad.y"), "rad   (expected a setup error)")
except RuntimeError as err:
    print("A) no InputDefault: setup error, as expected:", str(err).splitlines()[0])

p = model(input_defaults={"a": om.InputDefault(val=5.0, units="deg")})
p.run_model()
print("B) InputDefault(5 deg), no set_val:     c_deg.y =", p.get_val("c_deg.y"), "deg;  c_rad.y =", p.get_val("c_rad.y"), "rad;  get_val('a') =", p.get_val("a"), "  (expected 5 deg / 0.0873 rad)")

p = model(input_defaults={"a": om.InputDefault(val=5.0, units="deg")})
p.set_val("a", 7.0)
p.run_model()
print("C) InputDefault(5 deg), set_val(a, 7):  c_deg.y =", p.get_val("c_deg.y"), "deg;  c_rad.y =", p.get_val("c_rad.y"), "rad   (expected 7 deg / 0.1222 rad)")

# D) Totals wrt a promoted input whose leaves have different units.
p = model(input_defaults={"a": om.InputDefault(val=5.0, units="deg")})
p.set_val("a", 7.0)
p.run_model()
for mode in ("fwd", "rev"):
    J = p.compute_totals(of=["c_deg.y", "c_rad.y"], wrt=["a"], mode=mode).todense().ravel()
    print(f"D) d[c_deg.y, c_rad.y]/da ({mode}) =", J, "  (expected [1, 0.01745329] in deg-based 'a')")
