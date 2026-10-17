"""
B-009: DesignVar.units must apply to total derivatives, not only to values and bounds.

y = a**2 with `a` in deg, evaluated at a = 114.6 deg. With the DV declared in rad the total must be
dy/da_rad = 2 * a_deg * (180/pi) ~ 13131; per promoted-input unit it would be dy/da_deg = 2 * a_deg = 229.2.
Fixed on om4 main (86c489ea, issue #370).
"""

import numpy as np

import om4.api as om
from om4.variables.design_var import DesignVar


class Sq(om.ExplicitComponent):
    inputs: dict[str, om.InputVar] = {"a": om.InputVar(shape=(1,), units="deg")}
    outputs: dict[str, om.OutputVar] = {"y": om.OutputVar(shape=(1,), units=None)}
    partials: list[om.PartialsSpec] = [om.PartialsSpec(of="y", wrt="a")]

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["y"] = inputs["a"] ** 2

    def compute_partials(self, inputs, partials):
        partials["y", "a"] = 2.0 * inputs["a"]


a_deg = 114.6
p = om.Problem(
    model=om.Group(subsystems={"c": Sq()}, design_vars={"c.a": DesignVar(units="rad")}),
)
p.set_val("c.a", a_deg, units="deg")
p.run_model()
for mode in ("fwd", "rev"):
    J = float(np.ravel(np.asarray(p.compute_totals(of=["c.y"], wrt=["c.a"], mode=mode)["c.y", "c.a"]))[0])
    print(
        f"{mode}: dy/da = {J:.2f}  (expected {2 * a_deg * 180 / np.pi:.2f} per rad; {2 * a_deg:.2f} would be per deg)"
    )
