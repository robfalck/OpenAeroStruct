from typing import Any

from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl


class ReynoldsComp(om.ExplicitComponent):
    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = VarDecl()
        _spec = data

        d.add_input("rho", val=1.0, units="slug/ft**3")
        d.add_input("mu", val=1.0, units="lbf*s/ft**2")
        d.add_input("v", val=1.0, units="ft/s")

        d.add_output("re", val=1.0, units="1/ft")

        d.declare_partials("re", ["rho", "mu", "v"])
        return d.into(_spec)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["re"] = inputs["rho"] * inputs["v"] / inputs["mu"]

    def compute_partials(self, inputs, partials):
        partials["re", "rho"] = inputs["v"] / inputs["mu"]
        partials["re", "v"] = inputs["rho"] / inputs["mu"]
        partials["re", "mu"] = -inputs["rho"] * inputs["v"] / inputs["mu"] ** 2
