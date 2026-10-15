from typing import Any

from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values


class MultiCD(om.ExplicitComponent):
    """Sum the drag coefficients of `n_points` flight points."""

    n_points: int

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = vars(field_values(cls, data))
        d = VarDecl()
        _spec = data

        for i in range(f["n_points"]):
            d.add_input(str(i) + "_CD", val=0.0, units=None)

        d.add_output("CD", val=0.0, units=None)
        d.declare_partials("*", "*", val=1.0)
        return d.into(_spec)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        CD = 0.0
        for i in range(self.n_points):
            CD = CD + inputs[str(i) + "_CD"]
        outputs["CD"] = CD
