from typing import Any

from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class SumAreas(om.ExplicitComponent):
    """
    Compute the total surface area of the entire aircraft as a sum of its
    individual surfaces' surface areas.

    Parameters
    ----------
    S_ref : float
        Surface area for one lifting surface.

    Returns
    -------
    S_ref_total : float
        Total surface area of the aircraft based on the sum of individual
        surface areas.

    """

    surfaces: list[Surface]

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = vars(field_values(cls, data))
        d = VarDecl()
        _spec = data  # the setup code below may rebind `data`

        for surface in f["surfaces"]:
            name = surface["name"]
            d.add_input(name + "_S_ref", val=1.0, units="m**2")

        d.add_output("S_ref_total", val=0.0, units="m**2", tags=["mphys_result"])

        d.declare_partials("*", "*", val=1.0)
        return d.into(_spec)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["S_ref_total"] = 0.0
        for surface in self.surfaces:
            name = surface["name"]
            S_ref = inputs[name + "_S_ref"]
            outputs["S_ref_total"] += S_ref
