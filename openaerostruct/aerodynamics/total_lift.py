from typing import Any

from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl
from openaerostruct.utils.surface import Surface


class TotalLift(om.ExplicitComponent):
    """
    Calculate total lift in force units by summing the induced CL
    with the CL0.

    Parameters
    ----------
    CL1 : float
        Induced coefficient of lift (CL) for the lifting surface.

    Returns
    -------
    CL : float
        Total coefficient of lift (CL) for the lifting surface.
    """

    surface: Surface

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = VarDecl()
        _spec = data  # the setup code below may rebind `data`

        d.add_input("CL1", val=1.0)

        d.add_output("CL", val=1.0, tags=["mphys_result"])

        d.declare_partials("CL", "CL1", val=1.0)
        return d.into(_spec)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["CL"] = inputs["CL1"] + self.surface["CL0"]
