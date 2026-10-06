from typing import Any

from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl
from openaerostruct.utils.surface import Surface


class TotalDrag(om.ExplicitComponent):
    """Calculate total drag in force units.

    Parameters
    ----------
    CDi : float
        Induced coefficient of drag (CD) for the lifting surface.
    CDv : float
        Calculated coefficient of viscous drag for the lifting surface.

    Returns
    -------
    CD : float
        Total coefficient of drag (CD) for the lifting surface.
    """

    surface: Surface

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = VarDecl()
        _spec = data  # the setup code below may rebind `data`

        d.add_input("CDi", val=1.0)
        d.add_input("CDv", val=1.0)
        d.add_input("CDw", val=1.0)

        d.add_output("CD", val=1.0, tags=["mphys_result"])

        d.declare_partials("CD", "CDi", val=1.0)
        d.declare_partials("CD", "CDv", val=1.0)
        d.declare_partials("CD", "CDw", val=1.0)
        return d.into(_spec)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["CD"] = inputs["CDi"] + inputs["CDv"] + inputs["CDw"] + self.surface["CD0"]
