from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class NonIntersectingThickness(om.ExplicitComponent):
    """
    Create a constraint so the thickness of the spar does not intersect
    itself in the center of the spar. Basically, the thickness must be less
    than or equal to the radius.

    Parameters
    ----------
    thickness[ny-1] : numpy array
        Thickness of each element of the FEM spar.
    radius[ny-1] : numpy array
        Radius of each element of the FEM spar.

    Returns
    -------
    thickness_intersects[ny-1] : numpy array
        If all the values are negative, each element does not intersect itself.
        If a value is positive, then the thickness within the hollow spar
        intersects itself and presents an impossible design.
        Add a constraint on 'thickness_intersects' with upper=0.
    """

    surface: Surface

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        ny = f.surface["mesh"].shape[1]
        d = VarDecl()

        d.add_input("thickness", val=np.zeros((ny - 1)), units="m")
        d.add_input("radius", val=np.zeros((ny - 1)), units="m")
        d.add_output("thickness_intersects", val=np.zeros((ny - 1)), units="m")

        arange = np.arange(ny - 1)
        d.declare_partials("thickness_intersects", "thickness", rows=arange, cols=arange, val=1.0)
        d.declare_partials("thickness_intersects", "radius", rows=arange, cols=arange, val=-1.0)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["thickness_intersects"] = inputs["thickness"] - inputs["radius"]
