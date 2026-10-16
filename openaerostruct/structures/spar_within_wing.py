from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om
from om4.derivatives.differentiator import Differentiator
from om4.derivatives.complex_step import ComplexStep
from om4.utils.polymorphic import Polymorphic

from openaerostruct.structures.utils import radii
from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class SparWithinWing(om.ExplicitComponent):
    """
    Create a constraint to see if the spar is within the wing.
    This is based on the wing's t/c and the spar radius.

    .. warning::
        This component has not been extensively tested.
        It may require additional coding to work as intended.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Array defining the nodal points of the lifting surface.
    radius[ny-1] : numpy array
        Radius of each element of the FEM spar.
    t_over_c[ny-1] : numpy array
        The streamwise thickness-to-chord ratio of each VLM panel.

    Returns
    -------
    spar_within_wing[ny-1] : numpy array
        If all the values are negative, each element is within the wing,
        based on the surface's t_over_c value and the current chord.
        Add a constraint on 'spar_within_wing' with upper=0.
    """

    surface: Surface
    differentiator: Polymorphic[Differentiator] | None = ComplexStep()

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        nx, ny = f.surface["mesh"].shape[:2]
        d = VarDecl()

        d.add_input("mesh", val=np.zeros((nx, ny, 3)), units="m")
        d.add_input("radius", val=np.zeros((ny - 1)), units="m")
        d.add_input("t_over_c", val=np.zeros((ny - 1)))
        d.add_output("spar_within_wing", val=np.zeros((ny - 1)), units="m")

        d.declare_partials("*", "*")
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        max_radius = radii(inputs["mesh"], inputs["t_over_c"])
        outputs["spar_within_wing"] = inputs["radius"] - max_radius
