from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om
from om4.derivatives.complex_step import ComplexStep
from om4.derivatives.differentiator import Differentiator
from om4.utils.polymorphic import Polymorphic

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


def norm(vec):
    return np.sqrt(np.sum(vec**2))


class WingboxFuelVol(om.ExplicitComponent):
    """
    Computes the internal volumes of the wingbox segments.

    Parameters
    ----------
    nodes[ny, 3] : numpy array
        Coordinates of FEM nodes.
    A_int[ny-1] : numpy array
        Internal cross-sectional area of each wingbox segment.

    Returns
    -------
    fuel_vols[ny-1] : numpy array
        Internal volume of each wingbox segment.
    """

    surface: Surface
    differentiator: Polymorphic[Differentiator] | None = ComplexStep()

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        ny = f.surface["mesh"].shape[1]
        d = VarDecl()

        d.add_input("nodes", val=np.zeros((ny, 3)), units="m")
        d.add_input("A_int", val=np.zeros((ny - 1)), units="m**2")
        d.add_output("fuel_vols", val=np.zeros((ny - 1)), units="m**3")

        d.declare_partials("*", "*")
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nodes = inputs["nodes"]
        ny = nodes.shape[0]

        element_lengths = np.zeros(ny - 1, dtype=type(nodes[0, 0]))

        for i in range(ny - 1):
            element_lengths[i] = norm(nodes[i + 1] - nodes[i])

        # Next we multiply the element lengths with the A_int for the internal volumes of the wingobox segments
        vols = element_lengths * inputs["A_int"]

        outputs["fuel_vols"] = vols
