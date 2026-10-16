from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


def get_fem_origin(surface):
    """Return the chordwise location of the FEM nodes as a fraction of the chord."""
    if surface["fem_model_type"].lower() == "tube":
        return surface["fem_origin"]
    y_upper = surface["data_y_upper"]
    x_upper = surface["data_x_upper"]
    y_lower = surface["data_y_lower"]

    return (x_upper[0] * (y_upper[0] - y_lower[0]) + x_upper[-1] * (y_upper[-1] - y_lower[-1])) / (
        (y_upper[0] - y_lower[0]) + (y_upper[-1] - y_lower[-1])
    )


class ComputeNodes(om.ExplicitComponent):
    """
    Compute FEM nodes based on aerodynamic mesh.

    The FEM nodes are placed at fem_origin * chord,
    with the default fem_origin = 0.35.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Array defining the nodal points of the lifting surface.

    Returns
    -------
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.

    """

    surface: Surface

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        surface = f.surface
        nx, ny = surface["mesh"].shape[:2]
        d = VarDecl()

        w = get_fem_origin(surface)

        d.add_input("mesh", val=np.zeros((nx, ny, 3)), units="m")
        d.add_output("nodes", val=np.zeros((ny, 3)), units="m")

        n = ny * 3

        vals = np.zeros((2 * n))
        vals[:n] = 1 - w
        vals[n:] = w

        arange = np.arange(n)

        rows = np.hstack((arange, arange))
        cols = np.hstack((arange, arange + (nx - 1) * n))

        d.declare_partials("nodes", "mesh", rows=rows, cols=cols, val=vals)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        w = get_fem_origin(self.surface)
        mesh = inputs["mesh"]
        outputs["nodes"] = (1 - w) * mesh[0, :, :] + w * mesh[-1, :, :]
