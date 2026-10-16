from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class Disp(om.ExplicitComponent):
    """
    Reshape the flattened displacements from the linear system solution into
    a 2D array so we can more easily use the results.

    The solution to the linear system has meaingless entires due to the
    constraints on the FEM model. The displacements from this portion of
    the linear system are not needed, so we select only the relevant
    portion of the displacements for further calculations.

    Parameters
    ----------
    disp_aug[6*(ny+1)] : numpy array
        Augmented displacement array. Obtained by solving the system
        K * disp_aug = forces, where forces is a flattened version of loads.

    Returns
    -------
    disp[6*ny] : numpy array
        Actual displacement array formed by truncating disp_aug.

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

        d.add_input("disp_aug", val=np.zeros(((ny + 1) * 6)), units="m")
        d.add_output("disp", val=np.zeros((ny, 6)), units="m")

        n = ny * 6
        arange = np.arange((n))
        d.declare_partials("disp", "disp_aug", val=1.0, rows=arange, cols=arange)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Obtain the relevant portions of disp_aug and store the reshaped
        # displacements in disp
        outputs["disp"] = inputs["disp_aug"][:-6].reshape((-1, 6))
