from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class Energy(om.ExplicitComponent):
    """Compute strain energy.

    Parameters
    ----------
    disp[ny, 6] : numpy array
        Actual displacement array formed by truncating disp_aug.
    loads[ny, 6] : numpy array
        Array containing the loads applied on the FEM component,
        computed from the sectional forces.

    Returns
    -------
    energy : float
        Total strain energy of the structural component.

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

        d.add_input("disp", val=np.zeros((ny, 6)), units="m")
        d.add_input("loads", val=np.zeros((ny, 6)), units="N")
        d.add_output("energy", val=0.0, units="N*m")

        d.declare_partials("*", "*")
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["energy"] = np.sum(inputs["disp"] * inputs["loads"])

    def compute_partials(self, inputs, partials):
        partials["energy", "disp"] = inputs["loads"].real.reshape(1, -1)
        partials["energy", "loads"] = inputs["disp"].real.reshape(1, -1)
