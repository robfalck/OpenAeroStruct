from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class CreateRHS(om.ExplicitComponent):
    """
    Compute the right-hand-side of the K * u = f linear system to solve for the displacements.
    The RHS is based on the loads. For the aerostructural case, these are
    recomputed at each design point based on the aerodynamic loads.

    Parameters
    ----------
    loads[ny, 6] : numpy array
        Flattened array containing the loads applied on the FEM component,
        computed from the sectional forces.

    Returns
    -------
    forces[6*(ny+1)] : numpy array
        Right-hand-side of the linear system. The loads from the aerodynamic
        analysis or the user-defined loads.
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

        d.add_input("total_loads", val=np.zeros((ny, 6)), units="N")
        d.add_output("forces", val=np.ones(((ny + 1) * 6)), units="N")

        n = ny * 6
        arange = np.arange((n))
        d.declare_partials("forces", "total_loads", val=1.0, rows=arange, cols=arange)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ny = self.surface["mesh"].shape[1]
        forces = np.zeros((ny + 1) * 6, dtype=inputs["total_loads"].dtype)

        # Populate the right-hand side of the linear system using the
        # prescribed or computed loads
        forces[: 6 * ny] += inputs["total_loads"].reshape(ny * 6)

        # Remove extremely small values from the RHS so the linear system
        # can more easily be solved
        forces[np.abs(forces) < 1e-6] = 0.0
        outputs["forces"] = forces
