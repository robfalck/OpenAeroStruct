from functools import cached_property
from typing import Any

import numpy as np
from pydantic import PrivateAttr, model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values, vlm_system_size
from openaerostruct.utils.surface import Surface
from scipy.linalg import lu_factor, lu_solve



class SolveMatrix(om.ImplicitComponent):
    """
    Solve the AIC linear system to obtain the vortex ring circulations.

    Parameters
    ----------
    mtx[system_size, system_size] : numpy array
        Final fully assembled AIC matrix that is used to solve for the
        circulations.
    rhs[system_size] : numpy array
        Right-hand side of the AIC linear system, constructed from the
        freestream velocities and panel normals.

    Returns
    -------
    circulations[system_size] : numpy array
        The vortex ring circulations obtained by solving the AIC linear system.

    """

    surfaces: list[Surface]

    _lu: tuple | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = vars(field_values(cls, data))
        d = VarDecl()
        _spec = data  # the setup code below may rebind `data`

        system_size = 0

        for surface in f["surfaces"]:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]

            system_size += (nx - 1) * (ny - 1)

        d.add_input("mtx", shape=(system_size, system_size), units="1/m")
        d.add_input("rhs", shape=system_size, units="m/s")
        d.add_output("circulations", shape=system_size, units="m**2/s", tags=["mphys_coupling"])

        d.declare_partials(
            "circulations",
            "circulations",
            rows=np.outer(np.arange(system_size), np.ones(system_size, int)).flatten(),
            cols=np.outer(np.ones(system_size, int), np.arange(system_size)).flatten(),
        )
        d.declare_partials(
            "circulations",
            "mtx",
            rows=np.outer(np.arange(system_size), np.ones(system_size, int)).flatten(),
            cols=np.arange(system_size**2),
        )
        d.declare_partials(
            "circulations",
            "rhs",
            val=-1.0,
            rows=np.arange(system_size),
            cols=np.arange(system_size),
        )
        return d.into(_spec)

    @cached_property
    def system_size(self) -> int:
        """Total number of VLM panels over all surfaces."""
        return vlm_system_size(self.surfaces)

    def compute_residuals(self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None):
        residuals["circulations"] = inputs["mtx"].dot(outputs["circulations"]) - inputs["rhs"]

    def solve_nonlinear(self, inputs, outputs):
        self._lu = lu_factor(inputs["mtx"])

        outputs["circulations"] = lu_solve(self._lu, inputs["rhs"])

    def compute_partials(self, inputs, outputs, partials):
        system_size = self.system_size
        self._lu = lu_factor(inputs["mtx"])

        partials["circulations", "circulations"] = inputs["mtx"].flatten()
        partials["circulations", "mtx"] = np.outer(np.ones(system_size), outputs["circulations"]).flatten()

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == "fwd":
            d_outputs["circulations"] = lu_solve(self._lu, d_residuals["circulations"], trans=0)
        else:
            d_residuals["circulations"] = lu_solve(self._lu, d_outputs["circulations"], trans=1)
