from functools import cached_property
from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values, vlm_system_size
from openaerostruct.utils.surface import Surface
from scipy.sparse import csc_matrix



def _horseshoe_coo(surfaces):
    """COO entries of the linear map from vortex-ring to horseshoe circulations."""
    system_size = vlm_system_size(surfaces)

    # To convert between the two circulations, we simply need to set up a
    # matrix that linearly transforms the vortex ring circulations to
    # the horseshoe circulations. Again, because this is a linear
    # transformation, the derivatives are in fact the matrix itself.
    data = [np.ones(system_size)]
    rows = [np.arange(system_size)]
    cols = [np.arange(system_size)]

    ind_1 = 0
    ind_2 = 0
    for surface in surfaces:
        mesh = surface["mesh"]
        nx = mesh.shape[0]
        ny = mesh.shape[1]
        num = (nx - 1) * (ny - 1)

        ind_2 += num

        arange = np.arange(num).reshape((nx - 1), (ny - 1))

        data_ = -np.ones((nx - 2) * (ny - 1))
        rows_ = ind_1 + arange[1:, :].flatten()
        cols_ = ind_1 + arange[:-1, :].flatten()

        data.append(data_)
        rows.append(rows_)
        cols.append(cols_)

        ind_1 += num

    data = np.concatenate(data)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    return data, rows, cols


class HorseshoeCirculations(om.ExplicitComponent):
    """
    Convert the previously-computed vortex ring circulations into horseshoe
    circulations. Vortex rings and horseshoe vortices produce the same linear
    space, but with a different parameterization. It's easier to compute the
    circulations using a vortex ring approach, but it's easier to compute the
    forces acting on the surface by using the horseshoe circulations.
    That's why we have this component, to convert from one circulation
    space to the other.

    Parameters
    ----------
    circulations[system_size] : numpy array
        The vortex ring circulations obtained by solving the AIC linear system.

    Returns
    -------
    horseshoe_circulations[system_size] : numpy array
        The equivalent horseshoe circulations obtained by intelligently summing
        the vortex ring circulations, accounting for overlaps between rings.
    """

    surfaces: list[Surface]

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = vars(field_values(cls, data))
        d = VarDecl()
        _spec = data  # the setup code below may rebind `data`

        surfaces = f["surfaces"]

        system_size = 0

        # Loop through all the surfaces to obtain the total system size,
        # which is the number of panels in the total system.
        for surface in surfaces:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]

            system_size += (nx - 1) * (ny - 1)

        d.add_input("circulations", shape=system_size, units="m**2/s", tags=["mphys_coupling"])
        d.add_output("horseshoe_circulations", shape=system_size, units="m**2/s")

        data, rows, cols = _horseshoe_coo(surfaces)

        d.declare_partials("horseshoe_circulations", "circulations", val=data, rows=rows, cols=cols)
        return d.into(_spec)

    @cached_property
    def _mtx(self):
        """Sparse vortex-ring -> horseshoe circulation map (constant)."""
        size = vlm_system_size(self.surfaces)
        data, rows, cols = _horseshoe_coo(self.surfaces)
        return csc_matrix((data, (rows, cols)), shape=(size, size))

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["horseshoe_circulations"] = self._mtx.dot(inputs["circulations"])
